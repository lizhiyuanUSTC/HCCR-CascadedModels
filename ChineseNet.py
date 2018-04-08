import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        expand_1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        expand_3x3 = slim.conv2d(inputs, num_outputs, [3, 3], stride=1, scope='3x3')
    return tf.concat([expand_1x1, expand_3x3], 3)


def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        net = squeeze(inputs, squeeze_depth)
        net = expand(net, expand_depth)
    return net


def prelu():
    def op(inputs):
        alpha = tf.get_variable(name='alpha', shape=(), initializer=tf.constant_initializer(value=0.25))
        return tf.maximum(alpha * inputs, inputs)
    return op


def print_info():
    total_params = 0
    for var in tf.trainable_variables():
        params = 1
        for dim in var.shape:
            params *= int(dim)
        total_params += params
    print('====================================================')
    print('%15s:' % 'total_params', total_params)


def global_conv(inputs, scope):
    with tf.variable_scope(scope):
        weights = tf.get_variable('weights', shape=[1, inputs.shape[1], inputs.shape[2], inputs.shape[3]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        global_conv_result = tf.reduce_sum(tf.reduce_sum(tf.multiply(inputs, weights), axis=1), axis=1)
        return prelu()(slim.batch_norm(global_conv_result,
                                       decay=0.995,
                                       epsilon=0.001,
                                       updates_collections=None,
                                       variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES],
                                       scale=True,
                                       is_training=False))


def get_mid_output(inputs, squeeze_num, scope):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected]):
            fire = fire_module(inputs, squeeze_num, squeeze_num*4, scope='fire')
            print(scope+'/fire:', fire.shape[1:])
            pre_logits = global_conv(fire, scope='pre_logits')
            print(scope+'/pre_logits:', pre_logits.shape[1:])
            logits = slim.fully_connected(pre_logits, 3755, activation_fn=None, normalizer_fn=None, scope='logits')
    return logits


def get_final_output(inputs):
    with tf.variable_scope('final'):
        pre_logits = global_conv(inputs, scope='pre_logits')
        print('final/pre_logits:', pre_logits.shape[1:])
        logits = slim.fully_connected(pre_logits, 3755, activation_fn=None, normalizer_fn=None, scope='logits')
    return logits


def squeezenet(images, th=1.0):
    final_count = tf.get_variable('final_count', shape=(), trainable=False, initializer=tf.zeros_initializer)
    conv1 = slim.conv2d(images, 64, [3, 3], stride=1, scope='conv1')
    print('conv1:', conv1.shape[1:])

    pool1 = slim.max_pool2d(conv1, [2, 2], [2, 2], padding='SAME')
    print('pool1:', pool1.shape[1:])

    fire2 = fire_module(pool1, 16, 64, scope='fire2')
    print('fire2:', fire2.shape[1:])

    fire3 = fire_module(fire2, 16, 64, scope='fire3')
    print('fire3:', fire3.shape[1:])

    pool3 = slim.max_pool2d(fire3, [2, 2], [2, 2], padding='SAME')
    print('pool3:', pool3.shape[1:])

    fire4 = fire_module(pool3, 32, 128, scope='fire4')
    print('fire4:', fire4.shape[1:])

    logits_A = get_mid_output(fire4, 32, scope='mid-A')
    prob_A = tf.nn.softmax(logits_A, axis=1)

    def mid_A():
        final_count_op = tf.assign_add(final_count, 0)
        return prob_A, final_count_op


    def final():
        final_count_op = tf.assign_add(final_count, 1)
        fire5 = fire_module(fire4, 32, 128, scope='fire5')
        print('fire5:', fire5.shape[1:])

        pool5 = slim.max_pool2d(fire5, [2, 2], [2, 2], padding='SAME')
        print('pool5:', pool5.shape[1:])

        fire6 = fire_module(pool5, 48, 192, scope='fire6')
        print('fire6:', fire6.shape[1:])

        logits_B = get_mid_output(fire6, 48, scope='mid-B')
        prob_B = tf.nn.softmax(logits_B, axis=1)

        fire7 = fire_module(fire6, 48, 192, scope='fire7')
        print('fire7:', fire7.shape[1:])

        fire8 = fire_module(fire7, 64, 256, scope='fire8')
        print('fire8:', fire8.shape[1:])

        fire9 = fire_module(fire8, 64, 256, scope='fire9')
        print('fire9:', fire9.shape[1:])

        logits = get_final_output(fire9)
        prob = tf.nn.softmax(logits, axis=1)
        return 0.3 * prob_B + 0.7 * prob, final_count_op

    return tf.cond(tf.greater(tf.reduce_max(prob_A), th), mid_A, final)



def inference(images, th=1.0):
    batch_norm_parameters = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        'scale': True,
        'is_training': False}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=prelu(),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_parameters):
        return squeezenet(images, th)


def load_weights(sess):
    print('***Loading weights***')
    f = open('tmp/weights.bin', 'rb')
    for var in tf.trainable_variables():
        shape = [int(i) for i in var.shape]
        var_name = var.name
        len = 1
        for i in shape:
            len *= i
        if 'alpha' in var_name:
            weights = np.fromfile(f, dtype=np.float32, count=1)[0]
        elif 'weights:0' in var_name:
            min_w = np.fromfile(f, dtype=np.float32, count=1)[0]
            step = np.fromfile(f, dtype=np.float64, count=1)[0]

            if '/logits/weights:0' in var_name:
                weights = np.fromfile(f, dtype=np.uint8, count=int(len/2))
                weights_1 = weights % 16
                weights_0 = np.array(weights / 16, dtype=np.uint8)
                weights = np.concatenate([weights_0, weights_1], axis=0)
            else:
                weights = np.fromfile(f, dtype=np.uint8, count=len)
            weights = np.array(weights, np.float64) * step + min_w
            weights = np.array(weights, np.float32)
            weights = np.reshape(weights, shape)
        else:
            weights = np.fromfile(f, dtype=np.float32, count=shape[-1])
        sess.run(tf.assign(var, weights))
    f.close()
