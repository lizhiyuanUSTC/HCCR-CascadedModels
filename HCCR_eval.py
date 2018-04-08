from datetime import datetime
import time
import os
import numpy as np
import tensorflow as tf

import ChineseNet

eval_num = 224419
#eval_num = 3000


def evaluate():
    print('***Loading data***')
    images_array = np.load('tmp/images.npy')
    labels_array = np.load('tmp/labels.npy')
    with tf.Graph().as_default():
        # Build a Graph that computes the logits predictions from the
        # inference model.
        images = tf.placeholder(dtype=tf.uint8, shape=(1, 64, 64, 1))
        labels = tf.placeholder(dtype=tf.uint16, shape=(1,))
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int32)
        logits, count = ChineseNet.inference(images, th=0.98)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        ChineseNet.load_weights(sess)

        true_count = 0
        start_time = time.time()
        for i in range(eval_num):
            _top_k_op, _count = sess.run([top_k_op, count],
                                          feed_dict={images: images_array[i],
                                          labels: labels_array[i]})
            true_count += _top_k_op[0]
            if (i+1) % 1000 == 0:
                print(datetime.now(), i+1, _count)
        duration = time.time() - start_time
        # Compute precision @ 1.
        precision = true_count / eval_num
        print('%d / %d = %.4f' % (true_count, eval_num, precision))
        print(duration, duration * 1000 / eval_num)
        print('%d images completed at Mid-A, %d images completed at Final'
              % (eval_num - _count, _count))

def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()