#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)
def resnet_block(inputs, nb_blocks, out_channels, downsample=False,
             global_pool=False, scope = 'resnet'):
    with tf.variable_scope(scope, 'resnet', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d],weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(), normalizer_fn = slim.batch_norm,activation_fn = tf.nn.relu, padding='SAME'):
            resnet = inputs
            in_channels = inputs.get_shape().as_list()[-1]

            for i in range(nb_blocks):
                identity = resnet
                if not downsample:
                    downsample_strides = 1
                    resnet = slim.conv2d(resnet, out_channels, [3,3])
                    resnet = slim.conv2d(resnet, out_channels, [3,3])
                if in_channels != out_channels:
                    resnet = slim.conv2d(resnet, in_channels, [1,1])
    resnet = resnet + identity
    return resnet
def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X,a,1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
    X = tf.split(X,b,1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X


if __name__ == "__main__":
    with tf.Session() as sess:
        x = np.arange(2*16*16).reshape(2, 8, 8, 4)
        X = tf.placeholder("float32", shape=(2, 8, 8, 4), name="X")# tf.Variable(x, name="X")
        Y = PS(X, 2)
        y = sess.run(Y, feed_dict={X: x})

        x2 = np.arange(2*3*16*16).reshape(2, 8, 8, 4*3)
        X2 = tf.placeholder("float32", shape=(2, 8, 8, 4*3), name="X")# tf.Variable(x, name="X")
        Y2 = PS(X2, 2, color=True)
        y2 = sess.run(Y2, feed_dict={X2: x2})
        print y2.shape
    plt.imshow(y[0, :, :, 0], interpolation="none")
    plt.show()

