#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2 & 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TF imports
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.layers import utils
from tensorflow import initializers


class SimplestLowpass(Conv1D):
    """
    Return a TF layer which computes the simplest lowpass filter via 1D
    convolution. Subclasses `tf.layers.Conv1D`.

    Note that padding is causal, _i.e._ a single zero sample is added at the
    beginning (and not the end). Thus the slight additions to the `build` and
    `call` methods.
    """
    def __init__(self,
                 b_coefs=[0.5, 0.5],
                 activation=None,  # Linear function
                 activity_regularizer=None,
                 trainable=False,
                 name=None,
                 **kwargs):
        # `kernel_size` is the length of the `b` vector. Here we have just `b0`
        # and `b1`.
        self.rank = 1  # Perform 1D convolution
        self.kernel_size = utils.normalize_tuple(2, self.rank, 'kernel_size')
        self.activation = activation
        self.b_coefs = b_coefs

        # Instantiate from parent, `Conv1D`
        super(SimplestLowpass, self).__init__(
            filters=1,  # We're implementing a single filter
            kernel_size=self.kernel_size,
            strides=1,  # Proceed sample by sample
            padding='valid',  # Preserve size if input signal in output
            data_format='channels_last',
            dilation_rate=1,  # No dilation
            activation=self.activation,
            use_bias=False,
            kernel_initializer=initializers.constant(self.b_coefs),
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            name=name,
            **kwargs)

    def build(self, input_shape):
        input_shape._dims[1] += 1  # Add a single sample of causal padding
        super(SimplestLowpass, self).build(input_shape)

    def call(self, inputs):
        # Add causal padding
        inputs = tf.pad(inputs,
                        [[0, 0], [1, 0], [0, 0]])
        return super(SimplestLowpass, self).call(inputs)


def simplest_lowpass(inputs, **kwargs):
    """
    Functional interface for simplest lowpass filter layer.

    Arguments:
        inputs: Tensor input.
    """
    layer = SimplestLowpass()

    # Handle case where inputs are of shape (batch_size, n_samples). Recall we
    # need a shape of (batch_size, n_samples, n_features), where n_features is
    # probably 1.
    n_input_dims = len(inputs.shape)
    if n_input_dims == 2:
        tf.expand_dims(inputs, -1)
    elif n_input_dims == 1:
        # Assume single signal to filter.
        tf.expand_dims(tf.expand_dims(inputs, -1), 0)
    elif n_input_dims > 3:
        raise ValueError('Expected no more than 3 input dims. '
                         'Got {}: {}'.format(n_input_dims, inputs.shape))

    # Actually apply the filter
    outputs = layer.apply(inputs, **kwargs)

    # Squeeze outputs to conform to input shape
    if n_input_dims == 2:
        tf.squeeze(outputs, -1)
    elif n_input_dims == 1:
        tf.squeeze(tf.squeeze(outputs, -1), 0)

    return outputs


if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    from scipy.signal import lfilter

    batch_size = 10
    sample_length = 100

    x = tf.placeholder(shape=[None, None, 1], dtype=tf.float32)
    y = simplest_lowpass(x)

    # Get the filtered output from TF
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        x_vals = np.random.normal(0, 1, (batch_size, sample_length, 1))
        y_vals, = sess.run([y], feed_dict={x: x_vals})

    # Compute the filtered output independently with the `scipy.signal` module
    y_vals_expected = lfilter([0.5, 0.5], [1.0], x_vals, axis=1)

    if np.allclose(y_vals, y_vals_expected):
        print("Success: The TensorFlow layer computed the desired output.")
    else:
        print("Fail: TensorFlow output does not conform to expected output.")
