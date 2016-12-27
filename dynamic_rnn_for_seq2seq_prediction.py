import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from dynamic_rnn_for_sequence_prediction import DynamicRNN

class DynamicRNNSeq2Seq(DynamicRNN):
    def softmax_loss(self):
        """A softmax operations on the final output of the RNN
        """
        # masking the losses for padded input
        masker = tf.sequence_mask(self.sequence_length, tf.shape(self.outputs)[1])
        masker_reshaped = tf.reshape(tf.cast(masker, tf.float32), (-1,))

        target_reshaped = tf.tile(tf.reshape(self.target, (-1, 1)), (1, tf.shape(self.outputs)[1]))
        target_reshaped = tf.reshape(target_reshaped, (-1,))

        output_reshaped = tf.reshape(self.outputs, (-1, self.state_size))

        logit = tf.matmul(output_reshaped, self.W_softmax) + self.b_softmax
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, target_reshaped)
        masked_losses = losses * masker_reshaped
        self.loss = tf.reduce_sum(losses) / tf.reduce_sum(tf.cast(self.sequence_length, tf.float32))

        # final predictions
        predictions = tf.cast(tf.argmax(tf.nn.softmax(logit), dimension=1), tf.int32)
        correct = tf.cast(tf.equal(predictions, target_reshaped), tf.float32) * masker_reshaped
        self.accuracy = tf.reduce_sum(correct) / tf.reduce_sum(tf.cast(self.sequence_length, tf.float32))
