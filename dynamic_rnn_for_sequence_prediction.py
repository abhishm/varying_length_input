import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class DynamicRNN(object):
    def __init__(self, state_size, batch_size, num_classes, vocab_size,
                 learning_rate, dropout_prob, suffix, train_initial_state=True,
                 summary_every=100):
        """Create a Basic RNN classfier with the given STATE_SIZE,
        NUM_STEPS, and NUM_CLASSES
        """
        self.state_size = state_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.suffix = suffix
        self.train_initial_state = train_initial_state
        self.summary_every = summary_every

        # tensorflow machinery
        self.session = tf.Session()
        self.summary_writer = tf.train.SummaryWriter(os.path.join(os.getcwd(), "tensorboard_{0}/".format(self.suffix)))
        self.no_op = tf.no_op()

        # counters
        self.train_itr = 0

        # create, initialize, and save variables
        self.create_graph()
        var_lists = tf.get_collection(tf.GraphKeys.VARIABLES)
        self.session.run(tf.initialize_variables(var_lists))
        self.saver = tf.train.Saver(max_to_keep=1)

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        # add the graph
        self.summary_writer.add_graph(self.session.graph)
        self.summary_every = summary_every

    def create_placeholders(self):
        self.input = tf.placeholder(tf.int32, shape=(self.batch_size, None), name="input")
        self.target = tf.placeholder(tf.int32, shape=(self.batch_size,), name="target")
        self.sequence_length = tf.placeholder(tf.int32, shape=(self.batch_size,), name="sequence_length")
        self.keep_prob = tf.placeholder_with_default(1.0, ())

    def create_variables(self):
        """Create variables for one layer RNN and the softmax
        """
        with tf.variable_scope("embeddings"):
            self.W_embeddings = tf.get_variable("embeddings", (self.vocab_size, self.state_size),
                                                initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("init_state"):
            self.init_state = tf.get_variable("init_state", (1, self.state_size),
                                              initializer=tf.constant_initializer(0.0))

        with tf.variable_scope("softmax"):
            self.W_softmax = tf.get_variable("W_softmax", [self.state_size, self.num_classes])
            self.b_softmax = tf.get_variable("b_softmax", [self.num_classes],
                                       initializer=tf.constant_initializer(0))

    def rnn(self):
        """ multi step RNN using tensorflow api "dynamic_rnn"
        TODO: Incorporate dropout in rnn_cell
        """
        self.rnn_inputs = tf.nn.embedding_lookup(self.W_embeddings, self.input)
        self.gru_cell = tf.nn.rnn_cell.GRUCell(self.state_size)
        if self.train_initial_state:
            self.batched_init_state = tf.tile(self.init_state, [self.batch_size, 1])
        else:
            self.batched_init_state = self.gru_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(self.gru_cell, self.rnn_inputs, self.sequence_length,
                                        initial_state=self.batched_init_state)
        self.outputs = tf.nn.dropout(self.outputs, self.keep_prob)

    def softmax_loss(self):
        """A softmax operations on the final output of the RNN
        """
        # We need to gather the final outputs from the rnn layers
        ids = (tf.shape(self.outputs)[1] * tf.range(self.batch_size)) + (self.sequence_length - 1)
        final_output = tf.gather(tf.reshape(self.outputs, [-1, self.state_size]), ids)
        logit = tf.matmul(final_output, self.W_softmax) + self.b_softmax
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, self.target)
        self.loss = tf.reduce_mean(losses)
        # final predictions
        predictions = tf.cast(tf.argmax(tf.nn.softmax(logit), dimension=1), tf.int32)
        correct = tf.equal(predictions, self.target)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    def create_variables_for_optimizations(self):
        """create variables for optimizing
        NB. Implement gradient clipping to overcome exploding gradient
        """
        with tf.name_scope("optimization"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_variables)
            self.train_op = self.optimizer.apply_gradients(self.gradients)

    def create_summaries(self):
        """create summary variables
        """
        self.loss_summary = tf.scalar_summary("loss", self.loss)
        self.accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)
        self.gradient_summaries = []
        for grad, var in self.gradients:
            if grad is not None:
                gradient_summary = tf.histogram_summary(var.name + "/gradient", grad)
                self.gradient_summaries.append(gradient_summary)
        self.weight_summaries = []
        weights = tf.get_collection(tf.GraphKeys.VARIABLES)
        for w in weights:
            weight_summary = tf.histogram_summary(w.name, w)
            self.weight_summaries.append(weight_summary)

    def merge_summaries(self):
        """Merge all sumaries
        """
        self.summarize = tf.merge_summary([self.loss_summary, self.accuracy_summary]
                                            + self.weight_summaries
                                            + self.gradient_summaries)

    def create_graph(self):
        self.create_placeholders()
        self.create_variables()
        self.rnn()
        self.softmax_loss()
        self.create_variables_for_optimizations()
        self.create_summaries()
        self.merge_summaries()

    def update_params(self, batch):
        """Given a batch of data, update the network to minimize the loss
        """
        write_summay = self.train_itr % self.summary_every == 0
        _, summary = self.session.run([self.train_op,
                                    self.summarize if write_summay else self.no_op],
                                    feed_dict={self.input: batch[0],
                                    self.target:batch[1],
                                    self.sequence_length:batch[2]},
                                    self.keep_prob: self.dropout_prob)
        if write_summay:
            self.summary_writer.add_summary(summary, self.train_itr)
            self.saver.save(self.session, os.path.join(os.getcwd(), "model_{0}.ckpt".format(self.suffix)), global_step=self.train_itr)
        self.train_itr += 1
