"""Sequence-to-sequence model for robot motion generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from utils import rnn_cell_extensions # my extensions of the tf repos


class Act2ActModel(object):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
        architecture,
        source_seq_size,
        context_len,
        target_seq_size,
        rnn_size, # hidden recurrent layer size
        num_layers,
        max_gradient_norm,
        batch_size,
        learning_rate,
        learning_rate_decay_factor,
        summaries_dir,
        loss_to_use,
        residual_velocities=False,
        dtype=tf.float32):
        """Create the model.
        Args:
            source_seq_size: (length, size) of the input sequence.
            target_seq_size: (length, size) of the target sequence.
            rnn_size: number of units in the rnn.
            num_layers: number of rnns to stack.
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
            learning_rate_decay_factor: decay learning rate by this much when needed.
            summaries_dir: where to log progress for tensorboard.
            loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
                each timestep to compute the loss after decoding, or to feed back the
                prediction from the previous time-step.
            residual_velocities: whether to use a residual connection that models velocities.
            dtype: the data type to use to store internal variables.
        """

        # Summary writers for train and test runs
        self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'train')))
        self.test_writer  = tf.summary.FileWriter(os.path.normpath(os.path.join(summaries_dir, 'test')))

        self.source_seq_len = source_seq_size
        self.context_len = context_len
        self.target_seq_len = target_seq_size
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # === Create the RNN that will keep the state ===
        # === Contexter ===
        print('rnn_size = {0}'.format(rnn_size))

        # Transform the inputs
        with tf.name_scope("inputs"):
            con_in = tf.placeholder(dtype, shape=[None, source_seq_size[0], source_seq_size[1]], name="con_in")
            self.context_inputs = con_in

            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
            con_in = tf.unstack(con_in, source_seq_size[0], 1)

        # Define a lstm cell with tensorflow
        with vs.variable_scope("basic_lstm"):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0)
            outputs_human, states = tf.contrib.rnn.static_rnn(lstm_cell, con_in, dtype=tf.float32)

        outputs_human = tf.layers.dense(outputs_human[-1], context_len, name='fc')
        # prediction = tf.nn.softmax(logits)
        # self.outputs = logits

        # === Encoder & Decoder ===
        cell_robot = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0)
        if num_layers > 1:
            cell_robot = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.rnn_size, forget_bias=1.0) for _ in range(num_layers)])

        # Transform the inputs
        with tf.name_scope("inputs"):
            enc_in = tf.placeholder(dtype, shape=[None, source_seq_size[0]-1, target_seq_size[1]], name="enc_in")
            dec_in = tf.placeholder(dtype, shape=[None, target_seq_size[0], target_seq_size[1]], name="dec_in")
            dec_out = tf.placeholder(dtype, shape=[None, target_seq_size[0], target_seq_size[1]], name="dec_out")

            self.encoder_inputs = enc_in
            self.decoder_inputs = dec_in
            self.decoder_outputs = dec_out

            context_tiled_1 = tf.tile(tf.expand_dims(tf.cast(outputs_human, dtype=tf.float32), axis=1),
                                      multiples=[1, source_seq_size[0]-1, 1])
            enc_in = tf.concat([enc_in, context_tiled_1], 2)
            context_tiled_2 = tf.tile(tf.expand_dims(tf.cast(outputs_human, dtype=tf.float32), axis=1),
                                      multiples=[1, target_seq_size[0], 1])
            dec_in = tf.concat([dec_in, context_tiled_2], 2)

            enc_in = tf.transpose(enc_in, [1, 0, 2])
            dec_in = tf.transpose(dec_in, [1, 0, 2])
            dec_out = tf.transpose(dec_out, [1, 0, 2])

            enc_in = tf.reshape(enc_in, [-1, target_seq_size[1] + context_len])
            dec_in = tf.reshape(dec_in, [-1, target_seq_size[1] + context_len])
            dec_out = tf.reshape(dec_out, [-1, target_seq_size[1]])

            enc_in = tf.split(enc_in, source_seq_size[0]-1, axis=0)
            dec_in = tf.split(dec_in, target_seq_size[0], axis=0)
            dec_out = tf.split(dec_out, target_seq_size[0], axis=0)

        # === Add space decoder ===
        cell_robot = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell_robot, target_seq_size[1],
                                                                   w_name="proj_w_out2", b_name="proj_b_out2")

        # Finally, wrap everything in a residual layer if we want to model velocities
        if residual_velocities:
            cell_robot = rnn_cell_extensions.ResidualWrapper(cell_robot)

        # Store the outputs here
        outputs_robot = []

        # Define the loss function
        lf = None
        if loss_to_use == "sampling_based":
            def lf(prev, i): # function for sampling_based loss
                return prev
        elif loss_to_use == "supervised":
            pass
        else:
            raise(ValueError, "unknown loss: %s" % loss_to_use)

        # Build the RNN
        if architecture == "basic":
            _, enc_state = tf.contrib.rnn.static_rnn(cell_robot, enc_in, dtype=tf.float32)  # Encoder
            outputs_robot, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(dec_in, enc_state, cell_robot, loop_function=lf)  # Decoder
        elif architecture == "tied":
            outputs_robot, self.states = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(enc_in, dec_in, cell_robot, loop_function=lf)
        else:
            raise (ValueError, "Unknown architecture: %s" % architecture)
        outputs_robot = [tf.slice(outputs_robot[i], [0, 0], [-1, target_seq_size[1]]) for i in range(len(outputs_robot))]
        self.outputs = outputs_robot

        with tf.name_scope("loss_angles"):
            # loss_angles = tf.exp(-tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs_robot))))
            loss_angles = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs_robot)))

        self.loss         = loss_angles
        self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()

        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        # Update all the trainable parameters
        gradients = tf.gradients(self.loss, params)

        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        # Keep track of the learning rate
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    def step(self, session, context_inputs, encoder_inputs, decoder_inputs, decoder_outputs,
             forward_only, srnn_seeds=False):
        """Run a step of the model feeding the given inputs.
        Args
            session: tensorflow session to use.
            encoder_inputs: list of numpy vectors to feed as encoder inputs.
            decoder_inputs: list of numpy vectors to feed as decoder inputs.
            decoder_outputs: list of numpy vectors that are the expected decoder outputs.
            forward_only: whether to do the backward step or only forward.
            srnn_seeds: True if you want to evaluate using the sequences of SRNN
        Returns
            A triple consisting of gradient norm (or None if we did not do backward),
            mean squared error, and the outputs.
        Raises
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """
        input_feed = {self.context_inputs: context_inputs,
                      self.encoder_inputs: encoder_inputs,
                      self.decoder_inputs: decoder_inputs,
                      self.decoder_outputs: decoder_outputs}

        # Output feed: depends on whether we do a backward step or not.
        if not srnn_seeds:
            if not forward_only:

                # Training step
                output_feed = [self.updates,         # Update Op that does SGD.
                               self.gradient_norms,  # Gradient norm.
                               self.loss,
                               self.loss_summary,
                               self.learning_rate_summary]

                outputs = session.run(output_feed, input_feed)
                return outputs[1], outputs[2], outputs[3], outputs[4]  # Gradient norm, loss, summaries

            else:
                # Validation step, not on SRNN's seeds
                output_feed = [self.loss,  # Loss for this batch.
                               self.loss_summary]

                outputs = session.run(output_feed, input_feed)
                return outputs[0], outputs[1]  # No gradient norm
        else:
            # Validation on SRNN's seeds
            output_feed = [self.loss, # Loss for this batch.
                           self.outputs,
                           self.loss_summary]

            outputs = session.run(output_feed, input_feed)

            return outputs[0], outputs[1], outputs[2]  # No gradient norm, loss, outputs.
