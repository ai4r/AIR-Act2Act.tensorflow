import random
import time
import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from tqdm import tqdm

from data.constants import *
from data.normalization import count_feature, denormalize_feature
from data.extract_data import b_iter
from act2act.model import Act2ActModel
from act2act.draw import get_data_files, draw

# Learning
tf.app.flags.DEFINE_float("learning_rate", .001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 1000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", int(1e4), "Iterations to train for.")

# Architecture
tf.app.flags.DEFINE_string("architecture", "basic", "Seq2seq architecture to use: [basic, tied].")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual_velocities", True, "Add a residual connection that effectively models velocities")

# Directories
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("../model/"), "Training directory.")

tf.app.flags.DEFINE_string("action", "all", "The action to train on.")
tf.app.flags.DEFINE_string("loss_to_use", "sampling_based", "The type of loss to use, supervised or sampling_based")
tf.app.flags.DEFINE_integer("test_every", 100, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("save_every", 100, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.normpath(os.path.join(FLAGS.train_dir,
    'in_{0}'.format(source_len), 'out_{0}'.format(target_len))
)

summaries_dir = os.path.normpath(os.path.join(train_dir, "log"))  # Directory for TB summaries


def create_model(session, sampling=False):
    """Create translation model and initialize or load parameters in session."""
    model = Act2ActModel(
        FLAGS.architecture,
        (source_len, count_feature(human_feature_type) + dist_len),
        context_len,
        (target_len, count_feature(robot_feature_type)),
        FLAGS.size, # hidden layer size
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        summaries_dir,
        FLAGS.loss_to_use if not sampling else "sampling_based",
        FLAGS.residual_velocities,
        dtype=tf.float32)

    if FLAGS.load <= 0:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    else:
        ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
        print( "train_dir", train_dir )
        if ckpt and ckpt.model_checkpoint_path:
            # Check if the specific checkpoint exists
            if os.path.isfile(os.path.join(train_dir, "checkpoint-{0}.index".format(FLAGS.load))):
                ckpt_name = os.path.normpath(os.path.join(os.path.join(train_dir, "checkpoint-{0}".format(FLAGS.load))))
            else:
                raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))

            print("Loading model {0}".format( ckpt_name ))
            model.saver.restore( session, ckpt.model_checkpoint_path )
            return model
        else:
            print("Could not find checkpoint. Aborting.")
            raise(ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))
    model.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    return model


def create_session():
    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count))


def data_generator(batch_size, data_type):
    if data_type not in ['test', 'train', 'validate']:
        raise (ValueError, "'{0}' is not an appropriate data type.".format(data_type))

    data_path = dst_test if data_type == 'test' else dst_train
    _, data_files, data_names = get_data_files(data_path)

    n_train = int(len(data_files) * 0.9)
    if data_type == 'test':
        data_names = data_names[:batch_size]
        data_files_backup = data_files
    if data_type == 'train':
        data_files = data_files[:n_train]
    if data_type == 'validate':
        data_files = data_files[n_train:]

    idx_test_data = 0
    while True:
        context_inp = []
        encoder_inp = []
        decoder_inp = []
        decoder_out = []
        if data_type == 'test':
            if idx_test_data >= len(data_names):
                break
            data_files = [name for name in data_files_backup if data_names[idx_test_data] in name]
            batch_size = len(data_files)
        else:
            random.shuffle(data_files)

        for i in range(batch_size):
            with np.load(data_files[i]) as data:
                human_seq = data['human_seq']
                robot_seq = data['robot_seq']
                if data_type == 'test':
                    robot_seq = np.vstack((robot_seq, [robot_seq[-1]] * (target_len - 1)))
                if data_type == 'train' and random.random() < 0.9:
                    def add_noise(matrix):
                        noise = np.random.normal(0, 0.01, matrix.shape)
                        matrix += noise
                    add_noise(human_seq)
                    add_noise(robot_seq[:target_len])

                context_inp.append(human_seq)
                encoder_inp.append(robot_seq[:(source_len-1)])
                decoder_inp.append(robot_seq[(source_len-1):-1])
                decoder_out.append(robot_seq[-target_len:])

        if data_type == 'test':
            yield data_names[idx_test_data], \
                  np.array(context_inp), np.array(encoder_inp), np.array(decoder_inp), np.array(decoder_out)
            idx_test_data += 1
        else:
            yield np.array(context_inp), np.array(encoder_inp), np.array(decoder_inp), np.array(decoder_out)


def train():
    """Train a seq2seq model on human motion"""
    # Limit TF to take a fraction of the GPU memory
    with create_session() as sess:
        # === Create the model ===
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        model = create_model(sess)
        model.train_writer.add_graph(sess.graph)
        print("Model created")

        # === Generator ===
        train_data_gen = data_generator(batch_size=FLAGS.batch_size, data_type='train')
        validate_data_gen = data_generator(batch_size=FLAGS.batch_size, data_type='validate')

        # === This is the training loop ===
        step_time, loss, val_loss = 0.0, 0.0, 0.0
        current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
        previous_losses = []

        step_time, loss = 0, 0
        for _ in xrange(FLAGS.iterations):
            start_time = time.time()

            # === Training step ===
            context_inp, encoder_inp, decoder_inp, decoder_out = train_data_gen.__next__()
            _, step_loss, loss_summary, lr_summary = model.step(sess, context_inp, encoder_inp,
                                                                decoder_inp, decoder_out, False)
            model.train_writer.add_summary(loss_summary, current_step)
            model.train_writer.add_summary(lr_summary, current_step)

            if current_step % 10 == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

            step_time += (time.time() - start_time) / FLAGS.test_every
            loss += step_loss / FLAGS.test_every
            current_step += 1

            # === step decay ===
            if current_step % FLAGS.learning_rate_step == 0:
                sess.run(model.learning_rate_decay_op)

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.test_every == 0:
                # === Validation with randomly chosen seeds ===
                forward_only = True

                context_inp, encoder_inp, decoder_inp, decoder_out = validate_data_gen.__next__()
                step_loss, loss_summary = model.step(sess, context_inp, encoder_inp, decoder_inp, decoder_out, forward_only)
                val_loss = step_loss  # Loss book-keeping

                model.test_writer.add_summary(loss_summary, current_step)

                print()
                print("{0: <16} |".format("milliseconds"), end="")
                for ms in [80, 160, 320, 400, 560, 1000]:
                    print(" {0:5d} |".format(ms), end="")
                print()

                # # === Validation with srnn's seeds ===
                print()
                print("============================\n"
                      "Global step:         %d\n"
                      "Learning rate:       %.4f\n"
                      "Step-time (ms):     %.4f\n"
                      "Train loss avg:      %.4f\n"
                      "--------------------------\n"
                      "Val loss:            %.4f\n"
                      "============================" % (model.global_step.eval(),
                                                        model.learning_rate.eval(), step_time * 1000, loss,
                                                        val_loss))
                print()

                # ===  Save the model and the behavior generation results ===
                if current_step % FLAGS.save_every == 0:
                    print("Saving the model...")
                    start_time = time.time()
                    model.saver.save(sess, os.path.normpath(
                        os.path.join(train_dir, "checkpoint-{:04}".format(current_step), 'checkpoint')),
                                     global_step=current_step)

                    print("done in {0:.2f} ms".format((time.time() - start_time) * 1000))

                    print("Saving the behavior generation results by the model...")
                    pbar = tqdm(total=10)
                    test_data_gen = data_generator(batch_size=10, data_type='test')
                    for data_name, context_inputs, encoder_inputs, decoder_inputs, decoder_outputs in test_data_gen:
                        tru_robot_joint_positions, gen_robot_joint_positions = list(), list()
                        tru_human_joint_positions = list()
                        gen_results = list()
                        for idx, val_decoder_out in enumerate(decoder_outputs):
                            if idx == 0 and not b_iter:
                                for each_encoder_inp in encoder_inputs[0]:
                                    joints = denormalize_feature(each_encoder_inp, robot_feature_type)
                                    tru_robot_joint_positions.append(joints)
                                    gen_robot_joint_positions.append(joints)
                                for each_context_inp in context_inputs[0]:
                                    joints = denormalize_feature(each_context_inp[dist_len:], human_feature_type)
                                    tru_human_joint_positions.append(joints)

                            context_inp = context_inputs[idx]
                            encoder_inp = encoder_inputs[idx] if len(gen_results[:-1]) == 0 else np.vstack((
                                encoder_inputs[idx][:-len(gen_results[:-1])], np.array(gen_results[:-1])))
                            decoder_inp = decoder_inputs[idx] if len(gen_results) == 0 else np.vstack((
                                np.array(gen_results[-1]), decoder_inputs[idx][1:]))
                            decoder_out = decoder_outputs[idx]
                            _, next_seq, _ = model.step(sess, np.array([context_inp]), np.array([encoder_inp]),
                                                        np.array([decoder_inp]), np.array([decoder_out]),
                                                        forward_only=True, srnn_seeds=True)

                            gen_results.append(next_seq[0][0])
                            gen_results = gen_results[-source_len:]
                            gen_robot_joint_positions.append(denormalize_feature(next_seq[0][0], robot_feature_type))
                            tru_robot_joint_positions.append(denormalize_feature(val_decoder_out[0], robot_feature_type))
                            tru_human_joint_positions.append(denormalize_feature(context_inp[-1][1:], human_feature_type))

                        draw([tru_human_joint_positions, tru_robot_joint_positions], os.path.normpath(
                            os.path.join(train_dir, "checkpoint-{:04}".format(current_step), data_name + '_trh.mp4')))
                        draw([tru_human_joint_positions, gen_robot_joint_positions], os.path.normpath(
                            os.path.join(train_dir, "checkpoint-{:04}".format(current_step), data_name + '_gen.mp4')))

                        pbar.update(1)
                    pbar.close()

                # Reset global time and loss
                step_time, loss = 0, 0


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
