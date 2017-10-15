from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import TensorflowUtils as utils
import read_LaMemDataset as lamem
import read_postprocess_data as rd
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size")
tf.flags.DEFINE_string("logs_dir", "logs1/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/LaMem/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.9", "Beta 1 value to use in Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
IMAGE_SIZE = 128
ADVERSARIAL_LOSS_WEIGHT = 1e-3
step = 3
IMAGE_SIZE1 = 100
IMAGE_SIZE2 = 100
NUM_OF_CLASSESS = 100

def vgg_net(weights, image):
    layers = (
        'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i + 2][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def model_(images, keep_prob):
    print("setting up vgg initialized conv layers ...")
    vgg = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    weights = np.squeeze(vgg['layers'])

    W0 = utils.weight_variable([3, 3, 1, 64], name="W0")
    b0 = utils.bias_variable([64], name="b0")
    conv0 = utils.conv2d_basic(images, W0, b0)
    hrelu0 = tf.nn.relu(conv0, name="relu")

    image_net = vgg_net(weights, hrelu0)
    vgg_final_layer = image_net["relu5_3"]

    pool5 = utils.max_pool_2x2(vgg_final_layer)
    print(pool5)
    W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
    b6 = utils.bias_variable([4096], name="b6")
    conv6 = utils.conv2d_basic(pool5, W6, b6)
    relu6 = tf.nn.relu(conv6, name="relu6")
    relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

    W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
    b7 = utils.bias_variable([4096], name="b7")
    conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
    relu7 = tf.nn.relu(conv7, name="relu7")
    relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

    W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
    b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
    conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
    print(conv8)
    # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
    lstm_input_size = 4 * 4 * NUM_OF_CLASSESS
    pre_lstm_in = tf.reshape(conv8, [FLAGS.batch_size * step, 1, lstm_input_size])
    lstm_in = tf.reshape(pre_lstm_in, [FLAGS.batch_size, step, lstm_input_size])
    #lstm_input_size = lstm_in.get_shape().as_list()[2]
    x = tf.transpose(lstm_in, [1, 0, 2])
    pre_x_shape = x.get_shape()
    
    x = tf.reshape(x, [FLAGS.batch_size * step, lstm_input_size])
    
    x = tf.split(x, step, 0)
    
    
    
    
    lstm_cell1 = tf.contrib.rnn.GRUCell(lstm_input_size)
    #lstm_cell1 = tf.contrib.rnn.AttentionCellWrapper(cell=lstm_cell1,  attn_length=1,  attn_vec_size=2, state_is_tuple=True)
    lstm_cell1 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell1, output_keep_prob=0.9)
    
    
    lstm_cell2 = tf.contrib.rnn.GRUCell(lstm_input_size)
    lstm_cell2 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell2, output_keep_prob=0.9)
    # Get lstm cell output
    lstm_outputs, states, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_cell1, lstm_cell2, x, dtype=tf.float32)


    lstm_outputs = tf.stack(lstm_outputs)
    tmp_shape = lstm_outputs.shape
    print(lstm_outputs)
    lstm_outputs = tf.slice(lstm_outputs, [0,0,0], [tf.to_int32(tmp_shape[0]),tf.to_int32(tmp_shape[1]),tf.to_int32(pre_x_shape[2])])
    print(lstm_outputs)
    lstm_outputs = tf.reshape(lstm_outputs, pre_x_shape)
    lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
    lstm_outputs = tf.reshape(lstm_outputs, [FLAGS.batch_size * step, 4, 4, NUM_OF_CLASSESS]) 

    # now to upscale to actual image size
    deconv_shape1 = image_net["pool4"].get_shape()
    W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
    b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
    conv_t1 = utils.conv2d_transpose_strided(lstm_outputs, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
    fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

    deconv_shape2 = image_net["pool3"].get_shape()
    W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
    b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
    fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

    shape = tf.shape(images)
    deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 2])
    W_t3 = utils.weight_variable([16, 16, 2, deconv_shape2[3].value], name="W_t3")
    b_t3 = utils.bias_variable([2], name="b_t3")
    pred = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

    return tf.concat(values=[images, pred], axis = 3, name="pred_image")


def train(loss, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    for grad, var in grads:
        utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    with tf.device('/gpu:2'):
      keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
      images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE1, IMAGE_SIZE2, 1], name='L_image')
      lab_images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE1, IMAGE_SIZE2, 3], name="LAB_image")

      pred_image = model_(images, keep_probability)

      gen_loss_mse = tf.reduce_mean(2 * tf.nn.l2_loss(pred_image - lab_images)) / (IMAGE_SIZE1 * IMAGE_SIZE2 * 100 * 100)

      train_variables = tf.trainable_variables()

      train_op = train(gen_loss_mse, train_variables)

      print("Setting up session")
      config = tf.ConfigProto()
      config.allow_soft_placement=True
      config.log_device_placement=True
      sess = tf.Session(config=config)
      summary_op = tf.summary.merge_all()
      saver = tf.train.Saver()
      summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
      sess.run(tf.initialize_all_variables())

      ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
      if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          print("Model restored...")
          
      rd.get_data_divided()
      rd.read_data()
      rd.initial_get_batch()

    for itr in xrange(MAX_ITERATION):
        train_images, train_annotations = rd.get_batch(FLAGS.batch_size, step, IMAGE_SIZE1, IMAGE_SIZE2, data_type = 1)
        train_images = train_images.reshape((FLAGS.batch_size * step, IMAGE_SIZE1, IMAGE_SIZE2, 3))
        train_annotations = train_annotations.reshape((FLAGS.batch_size * step, IMAGE_SIZE1, IMAGE_SIZE2, 1))
        feed_dict = {lab_images: train_images, images: train_annotations, keep_probability: 0.85}

        if itr % 10 == 0:
            mse, summary_str = sess.run([gen_loss_mse, summary_op], feed_dict=feed_dict)
            print("Iter: %d, MSE: %g" % (itr, mse))

        if itr % 100 == 0:
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            print("%s --> Model saved" % datetime.datetime.now())

        sess.run(train_op, feed_dict=feed_dict)

        if itr % 10000 == 0:
            FLAGS.learning_rate /= 2

if __name__ == "__main__":
    tf.app.run()
