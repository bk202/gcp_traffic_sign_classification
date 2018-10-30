import tensorflow as tf
from base_model import BaseModel

class ClassifierModel(BaseModel):
    def __init__(self, data_loader, config):
        super(ClassifierModel, self).__init__(config)
        self.data_loader = data_loader
        self.input = None
        self.labels = None
        self.is_training = None
        self.loss = None
        self.optimizer = None
        self.train_step = None
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        self.num_classes = config.num_classes
        self.learning_rate = config.learning_rate

        self.build_model()
        self.init_saver()
        return

    def build_model(self):
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE):
            # self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.num_channels])
            # self.labels = tf.placeholder(tf.float32, [None, self.num_classes])
            self.input, self.labels = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.input)
        tf.add_to_collection('inputs', self.labels)
        tf.add_to_collection('inputs', self.is_training)


        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            self.conv1, _, _ = self.convolution(self.input, 5, 32, name='conv1_block')
            with tf.variable_scope('max_pool1', reuse=tf.AUTO_REUSE):
                self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = self.pool1.get_shape().as_list()
            self.cnn_feed1 = tf.reshape(self.pool1, shape=[-1, shape[1] * shape[2] * shape[3]], name='cnn_feed1')

            self.conv2, _, _ = self.convolution(self.pool1, 5, 64, name='conv2_block')
            with tf.variable_scope('max_pool2', reuse=tf.AUTO_REUSE):
                self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = self.pool2.get_shape().as_list()
            self.cnn_feed2 = tf.reshape(self.pool2, shape=[-1, shape[1] * shape[2] * shape[3]], name='cnn_feed2')

            self.conv3, _, _ = self.convolution(self.pool2, 5, 128, name='conv3_block')
            with tf.variable_scope('max_pool3', reuse=tf.AUTO_REUSE):
                self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = self.pool3.get_shape().as_list()
            self.cnn_feed3 = tf.reshape(self.pool3, shape=[-1, shape[1] * shape[2] * shape[3]], name='cnn_feed3')

            self.fully_input = tf.concat([self.cnn_feed1, self.cnn_feed2, self.cnn_feed3], axis=1)

            self.fc1, _, _ = self.fully_connected(self.fully_input, hidden_nodes=1024, name='fc1_block')
            self.fc1 = tf.nn.relu(self.fc1)
            self.output, _, _ = self.fully_connected(self.fc1, hidden_nodes=self.num_classes, name='output_block')

        with tf.variable_scope('out_argmax', reuse=tf.AUTO_REUSE):
            self.out_argmax = tf.argmax(self.output, axis=-1, output_type=tf.int64, name='out_argmax')

        with tf.variable_scope('loss_acc', reuse=tf.AUTO_REUSE):
            #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output)
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.output)
            #self.loss = tf.reduce_mean(self.loss)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.out_argmax), tf.float32))

        with tf.variable_scope('train_step', reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=10, save_relative_paths=True)

    def convolution(self, input, kernel_size, output_depth, name):
        input_size = input.get_shape().as_list()[3]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # weights = tf.Variable(
            #     tf.truncated_normal([kernel_size, kernel_size, input_size, output_depth], stddev=0.01),
            #     name='conv'
            # )
            # biases = tf.Variable(
            #     tf.zeros(output_depth),
            #     name='biase'
            # )
            weights = tf.get_variable(
                name='conv',
                shape=[kernel_size, kernel_size, input_size, output_depth],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            biases = tf.get_variable(
                name='biases',
                shape=[output_depth],
                initializer=tf.zeros_initializer()
            )

            output = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
            output = tf.nn.relu(output + biases)
            print("{} -- {}" .format(name, output.shape))
            return output, weights, biases

    def fully_connected(self, input, hidden_nodes, name):
        input_size = input.get_shape().as_list()[1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # weights = tf.get_variable(
            #     tf.truncated_normal([input_size, hidden_nodes], stdde=1.0 / hidden_nodes),
            #     name='fully_connected'
            # )
            # biases = tf.get_variable(
            #     tf.zeros(hidden_nodes),
            #     name='biase'
            # )
            weights = tf.get_variable(
                name='fully_connected',
                shape=[input_size, hidden_nodes],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            biases = tf.get_variable(
                name='biases',
                shape=[hidden_nodes],
                initializer=tf.zeros_initializer()
            )

            output = tf.matmul(input, weights) + biases
            print("{} -- {}".format(name, output.shape))
            return output, weights, biases





