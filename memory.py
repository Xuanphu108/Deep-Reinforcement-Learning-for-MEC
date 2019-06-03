#  #################################################################
# version 1.0 -- April 2019. Written by Phu X. Nguyen (nxphu@gmail.com)
#  #################################################################

from __future__ import print_function
import tensorflow as tf
import numpy as np


# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        net_num,
        learning_rate=0.01,
        training_interval=10,
        batch_size=128,
        memory_size=1024,
        output_graph=False,
        # drop=0.5
    ):
        # net: [n_input, n_hidden_1st, n_hidded_2ed, n_output]
        assert(len(net) is 4)  # only 4-layer DNN

        self.net = net
        self.net_num = net_num
        self.training_interval = training_interval  # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        # self.drop = drop

        # stored # memory entry
        self.memory_counter = 1
        self.m_pred = []
        self.loss = []
        self.train_op = []
        self.cost_his = [[] for i in range(self.net_num)]

        # reset graph
        tf.reset_default_graph()

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))
        # construct memory network
        self._build_net()

        self.sess = tf.Session()

        # for tensorboard
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        def build_layers(h, c_names, net, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [net[0], net[1]], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, self.net[1]], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(h, w1) + b1)
                # print(w1.name)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [net[1], net[2]], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, net[2]], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('M'):
                w3 = tf.get_variable('w3', [net[2], net[3]], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, net[3]], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3

            return out

        # ------------------ build memory_net ------------------
        self.h = tf.placeholder(tf.float32, [None, self.net[0]], name='h')  # input
        self.m = tf.placeholder(tf.float32, [None, self.net[-1]], name='mode')  # for calculating loss
        self.is_train = tf.placeholder("bool")  # train or evaluate
        for i in range(self.net_num):
            with tf.variable_scope('memory%d_net' % i):
                w_initializer, b_initializer = \
                    tf.random_normal_initializer(0., 1/self.net[0]), tf.constant_initializer(0.1)  # config of layers
                self.m_pred.append(build_layers(self.h, ['memory%d_net_params' % i, tf.GraphKeys.GLOBAL_VARIABLES], self.net, w_initializer, b_initializer))
            with tf.variable_scope('loss%d' % i):
                self.loss.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.m, logits=self.m_pred[i])))
            with tf.variable_scope('train%d' % i):
                self.train_op.append(tf.train.AdamOptimizer(self.lr, 0.09).minimize(self.loss[i]))

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every 10 step
        if self.memory_counter >= 10 and self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        sample_index = []
        batch_memory = []
        h_train = []
        m_train = []
        if self.memory_counter > self.memory_size:
            for j in range(self.net_num):
                sample_index.append(np.random.choice(self.memory_size, size=self.batch_size))
        else:
            for j in range(self.net_num):
                sample_index.append(np.random.choice(self.memory_counter, size=self.batch_size))
        for j in range(self.net_num):
            batch_memory.append(self.memory[sample_index[j], :])
            h_train.append(batch_memory[j][:, 0: self.net[0]])
            m_train.append(batch_memory[j][:, self.net[0]:])
            _, cost = self.sess.run([self.train_op[j], self.loss[j]],
                                         feed_dict={self.h: h_train[j], self.m: m_train[j]})
            assert(cost > 0)
            self.cost_his[j].append(cost)

    def decode(self, h, N):
        # to have batch dimension when feed into tf placeholder
        m_list = []
        # h = h[np.newaxis, :]
        for k in range(self.net_num):
            m_pred = self.sess.run(self.m_pred[k], feed_dict={self.h: h})

            print(m_pred)
            '''
            vec = np.asarray(1 * (m_pred[0] > 0))
            print(vec)
            mat = vec.reshape(N, N)
            m_list.append(mat.T)
            '''

            Output = []

            for i in range(0, len(m_pred[0]), N):
                ColOut = np.zeros([1, N])

                '''
                MaxAgr = np.where(m_pred[0][i:i+N] == np.amax(m_pred[0][i:i+N]))
                print('Original MaxAgr')
                print(MaxAgr)
                MaxIndex = np.random.choice(MaxAgr[0], 1)
                print('Original MaxIndex')
                print(MaxIndex)
                '''

                MaxAgr = np.argmax(m_pred[0][i:i + N])
                MaxIndex = np.random.choice([MaxAgr], 1)

                if (np.count_nonzero(1*(m_pred[0][i:i+N] > 0)) == 0):  # len(m_pred[0][i:i+N])):
                    ColOut = ColOut
                else:
                    ColOut[0][MaxIndex[0]] = 1
                Output.append(ColOut[0])
            Output = np.asarray(Output)
            m_list.append(Output.T)
            print(m_list)
        return m_list

    def plot_cost(self):
        import matplotlib.pyplot as plt
        colors ="bgrcmykw"
        for p in range(self.net_num):
            plt.plot(np.arange(len(self.cost_his[p])), self.cost_his[p], colors[np.random.randint(0, 8)])
        plt.ylabel('Cost of MemoryDNN')
        plt.xlabel('training steps')
        plt.show()


