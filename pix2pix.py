from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils.utility import *
from utils.tfutils import *



class pix2pix(object):
    def __init__(self, sess, model,
                 batch_size=1,
                 sample_size=1,
                 train_size=1,
                 image_size=256,
                 output_size=256,
                 input_c_dim=3, output_c_dim=3,
                 dataset_name='facades',
                 epoch=200,
                 checkpoint_dir=None, sample_dir=None,
                 test_dir=None,
                 wasserstein=False):


        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
            wasserstein: use Discriminator loss function with wasserstenin distance
            selu: use Selu as activation function
        """

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.model = model

        self.sess = sess

        self.is_grayscale = (input_c_dim == 1)
        self.epoch = epoch
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.train_size = train_size
        self.output_size = output_size

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.test_dir = test_dir

        self.wasserstein = wasserstein



    """
    Building neural network.
    you can switch the activation function lrelu or selu
    please see the file utils.
    """

    def build_model(self, model, param):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_B = self.real_data[:, :, :, :self.input_c_dim]
        self.real_A = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = model.generator(self.real_A)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = model.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = model.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = model.generator(self.real_A, sampler=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        if(self.wasserstein):
                self.d_loss_real = tf.reduce_mean(self.D_logits)
                self.d_loss_fake = tf.reduce_mean(self.D_logits_)
                self.g_loss = self.d_loss_fake

                epsilon = tf.random_uniform([], 0.0, 1.0)
                x_hat = epsilon * tf.concat( [self.real_A, self.real_B ],3 ) + (1 - epsilon) *  tf.concat( [ self.real_A, self.fake_B ],3)
                _,d_hat= self.discriminator( x_hat , reuse=True)

                ddx = tf.gradients(d_hat, x_hat)[0]
                ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
                ddx = tf.reduce_mean(tf.square(ddx - 1.0) *10)

                self.d_loss = self.d_loss_real - self.d_loss_fake +ddx

        else:
                self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
                self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                            + param.L1_scale * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

                self.d_loss = self.d_loss_real + self.d_loss_fake


        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def train(self, param):


        """" Generator and Discriminator Optimizer. using Adam"""

        d_optim = tf.train.AdamOptimizer(param.lr, beta1=param.beta1).minimize(self.d_loss, var_list=self.d_vars)

        g_optim = tf.train.AdamOptimizer(param.lr, beta1=param.beta1).minimize(self.g_loss, var_list=self.g_vars)


        """
         initialize variables
         perhaps, initializing weight lead the bug when using activation function SELU
         Please git me hint ,The eroihito
        """

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)


        self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if load(self):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(self.epoch):
            data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))

            batch_idxs = min(len(data), self.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                upG = 2
                for i in range(0,upG-1):
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={ self.real_data: batch_images })
                    self.writer.add_summary(summary_str, counter)



                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, errD_fake: %.8f, errD_real: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake, errD_real, errG))

                if np.mod(counter, 100) == 1:
                    sample_model(self, self.sample_dir, epoch, idx)

                if np.mod(counter, 500) == 2:
                    save(self, self.checkpoint_dir, counter)


    def test(self, param):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('./datasets/{}/val/*.jpg'.format(self.dataset_name))

        # sort testing input
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)


        start_time = time.time()


        if load(self):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            save_images(samples, [self.batch_size, 1],
                        './{}/test_{:04d}.jpg'.format(self.test_dir, idx))
