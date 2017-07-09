from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils.utils import *


class pix2pixmodel(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, output_size=256,
                 gf_dim=64, df_dim=64,
                 input_c_dim=3, output_c_dim=3,
                 wasserstein=False,
                 SELU=False):

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn = [0 for i in range(4)]
        self.g_bn_e = [0 for i in range(9)]
        self.g_bn_d = [0 for i in range(8)]
        for i in range(1,4):
            print i
            self.d_bn[i] = (batch_norm(name='d_bn' + str(i)) )

        for i in range(2,9):
            self.g_bn_e[i] = ( batch_norm(name='g_bn_e'+  str(i)) )

        for i in range(1,8):
            self.g_bn_d[i] = ( batch_norm(name='g_bn_d' + str(i)))

        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size

        self.SELU = SELU
        self.wasserstein = wasserstein


    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            def lrelu_discriminator():
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn[1](conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn[2](conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn[3](conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                return h4

            def selu_discriminator():
                h0 = selu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = selu(conv2d(h0, self.df_dim*2, name='d_h1_conv'))
                h2 = selu(conv2d(h1, self.df_dim*4, name='d_h2_conv'))
                h3 = selu(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                return 0

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h4 = None

            if(self.SELU):
                h4 = selu_discriminator()
            else:
                h4 = lrelu_discriminator()

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, sampler=False, y=None):
        with tf.variable_scope("generator") as scope:
            
            layer_specs_deconv = [
                self.gf_dim * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                self.gf_dim * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                self.gf_dim * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                self.gf_dim * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                self.gf_dim * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                self.gf_dim * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                self.gf_dim * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
            ]


            
            layer_specs_conv = [
       	        (self.gf_dim * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (self.gf_dim * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (self.gf_dim * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (self.gf_dim * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (self.gf_dim * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (self.gf_dim * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
                (self.gf_dim, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            ]
            

            if sampler:
                scope.reuse_variables()

            def lrelu_generator():
                self.e = [0 for i in range(9)]
                self.d = [0 for i in range(9)]
                self.d_w = [0 for i in range(9)]
                self.d_b = [0 for i in range(9)]

                self.e[1] = conv2d(image, self.gf_dim, name= 'g_e1_conv'))
                for i in range(2,9):
                    self.e[i] = self.g_bn_e[i](conv2d(lrelu(self.e[i-1]), layer_specs_deconv[i-2], name='g_e'+ i + '_conv' ))


                s = self.output_size

                s = int(s/128)
                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                    [self.batch_size, self.s, self.s, self.gf_dim*8], name='g_d1', with_w=True)

                for i in range(2,9):
                    s = int(self.output_size/2**(9-i))

                    self.d[i], self.d_w[i], self.d_b[i] = deconv2d(tf.nn.relu(e8),
                          [self.batch_size, s, s, self.gf_dim*8], name='g_d' + str(i), with_w=True)
                    if(layer_specs_conv)d[i] = tf.nn.dropout(self.g_bn_d[i](self.d[i]), 0.5)
                    d[i] = tf.concat([d[i], e[9-i]], 3)

                self.d[8], self.d_w[8], self.d_b[8] = deconv2d(tf.nn.relu(d[7]),
                  [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)

                return e

            def selu_generator():

                e = conv2d(image, self.gf_dim, name= 'g_e1_conv'))
                for i in range(2,9):
                    e = (conv2d(selu(e), layer_specs_deconv[i-2], name='g_e'+ i + '_conv' ))
                return 0

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e[2](conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e[3](conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e[4](conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e[5](conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e[6](conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e[7](conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e[8](conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)





            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d[1](self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d[2](self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d[3](self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d[4](self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d[5](self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d[6](self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d[7](self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)



    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            e2 = self.g_bn_e[2](conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            e3 = self.g_bn_e[3](conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            e4 = self.g_bn_e[4](conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            e5 = self.g_bn_e[5](conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            e6 = self.g_bn_e[6](conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            e7 = self.g_bn_e[7](conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            e8 = self.g_bn_e[8](conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d[1](self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d[2](self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d[3](self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d[4](self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d[5](self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d[6](self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d[7](self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)

            return tf.nn.tanh(self.d8)
