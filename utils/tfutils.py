from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def load_random_samples(model):
    data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(model.dataset_name)), model.batch_size)
    sample = [load_data(sample_file) for sample_file in data]

    if (model.is_grayscale):
        sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
        sample_images = np.array(sample).astype(np.float32)
    return sample_images

def sample_model(model, sample_dir, epoch, idx):

    sample_images = load_random_samples(model)

    samples, d_loss, g_loss = model.sess.run(
        [model.fake_B_sample, model.d_loss, model.g_loss],
        feed_dict={model.real_data: sample_images}
    )

    save_images(samples, [model.batch_size, 1],
                './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
