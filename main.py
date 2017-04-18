from model import SRCNN, SRGAN
from utils import input_setup

import numpy as np
import tensorflow as tf

import pprint
import os
from config import FLAGS
def main(mode='srgan'):
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  with tf.Session() as sess:
    if mode=='srcnn':
        srcnn = SRCNN(sess, 
                      image_size=FLAGS.image_size, 
                      label_size=FLAGS.label_size, 
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim, 
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)
    srgan = SRGAN(sess)
    srgan.train()
    srgan.test()
    #srgan.test_rgb()
    #srcnn.test('Test/Set5/baby_GT.bmp', FLAGS)
    #srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
