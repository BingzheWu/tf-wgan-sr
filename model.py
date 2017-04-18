from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  fit_scale_crop,
  input2patches,
  ycbcr2rgb,
  read_h5_data,
)
from load_utils import save, load
from config import FLAGS
import time
import os
import skimage.color
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from ops import PS, resnet_block
import scipy.misc
import Image
slim =  tf.contrib.slim
class SRCNN(object):
  def __init__(self, 
               sess, 
               image_size=33,
               label_size=17, 
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size
    self.checkpoint_dir = 'checkpoint'
    self.c_dim = c_dim
    self.name= 'SRCNN'
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    self.pred = self.model()
    # Loss function (MSE)
    self.tv =tf.image.total_variation(self.pred)
    print(type(self.pred))
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))+0.0005*tf.reduce_mean(self.tv)
    self.saver = tf.train.Saver()

  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config)
    else:
      nx, ny = input_setup(self.sess, config)

    if config.is_train:     
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

    train_data, train_label = read_data(data_dir)

    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()
    if load(self):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 500 == 0:
            self.save(config.checkpoint_dir, counter)

    else:
      print("Testing...")

      result = self.pred.eval({self.images: train_data, self.labels: train_label})

      result = merge(result, [nx, ny])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      print(train_label.shape)
      self.imshow(result)
      imsave(result, image_path)
  def input_setup(self, label, config):
      import scipy.ndimage
      input_ = scipy.ndimage.interpolation.zoom(label, 1.0/3.0)
      input_ = scipy.ndimage.interpolation.zoom(input_, 3.)
      h, w = input_.shape
      print(h,w)
      sub_input_sequence = []
      sub_label_sequence = []
      nx = ny = 0
      for x in range(0, h-config.image_size+1, config.stride):
          nx+=1
          ny=0
          for y in range(0, w-config.image_size+1, config.stride):
              ny+=1
              sub_input = input_[x:x+config.image_size, y:y+config.image_size]
              sub_label = label[x:x+config.label_size, y:y+config.label_size]
              sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
              sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
              sub_input_sequence.append(sub_input)
              sub_label_sequence.append(sub_label)
      arrdata = np.asarray(sub_input_sequence)
      arrlabel = np.asarray(sub_label_sequence)
      return arrdata, arrlabel, nx, ny
  def test(self, img_path, config):
      import scipy.misc
      ycbcr = scipy.misc.imread(img_path, mode = 'YCbCr').astype(float)
      label = ycbcr[:,:,0]
      input_ = scipy.ndimage.interpolation.zoom(label, 1.0/3.0)
      input_ = scipy.ndimage.interpolation.zoom(input_, 3.)
      batch_data, batch_label, nx, ny= self.input_setup(label, config)
      self.sess.run(tf.initialize_all_variables())
      load(self)
      result = self.sess.run(self.pred, feed_dict = {self.images: batch_data, self.labels: batch_label})
      result = merge(result, [nx,ny])
      result = result.squeeze()
      print(result.shape)
      self.imshow(input_)
      self.imshow(result)
  def imshow(self, img):
      import matplotlib.pyplot as plt
      plt.imshow(img, cmap = 'gray')
      plt.show()
  def model(self):
    with slim.arg_scope([slim.conv2d], padding = 'VALID', weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=slim.l2_regularizer(0.00005)):
        conv1 = slim.conv2d(self.images, 64, [9,9], scope = 'conv1')
        conv2 = slim.conv2d(conv1, 32, [5,5], scope = 'conv2')
        conv3 = slim.conv2d(conv2, 1, [5,5], scope = 'conv3', activation_fn = tf.nn.tanh, normalizer_fn = None)
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)
class SRGAN(object):
    def __init__(self, sess, lr_patch_size = 24, is_crop = True,
                batch_size = 64, y_dim = None,
                z_dim = 100, gf_dim = 64, df_dim = 64,
                gfc_dim = 1024, dfc_dim = 1024, c_dim = 1, dataset_name = 'default',
                 checkpoint_dir = 'checkpoint'):
        self.config = FLAGS
        self.name = 'srgan_gray'
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = self.config.batch_size
        self.image_size = self.config.label_size
        self.c_dim = c_dim
        self.image_shape = [self.image_size, self.image_size, self.c_dim]
        self.lr_size = lr_patch_size
        self.y_dim  = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.train_data_path = 'train91/train_51'
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None]+self.image_shape, name = 'raw_images')
        self.lr_images = tf.image.resize_images(self.images, [self.lr_size, self.lr_size], tf.image.ResizeMethod.BICUBIC)
        self.up_image = tf.image.resize_images(self.lr_images, [self.image_size, self.image_size], tf.image.ResizeMethod.BICUBIC)
        self.raw_sum = tf.summary.image("raw_image", self.images)
        self.lr_sum = tf.summary.image("lr_image", self.lr_images)
        ## inference block of generator
        up, self.G = self.generator(self.lr_images)
        self.G_sum = tf.summary.image("G", self.G)
        self.up_sum = tf.summary.image("G", self.up_image)
        self.d_real = self.discriminator(self.images)
        self.d_fake = self.discriminator(self.G)

        ### build loss block
        label_true = tf.ones(self.config.batch_size)
        label_false = tf.zeros(self.config.batch_size)
        self.d_loss = tf.reduce_mean(self.d_real) - tf.reduce_mean(self.d_fake)
        self.d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_real, labels = label_true)
        self.d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake, labels = label_false)
        #self.d_loss = tf.reduce_mean(self.d_real_loss) +tf.reduce_mean(self.d_fake_loss)

        # gloss
        self.g_loss_dis = -tf.reduce_mean(self.d_fake)
        #self.g_loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake, labels = label_true))
        self.g_content_loss = tf.reduce_mean(tf.square(self.images-self.G))
        self.tv = tf.reduce_mean(tf.image.total_variation(self.G))
        self.g_loss = self.g_content_loss + 0.0001*self.g_loss_dis + 0.00001*self.tv
        self.g_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        ##build optim blocks
        t_vars = tf.model_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.d_optim = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(-self.d_loss, var_list = self.d_vars)
        self.g_optim = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.g_loss, var_list = self.g_vars)
        self.clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]
        self.saver = tf.train.Saver()
    def train(self):
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()
        load(self)
        self.g_sum = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter = 1
        start_time = time.time()
        train_data = read_h5_data(self.train_data_path)
        for epoch in xrange(self.config.epoch):
            batch_idxs = len(train_data) // self.config.batch_size
            for idx in xrange(0, batch_idxs):
                batch = train_data[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
                _,errD,tmp = self.sess.run([self.d_optim, self.d_loss, self.clip_d], 
                                                     feed_dict = {self.images: batch})
                '''
                _,errD = self.sess.run([self.d_optim, self.d_loss,], 
                                                     feed_dict = {self.images: batch})
                '''
                _, summary_str, errG = self.sess.run([self.g_optim, self.g_sum, self.g_content_loss], 
                                                     feed_dict = {self.images: batch})
                self.writer.add_summary(summary_str, counter)
                counter+=1
                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], gloss: [%.8f], dloss: [%.8f]" \
                        % ((epoch+1), counter, time.time()-start_time, errG, errD))
                if np.mod(counter, 500)==2:
                    save(self, counter)
                    
    def test(self, img_path='Test/Set5/baby_GT.bmp'):
        img_color = scipy.misc.imread(img_path, mode = 'YCbCr')
        #img_color = cv2.imread(img_path)
        #img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2YCR_CB)
        img_color = img_color.transpose(2,0,1)
        img, Cb, Cr = img_color[0], img_color[1], img_color[2]
        img = img*np.float32(1.0/255.0)
        img = img.astype(np.float32)
        img_lr = scipy.ndimage.interpolation.zoom(img, 1.0/self.config.scale, )
        img_lr = np.expand_dims(np.expand_dims(img_lr,axis=2), axis=0)
        self.sess.run(tf.initialize_all_variables())
        load(self)
        hr = self.generator(img_lr)[1].eval()
        hr = hr.squeeze()
        hr = fit_scale_crop(hr, img)
        hr =(hr*255.0)
        self.imshow(hr, is_gray=True)
    def test_rgb(self, img_path = 'Test/Set5/baby_GT.bmp'):
        #img_path = 'test.png'
        img = scipy.misc.imread(img_path)
        img_lr = scipy.ndimage.interpolation.zoom(img, 1.0/self.config.scale, mode = 'constant' )
        img_lr = cv2.resize(img, (171,171))
        img_lr = img_lr.astype(np.float32)
        img_lr = np.expand_dims(img_lr,axis=0)
        self.sess.run(tf.initialize_all_variables())
        load(self)
        hr = self.generator(img_lr).eval()
        hr = hr.squeeze()
        self.imshow(hr, is_gray=False)
    def get_color_image():
        img_color = img_color.transpose(2,0,1)
        Y, Cb, Cr = img_color[0], img_color[1], img_color[2]

    def imshow(self, img, is_gray = True):
        import matplotlib.pyplot as plt
        if is_gray:
            plt.imshow(img, cmap = 'gray')
        else:
            plt.imshow(img)
        plt.show()
    def generator(self, z):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform = True),weights_regularizer=slim.l2_regularizer(0.00005)):
            x = tf.image.resize_images(z, [self.config.label_size, self.config.label_size], tf.image.ResizeMethod.BICUBIC)
            h1 = slim.conv2d(z, 64, [3,3], activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, scope ='g_h1')
            #h2 = resnet_block(h1, 1, self.gf_dim)
            h3 = slim.conv2d(h1, 32, [3,3], activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, scope ='g_h3')
            #h5 = slim.conv2d(h3, 1, [1,1], activation_fn = None, normalizer_fn = slim.batch_norm, scope ='g_h5')
            h4 = slim.conv2d(h3, 64, [3,3], activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, scope ='g_h4')
            h5 = slim.conv2d(h4, 16, [3,3], activation_fn = None, normalizer_fn = slim.batch_norm, scope ='g_h5')
            h6 = PS(h5, 4)
            return x,tf.nn.sigmoid(h6)
    def discriminator(self,x):
        with slim.arg_scope([slim.conv2d], weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(), normalizer_fn = slim.batch_norm,activation_fn = tf.nn.relu, stride=2):
            conv1 = slim.conv2d(x, self.df_dim, [3,3], scope = 'd_conv1')
            conv2 = slim.conv2d(conv1, 2*self.df_dim, [3,3], scope = 'd_conv2')
            conv3 = slim.conv2d(conv2, self.df_dim, [3,3], scope = 'd_conv3')
            #fc1 = slim.fully_connected(slim.flatten(conv3), 256, scope = 'd_fc1', activation_fn = tf.nn.relu, normalizer_fn  = slim.batch_norm)
            fc2 = slim.fully_connected(slim.flatten(conv3), 1, scope = 'd_fc2', activation_fn = None)
            fc2 = tf.squeeze(fc2)
            return fc2
