#!/usr/bin/env python
# coding=utf-8
import os
import tensorflow as tf
def save(model, step):
    checkpoint_dir = model.checkpoint_dir
    model_name = model.name
    model_dir = "%s_%s" % (model_name, model.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.saver.save(model.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

def load(model):
    checkpoint_dir = model.checkpoint_dir
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % (model.name, model.image_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        model.saver.restore(model.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

