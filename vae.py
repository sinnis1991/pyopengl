import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import itertools
from glob import glob
from gl_model import gl_ob
from ops import *
from utils import *


checkpoint_dir = "./vae/checkpoint"
out_path = "./vae/out"
test_path = "./vae/test"
if_train = False
if_decompress = True

gl = gl_ob(128, 128, 64, mode='X')
gl.show_example()

mb_size = 64
z_dim = 100
X_dim = 128*128*1
h_dim = 128*8
image_size = 128
c = 0
lr = 1e-3

qx_dim = 128
pz_dim = 128

def plot(samples, name, path):
    batchSz = np.shape(samples)[0]
    nRows = np.ceil(batchSz / 8)
    nCols = min(8, batchSz)
    save_images(samples, [nRows, nCols],
                os.path.join(path, name))


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

q_bns = [batch_norm(name='q_bn{}'.format(i, )) for i in range(1, 5)]

X = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])


def Q(X):
    # X dim:128 x 128 x 1
    with tf.variable_scope("QX") as scope:
        h0 = tf.nn.relu(conv2d(X, qx_dim, name='q_h0_conv'))   # 64 x 64 x qx_dim
        h1 = tf.nn.relu(q_bns[0](conv2d(h0, qx_dim * 2, name='q_h1_conv'),True)) # 32 x 32 x qx_dim*2
        h2 = tf.nn.relu(q_bns[1](conv2d(h1, qx_dim * 4, name='q_h2_conv'),True)) # 16 x 16 x qx_dim*4
        h3 = tf.nn.relu(q_bns[2](conv2d(h2, qx_dim * 8, name='q_h3_conv'),True)) # 8 x 8 x qx_dim*8
        h4 = tf.nn.relu(q_bns[3](conv2d(h3, qx_dim * 16, name='q_h4_conv'),True)) # 4 x 4 x qx_dim*16
        # z_mu = linear(tf.reshape(h4, [-1, 4*4*qx_dim*16]), 100, name = 'q_h4_linear_mu') # 4 x 4 x qx_dim x 16 = 8192
        # z_logvar = linear(tf.reshape(h4, [-1, 4*4*qx_dim*16]), 100, name = 'q_h4_linear_logvar')
        # return z_mu, z_logvar
        z = linear(tf.reshape(h4, [-1, 4*4*qx_dim*16]), 100, name = 'q_h4_linear_mu') # 4 x 4 x qx_dim x 16 = 8192
        return z



def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

log_size = int(math.log(image_size) / math.log(2))
p_bns = [batch_norm(name='p_bn{}'.format(i, )) for i in range(log_size - 1)]

def P(z, reuse=False):
    with tf.variable_scope("PZ") as scope:
        if reuse:
            scope.reuse_variables()

        z_, h0_w, h0_b = linear(z, pz_dim * 16 * 4 * 4, 'p_h0_linear', with_w=True)

        hs0 = tf.reshape(z_, [-1, 4, 4, pz_dim * 16])
        hs0 = tf.nn.relu(p_bns[0](hs0,True))

        hs1, _, _ = conv2d_transpose(hs0,[mb_size, 8, 8, pz_dim * 8],
                                     name="p_h1",with_w=True)
        hs1 = tf.nn.relu(p_bns[1](hs1,True))

        hs2, _, _ = conv2d_transpose(hs1, [mb_size, 16, 16, pz_dim * 4],
                                     name="p_h2", with_w=True)
        hs2 = tf.nn.relu(p_bns[2](hs2, True))

        hs3, _, _ = conv2d_transpose(hs2, [mb_size, 32, 32, pz_dim * 2],
                                     name="p_h3", with_w=True)
        hs3 = tf.nn.relu(p_bns[3](hs3, True))

        hs4, _, _ = conv2d_transpose(hs3, [mb_size, 64, 64, pz_dim],
                                     name="p_h4", with_w=True)
        hs4 = tf.nn.relu(p_bns[4](hs4, True))

        hs5, _, _ = conv2d_transpose(hs4, [mb_size, 128, 128, 1],
                                     name="p_h5", with_w=True)

        # return tf.sign(tf.nn.tanh(hs5)), tf.nn.tanh(hs5)
        # return tf.sign(tf.nn.sigmoid(tf.nn.tanh(hs5))), tf.nn.tanh(hs5)
        # return tf.nn.sigmoid(hs5), hs5
        return tf.nn.tanh(hs5), hs5


# =============================== TRAINING ====================================
# imgs = dataset_files(data_path)
# imgs.sort()
# batch_idxs = len(imgs) // mb_size

# z_mu, z_logvar = Q(X)
# z_sample = sample_z(z_mu, z_logvar)
# # im, logits = P(z_sample)
# bit, logits = P(z_sample)

z_logits = Q(X)
z_sample =  tf.nn.tanh(z_logits)
bit, logits = P(z_sample)
# Sampling from random z
# X_samples,_ = P(z, True)
bit_samples, X_samples = P(z, True)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(bit - X)), 1)
# recon_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(logits - X)), 1)
# recon_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.nn.sigmoid_cross_entropy_with_logits(
#                                                     logits=logits,
#                                                     labels=X)), 1)

# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
# kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
# vae_loss = tf.reduce_mean(recon_loss + kl_loss)
vae_loss = tf.reduce_mean(recon_loss)

solver = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(vae_loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("""
    
        ======
        An existing model was found in the checkpoint directory.
        ======
    
        """)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("""
    
        ======
        An existing model was not found in the checkpoint directory.
        Initializing a new one.
        ======
    
        """)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)


    for it in range(10000,30000):

        batch_im, batch_y = gl.draw_ob()
        X_mb = np.reshape(batch_im/127.5-1.,(mb_size,image_size,image_size,1))

        _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

        print('Iter: {}   Loss: {:.4}'.format(it, loss))

        if it % 100 == 0:

            # print('Iter: [{}]   Loss: {:.4}'.format(it, loss))


            samples = sess.run(bit_samples, feed_dict={z: np.random.uniform(-1,1, [64,z_dim])})

            plot(samples, "out_{}.png".format(it), out_path)
            print("out saved in file: %s" % out_path)

            test_im, test_y = gl.draw_ob()
            X_test = np.reshape(test_im / 127.5-1.,(mb_size,image_size,image_size,1))
            samples_t = sess.run(bit, feed_dict={X: X_test})
            plot(samples_t, "{}_test.png".format(it), test_path)
            plot(X_test, "{}_real.png".format(it), test_path)
            print("test saved in file: %s" % test_path)


            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver.save(sess, checkpoint_dir+"/VAE.model")
            print("Model saved in file: %s" % checkpoint_dir)

# if if_decompress:
#
#     imgs_t = dataset_files(data_t_path)
#     imgs_t.sort()
#
#     test_files = imgs_t[:64]
#     test_mb = [get_image(test_file, 64)
#                              for test_file in test_files]
#     samples_t = sess.run(logits, feed_dict ={X:test_mb} )
#     plot(samples_t, "test.png".format(1),test_path)
