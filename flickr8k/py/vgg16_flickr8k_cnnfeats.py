########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
# Modified to read from flickr8k by Mark Hasegawa-Johnson, 3/22/2017
# Modified to generate CNN feature matrices, instead of FCN vectors, 7/21/2017
########################################################################################

import tensorflow as tf
import numpy as np
import re
from scipy.misc import imread, imresize
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'frossard_vgg16'))
from vgg16class import vgg16

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, '../frossard_vgg16/vgg16_weights.npz', sess)

    flickrpath = '.'
    
    jpgpath = 'jpg'
    imgfiles = os.listdir(jpgpath)
    outputdir = 'cnnfeats'
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    
    print('Running VGG for {} images'.format(len(imgfiles)))
    for imgfile in imgfiles:
        img1 = imread(flickrpath+'/jpg/'+imgfile, mode='RGB')
        img1 = imresize(img1, (224,224))
        print('Running VGG with image {}'.format(imgfile))
        conv5_3 = sess.run(vgg.conv5_3,  feed_dict={vgg.imgs: [img1]})[0]
        outputfile = outputdir + '/' + re.sub(r'_.\.jpg','',imgfile)
        print('Computed {} feats for {}, save in {}.npz'.format(conv5_3.shape,imgfile,outputfile))        
        np.savez(outputfile,conv5_3)

        
