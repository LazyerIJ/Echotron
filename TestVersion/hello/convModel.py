import tensorflow as tf
import numpy as np

class CNNModel():

    def __init__(self,time_size,mfcc_dim,filter_sizes,num_filters):

        time_size= time_size
        mfcc_dim = mfcc_dim
        filter_sizes = filter_sizes
        num_filters= num_filters

        self.x = tf.placeholder(tf.float32, [None,time_size,mfcc_dim,1])
        self.y = tf.placeholder(tf.float32, [None,time_size,mfcc_dim,1])

        print(self.x)
        with tf.variable_scope('mfcc_cnn'):
            pooled_outputs = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-bank-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [1, filter_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32,name="mfcc_conv_W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), dtype=tf.float32,name="mfcc_conv_b")
                    conv = tf.nn.conv2d(
                        self.x,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="mfcc_conv")
                    # Apply nonlinearityA
                    '''
                    h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="relu")
                    print('[*]h : {}'.format(h))
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1,1, 2, 1],
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name="mfcc_pool")
                    print('[*]pooled : ' , pooled)
                    '''
                    pooled=conv
                    pooled_outputs.append(pooled)
                    
            
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            mfcc_conv_output = tf.concat(pooled_outputs, 3)

        with tf.variable_scope('auto_encoder'):
            dense_1h = tf.layers.dense(inputs=mfcc_conv_output, 
                                       units=256, 
                                       activation=None)
            dropout_1h = tf.layers.dropout(inputs=dense_1h, 
                                           rate=0.4)

            dense_2h = tf.layers.dense(inputs=dropout_1h, 
                                       units=128, 
                                       activation=None)
            dropout_2h = tf.layers.dropout(inputs=dense_2h, 
                                           rate=0.4)

            dense_3h = tf.layers.dense(inputs=dropout_2h, 
                                       units=32, 
                                       activation=None)
            dropout_3h = tf.layers.dropout(inputs=dense_3h, 
                                           rate=0.4)

        with tf.variable_scope('logits'):
            self.logits = tf.layers.dense(inputs=dropout_3h,units=1)
            pred = tf.argmax(self.logits,axis=1)
        


        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.logits,self.y))))
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train = optimizer.minimize(self.loss)


