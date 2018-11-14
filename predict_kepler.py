import numpy as np
import pickle
import tensorflow as tf
import os, glob
import matplotlib
matplotlib.use('Agg');
from matplotlib import pyplot as plt
import gc
import imp
import pandas as pd

train = pd.read_csv('exoTrain.csv');
test = pd.read_csv('exoTest.csv');

train_out = train['LABEL'];
train_in = train.iloc[:,1:];
train_in = (train_in - train_in.mean()) / train_in.std();

train_out_onehot = np.zeros((train_out.shape[0], 2))
train_out_onehot[:,0] = (train['LABEL'] == 1)*1;
train_out_onehot[:,1] = (train['LABEL'] == 2)*1;

print(train_in.shape, train_out_onehot.shape);

test_out = test['LABEL'];
test_in = test.iloc[:,1:];
test_in = (test_in - test_in.mean()) / test_in.std();

test_out_onehot = np.zeros((test_out.shape[0], 2))
test_out_onehot[:,0] = (test['LABEL'] == 1)*1;
test_out_onehot[:,1] = (test['LABEL'] == 2)*1;

print(test_in.shape, test_out_onehot.shape);

num_classes = 2;
input_length = train_in.shape[1];
train_size = train_in.shape[0];
gamma = 1

X = tf.placeholder(tf.float32, [None, input_length])
Y = tf.placeholder(tf.float32, [None, num_classes]) 

learning_rate = tf.placeholder(tf.float32)

def get_next_batch(batch_size):
    idx = np.arange(train_size);
    np.random.shuffle(idx);
    return train_in.iloc[idx[:batch_size]], train_out_onehot[idx[:batch_size]];    

dense_weights = {
    'wd1': tf.Variable(tf.random_normal([input_length, 1000])),
    'wd2': tf.Variable(tf.random_normal([1000, num_classes]))
}

dense_biases = {
    'bd1': tf.Variable(tf.random_normal([1])),
    'bd2': tf.Variable(tf.random_normal([1])),
}

dense_regularizer = {
    'wd1' : tf.nn.l2_loss(dense_weights['wd1']),
    'wd2' : tf.nn.l2_loss(dense_weights['wd2']),
}

def dense_net(x, dense_weights, dense_biases):
    
    fc1 = tf.add(tf.matmul(x, dense_weights['wd1']), dense_biases['bd1']);
    out = tf.add(tf.matmul(fc1, dense_weights['wd2']), dense_biases['bd2']);
    
    return out;

logits = dense_net(X, dense_weights, dense_biases);
prediction = tf.nn.softmax(logits)

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y);
loss_op = tf.reduce_sum(entropy)

diff_sq = tf.multiply((logits - Y),(logits - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op + gamma*(dense_regularizer['wd1'] + dense_regularizer['wd2']));

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def run_epoch(sess, num_steps, initial_rate, anneal=False):
    
    sess.run(tf.global_variables_initializer())
    rate = initial_rate;
    
    training_error = [];
    validation_error = [];
    training_loss = [];
    test_accuracy = [];
    
    for step in range(1, num_steps+1): 
        if anneal:
            rate = initial_rate * (1 - step/num_steps);
        
        batch_x, batch_y = get_next_batch(128);
            
        l,t= sess.run([loss_op, train_op], feed_dict={X: batch_x, 
                                                      Y: batch_y, 
                                                      learning_rate: rate})
        training_loss.append(l);
        
        acc = sess.run([accuracy], feed_dict = {X: test_in,
                                                Y: test_out_onehot});
        test_accuracy.append(acc);
       	print(acc);
              
    training_loss = np.array(training_loss); 
    
    plt.figure();
    plt.plot(training_loss);
    plt.savefig('training.png');

    plt.figure();
    plt.plot(test_accuracy);
    plt.savefig('testing.png');

with tf.Session() as sess:
    run_epoch(sess, 1000, 0.01, True);
