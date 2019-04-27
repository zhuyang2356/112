# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 23:51:01 2019

@author: admin
"""
words_list=np.load(r'C:\nlpinaction\learning-nlp-master\chapter-8\sentiment-analysis\wordsList.npy')
words_list=words_list.tolist()
words_list=[word.decode('utf-8') for word in words_list]
len(words_list)
words_list[-200000]
word_Vectors=np.load(r'C:\nlpinaction\learning-nlp-master\chapter-8\sentiment-analysis\wordVectors.npy')
word_Vectors.shape
word_vectors[0,1]

#-------------------------------------
max_seq_num=250
max_seq_length=300

batch_size=24
lstm_units=64
num_labels=2
iterations=20000
num_dimensions=250
import numpy as np
ids = np.load(r'C:\nlpinaction\learning-nlp-master\chapter-8\sentiment-analysis\idsMatrix.npy')
ids[1]

from random import randint
#训练集从0-11500 正面，13500-25000反面。验证集11500-12500 正面，12500-13500反面
def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels

import tensorflow as tf
tf.reset_default_graph()

labels=tf.placeholder(tf.float32,[batch_size,num_labels])
input_data=tf.placeholder(tf.int32,[batch_size,max_seq_num])

data=tf.Variable(tf.zeros([batch_size,max_seq_length,num_dimensions]),dtype=tf.float32)
data=tf.nn.embedding_lookup(word_Vectors,input_data)

lstmCell=tf.contrib.rnn.BasicLSTMCell(lstm_units)
tf.__version__
lstmCell=tf.contrib.rnn.DropoutWrapper(cell=lstmCell,output_keep_prob=0.75)
value,_=tf.nn.dynamic_rnn(lstmCell,data,dtype=tf.float32)

weight=tf.Variable(tf.truncated_normal([lstm_units,num_labels]))
bias=tf.Variable(tf.constant(0.1,shape=[num_labels]))
value=tf.transpose(value,[1,0,2])
last=tf.gather(value,int(value.get_shape()[0])-1)
prediction=(tf.matmul(last,weight)+bias)

correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
print(tf.argmax(0.5,1))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=labels))
optimizer=tf.train.AdamOptimizer().minimize(loss)

import datetime
sess=tf.InteractiveSession()
#tensorflow用GPU
tf.device("/gpu:0")
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('Loss',loss)
tf.summary.scalar('Accuracy',accuracy)
merged=tf.summary.merge_all()
logdir='tensorboard/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'/'
writer=tf.summary.FileWriter(logdir,sess.graph)

for i in range(iterations):
    nextBatch,nextBatchLabels=get_train_batch()
    sess.run(optimizer,{input_data:nextBatch,labels:nextBatchLabels})
#    每50次写入一次leadboard
    if i%50==0:
        summary=sess.run(merged,{input_data:nextBatch,labels:nextBatchLabels})
        writer.add_summary(summary,i)
    if i%10000==0 and i !=0:
        save_path=saver.save(sess,r'C:\nlpinaction\tensorflow_saver',global_step=i)
        print('saved to %s',save_path)
writer.close()

print('a is a %s',sb)
sb='姬无命'    
tensorboard --logdir=tensorboard
http://localhost:6006/--host=127.0.0.1


