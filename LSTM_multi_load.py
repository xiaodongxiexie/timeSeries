#coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

saver=tf.train.import_meta_graph('data/tf0803/model/model.ckpt.meta')

graph=tf.get_default_graph()
x=graph.get_operation_by_name('x').outputs[0]
y=graph.get_operation_by_name('y').outputs[0]
pred=graph.get_collection('pred')
accuracy=graph.get_collection('accuracy')


pre=[]
with tf.Session() as sess:
    saver.restore(sess,'data/tf0803/model/model.ckpt')
    test_data=mnist.test.images.reshape((-1,28,28))[:500]
    test_label=mnist.test.labels[:500]
#    for i in range(128):
#        start=i*1000
#        end=(i+1)*1000
#        #x_arr=data_arr[start:end]
#        predict=sess.run(pred[0],feed_dict={x:x_arr})
#        final=tf.argmax(predict,1)
#        print('index:{0}'.format(i))
#        pre.append(final)
    
    acc=sess.run(accuracy, feed_dict={x: test_data, y: test_label})
    print('test accuracy is {:.6f}'.format(acc[0]))
