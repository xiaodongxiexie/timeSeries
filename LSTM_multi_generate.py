#coding: utf-8


import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
#%matplotlib inline

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

def main(lr=0.01,ti=10000,bz=500,opt='AdamOptimizer'):
    '''
    params:
            lr: 学习率
            ti: 迭代次数
            bz：batch_size
            opt: 优化器
    '''

    tf.reset_default_graph()

    #定义参数
    learning_rate = lr            #学习率
    training_iters = ti           #最大迭代次数
    batch_size =  bz              #minibatch选择的大小

    #网络参数
    n_input = 28                
    n_steps = 28                    

    n_hidden  = 100                # 隐藏层结点数
    n_hidden2 = 120
    n_hidden3 = 150
    n_hidden4 = 120
    n_hidden5 = 100
    n_hidden6 = 200

    n_classes = 10                  # 标签数量
    num_layers= 6                   #cell的层数            

    # 生成tf图
    x = tf.placeholder("float", [None, n_steps, n_input],name='x')
    y = tf.placeholder("float", [None, n_classes],name='y')

    # 定义W,B 
    weights = {
        'out':  tf.Variable(tf.random_normal([n_hidden, n_hidden2])),
        'out2': tf.Variable(tf.random_normal([n_hidden2, n_hidden3])),
        'out3': tf.Variable(tf.random_normal([n_hidden3, n_hidden4])),
        'out4': tf.Variable(tf.random_normal([n_hidden4, n_hidden5])),
        'out5': tf.Variable(tf.random_normal([n_hidden5, n_hidden6])),
        'out6': tf.Variable(tf.random_normal([n_hidden6, n_classes])),
    }
    biases = {
         'out':  tf.Variable(tf.random_normal([n_hidden2])),
         'out2': tf.Variable(tf.random_normal([n_hidden3])),
         'out3': tf.Variable(tf.random_normal([n_hidden4])),
         'out4': tf.Variable(tf.random_normal([n_hidden5])),
         'out5': tf.Variable(tf.random_normal([n_hidden6])),
         'out6': tf.Variable(tf.random_normal([n_classes])),
    }

    def real_len(x):
        dense_sign = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
        length = tf.reduce_sum(input_tensor=dense_sign, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def cut_output(output,length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(start=0, limit=batch_size)*max_length + (length-1) 
        flat = tf.reshape(output, [-1,output_size]) 
        result = tf.gather(flat, index) 
        return result

    def RNN(x, weights, biases,is_training=True):
        def LSTM(hidden,keep_prob=0.8):
            layer=rnn.GRUCell(hidden,reuse=tf.get_variable_scope().reuse)
            if is_training==True:
                return rnn.DropoutWrapper(cell=layer,output_keep_prob=keep_prob)
            else:
                return layer
        layer_out=rnn.GRUCell(200,reuse=tf.get_variable_scope().reuse)
        hidden_list = [n_hidden]
        layers=rnn.MultiRNNCell(cells=([LSTM(hidden) for  hidden in hidden_list]+[layer_out]))
        _initial_state=layers.zero_state(batch_size, tf.float32)
        
        length = real_len(x)
        outputs,states=tf.nn.dynamic_rnn(cell=layers,inputs=x,dtype=tf.float32,
            sequence_length=length,initial_state=_initial_state)

        outputs=cut_output(outputs,length)

        return tf.add(tf.matmul(outputs, weights['out6']), biases['out6'])
    
    with tf.variable_scope("rnn") as scope:   
        pred = RNN(x, weights, biases) 
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = getattr(tf.train, opt)(learning_rate=learning_rate).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))   
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    with tf.variable_scope(scope,reuse=True):   
        pred_test = RNN(x, weights, biases, is_training=False)
        correct_pred_test=tf.equal(tf.argmax(pred_test,1), tf.argmax(y,1))    
        accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))
        tf.add_to_collection('pred',pred_test)
        tf.add_to_collection('accuracy',accuracy_test)

    init = tf.global_variables_initializer()

    loss_list = []
    acc_list = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        test_len=batch_size
        test_data = mnist.test.images[:test_len].reshape(-1, n_steps, n_input)
        test_label = mnist.test.labels[:test_len]
        
        train_acc_best = 0
        test_acc_best = 0
        while step * batch_size <= training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x=batch_x.reshape((batch_size,n_steps,n_input))

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % 2 == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                loss_list.append(loss)
                acc_test=sess.run(accuracy_test, feed_dict={x: test_data, y: test_label})
                acc_list.append(acc_test)
                if step % 10 == 0:
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc)+", Test Accuracy= "+"{:.6f}".format(acc_test))
                    if acc >= train_acc_best and acc_test > test_acc_best:
                        import os
                        train_acc_best, test_acc_best = acc, acc_test
                        if not os.path.exists('data/tf0803/model'):
                            os.makedirs('data/tf0803/model/')
                        saver.save(sess, './data/tf0803/model/model.ckpt')                       
            step += 1
        print("Optimization Finished!")
        print(test_acc_best)

        
    def pplot(seq, note, num=0):
        plt.figure(num)
        plt.plot(seq)
        plt.title(note)
        plt.grid();
    pplot(loss_list, u'loss')
    pplot(acc_list, u'accuracy',1)
    plt.show()


if __name__ == '__main__':
    main()

