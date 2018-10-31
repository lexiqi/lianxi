from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('.',one_hot=True,reshape=False)

import tensorflow as tf

#超参数
learning_rate=0.00001
epochs=10
batch_size=128

test_valid_size=128
n_classes=10
dropout=0.75

#weights and biases
weights={
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
    'out':tf.Variable(tf.random_normal([1024,n_classes]))
}

biases={
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

#卷积
def con2d(x,W,b,stride=1):
    x=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

#最大池化
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

#model
def conv_net(x,weights,biases,dropout):
    #layer1 :28*28*1=>14*14*32
    conv1=con2d(x,weights['wc1'],biases['bc1'])
    conv1=maxpool2d(conv1)

    # layer2: 4*14*32=>7*7*64
    conv2 = con2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2)

    #fully connected layer :7*7*64=>1024
    fc1=tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1=tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,dropout)

    #output layer:class prediction  1024=>10
    out=tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

#session

#tf Graph input
x=tf.placeholder(tf.float32,[None,28,28,1])
y=tf.placeholder(tf.float32,[None,n_classes])
keep_prob=tf.placeholder(tf.float32)

#model
logits=conv_net(x,weights,biases,keep_prob)

#define loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#Accuracy
correct_pred=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Initializing the variables
init=tf.global_variables_initializer()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x,batch_y=mnist.train.next_batch(batch_size)

            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})

            #calculate batch loss and accuracy
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y,keep_prob:1.})

            valid_acc=sess.run(accuracy,feed_dict={x:mnist.validation.images[:test_valid_size],y:mnist.validation.labels[:test_valid_size],keep_prob:1.})


            print('Epoch:{:>2},Batch{:>3},Loss:{:>10.4f},ValidationAcc:{:.6f}'.format(
                epoch+1,
                batch+1,
                loss,
                valid_acc
            ))


    test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images[:test_valid_size],
                                          y:mnist.test.labels[:test_valid_size],
                                          keep_prob:1.})
    print('test acc:{}'.format(test_acc))


