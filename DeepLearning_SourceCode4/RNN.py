import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)
    lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden), rnn.BasicLSTMCell(num_hidden)])
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs/rnn', sess.graph)
    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Loss=" + "{:.4f}".format(loss) + ", Accuracy= " + "{:.3f}".format(acc))
    print("Optimization Finished!")
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    final_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
    print("Testing Accuracy:", final_accuracy)
    writer.close()