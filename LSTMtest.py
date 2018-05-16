import tensorflow as tf
from tensorflow.contrib import rnn
from random import *
import numpy as np


def sample(a):
    a = np.log(a) / 0.5
    dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(a))
    res = np.zeros_like(a, np.float32)
    res[np.random.choice(choices, p=dist)] = 1
    return res


# reading the text

text = open("shah.txt", "r").read()
chars = sorted(list(set(text)))
char_to_index = dict((ch, i) for i, ch in enumerate(chars))
index_to_char = dict((i, ch) for i, ch in enumerate(chars))
dict_size = len(char_to_index)
text_len = len(text)

for ch in chars:
    print ch
# set hyper parameters
learning_rate = 0.01
seq_len = 100
stride = 35
hidden_size = 128
batch_size = 100
test_size = 5
epoch_size = 50

#preparing the data
train_data = []
target_data = []

for i in range(0, text_len - seq_len, seq_len - stride):
    train_data.append([char_to_index[ch] for ch in text[i:i+seq_len]])
    target_data.append(char_to_index[text[i+seq_len]])

x_train = np.zeros((len(train_data), seq_len, dict_size))
y_train = np.zeros((len(target_data), seq_len, dict_size))

for seq_index, seq in enumerate(train_data):
    for i in range(seq_len):
        x_train[seq_index, i, seq[i]] = 1
        if i < seq_len-1:
            y_train[seq_index, i, seq[i+1]] = 1
        else:
            y_train[seq_index, i, target_data[seq_index]] = 1

# network mafakaaaaaaa

x = tf.placeholder(tf.float32, (None,seq_len,dict_size))
y = tf.placeholder(tf.float32, (None,seq_len,dict_size))

cell = rnn.BasicLSTMCell(hidden_size)
output, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
print state
tf.summary.histogram("outputs", output)

print output.shape
output_flat = tf.reshape(output, [-1, hidden_size])

softmax_w = tf.Variable(tf.truncated_normal(shape=[hidden_size, dict_size], mean=0, stddev=0.1))
softmax_b = tf.Variable(tf.zeros(dict_size))
tf.summary.histogram("w", softmax_w)
tf.summary.histogram("b", softmax_b)

prediction = tf.matmul(output_flat, softmax_w) + softmax_b
tf.summary.histogram("Predictions", prediction)

predictoin_flat = tf.reshape(prediction, [-1, dict_size])
y_flat = tf.reshape(y, [-1, dict_size])

tf.summary.histogram("targets", y)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_flat,logits=predictoin_flat))
train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

next_char_prediction = tf.nn.softmax(prediction[-1])

# training
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# showing shit
writer = tf.summary.FileWriter("./logs/", sess.graph)
merge = tf.summary.merge_all()

#saving shit

saver = tf.train.Saver()
saver.restore(sess=sess, save_path='./save/model')

for epoch in range(epoch_size):
    saver.save(sess,'./save/model')

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        print i
        sess.run(train_op,feed_dict={x: x_batch, y: y_batch})


#   Test the model:
    print (
            "-------------------------------------------------------- Epoch %d --------------------------------------------------------" % (
            epoch + 1))
    report_loss, merge_smry = sess.run([loss, merge], feed_dict={x: x_train, y: y_train})
    print ("Loss: %f" % (report_loss))
    tf.summary.scalar("loss", report_loss)
    smry = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=report_loss)])
    writer.add_summary(smry, epoch)
    writer.add_summary(merge_smry, epoch)
    print ("\nInput: ")
    test_input_chars = ''
    rand_idx = randint(0, len(x_train))
    x_test = x_train[rand_idx]
    x_test_idx = [np.where(w == 1)[0][0] for w in x_test]
    for w in train_data[rand_idx]:
        test_input_chars += index_to_char[w]
    print (" %s " % (test_input_chars))
    # Predict next 400 characters:
    res_chars = ''
    for i in range(400):
        # Predict the next character:
        x_test = np.reshape(x_test, [1, seq_len, dict_size])
        predicted = sess.run(next_char_prediction, feed_dict={x: x_test})
        probs = sample(predicted)
        probs_char = index_to_char[np.argmax(probs)]
        res_chars += probs_char
        # Add predicted char to x_test sequence:
        x_test = np.append(np.delete(x_test, 0, axis=1), np.reshape(probs, [1, 1, dict_size]), axis=1)
    print ("\nGenerated text: ")
    print (" %s \n\n" % (res_chars))


