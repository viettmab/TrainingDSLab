import random

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

NUM_CLASSES = 20


class MLP:
    def __init__(self, vocab_size, hidden_size):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size

    def build_graph(self):
        self._X = tf.placeholder(tf.float32, shape=[None, self._vocab_size])
        self._real_Y = tf.placeholder(tf.int32, shape=[None, ])

        weights_1 = tf.get_variable(name='weights_input_hidden',
                                    shape=(self._vocab_size, self._hidden_size),
                                    initializer=tf.random_normal_initializer(seed=2018)
                                    )
        biases_1 = tf.get_variable(name='biases_input_hidden',
                                   shape=(self._hidden_size),
                                   initializer=tf.random_normal_initializer(seed=2018)
                                   )
        weights_2 = tf.get_variable(name='weights_hidden_output',
                                    shape=(self._hidden_size, NUM_CLASSES),
                                    initializer=tf.random_normal_initializer(seed=2018)
                                    )
        biases_2 = tf.get_variable(name='biases_hidden_output',
                                   shape=(NUM_CLASSES),
                                   initializer=tf.random_normal_initializer(seed=2018)
                                   )
        hidden = tf.matmul(self._X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weights_2) + biases_2
        labels_one_hot = tf.one_hot(indices=self._real_Y, depth=NUM_CLASSES, dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
        loss = tf.reduce_mean(loss)
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels, loss

    def trainer(self, loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # An Operation that updates the variables
        return train_op


# Preprocessing the data
class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()

        self._data = []
        self._labels = []
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index, value = int(token.split(':')[0]), float(token.split(':')[1])
                vector[index] = value
            self._data.append(vector)
            self._labels.append(label)
        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._num_epoch = 0
        self._batch_id = 0

    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1  # Turn to new epoch
            self._batch_id = 0
            # Shuffle data
            indices = list(range(len(self._data)))
            random.seed(2018)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]

        return self._data[start:end], self._labels[start:end]


# Read the data from datasets
def load_dataset():
    train_data_reader = DataReader(
        data_path='../datasets/20news-bydate/train_tf_idf.txt',
        batch_size=50,
        vocab_size=vocab_size
    )
    test_data_reader = DataReader(
        data_path='../datasets/20news-bydate/test_tf_idf.txt',
        batch_size=50,
        vocab_size=vocab_size
    )
    return train_data_reader, test_data_reader


# Save parameters to datasets
def save_parameters(name, value, epoch):
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    if len(value.shape) == 1: # is a list
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number) for number in value[row]])
                                 for row in range(value.shape[0])])
    with open('../saved_paras/' + filename, 'w') as f:
        f.write(string_form)


# Get the value of parameters from datasets
def restore_parameters(name, epoch):
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    with open('../saved_paras/' + filename) as f:
        lines = f.read().splitlines()
    if len(lines) == 1:
        value = [float(number) for number in lines[0].split(',')]
    else:
        value = [[float(number) for number in lines[row].split(',')]
                 for row in range(len(lines))]
    return value



if __name__ == '__main__':
    # create a computation graph
    with open('../datasets/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    mlp = MLP(vocab_size=vocab_size, hidden_size=50)
    predicted_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss=loss, learning_rate=0.1)

    # Run the computational graph
    with tf.Session() as sess:
        train_data_reader, test_data_reader = load_dataset()
        step, MAX_STEP = 0, 10000

        sess.run(tf.global_variables_initializer())
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            plabels_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict={
                    mlp._X: train_data,
                    mlp._real_Y: train_labels
                }
            )
            step += 1
            print('Step: {}, loss: {}'.format(step, loss_eval))
            if (train_data_reader._batch_id == 0):
                trainable_variables = tf.trainable_variables()
                for variable in trainable_variables:
                    save_parameters(
                        name=variable.name,
                        value=variable.eval(session=sess),
                        epoch=train_data_reader._num_epoch
                    )

    # Evaluate model in test data
    with tf.Session() as sess:
        epoch = train_data_reader._num_epoch

        trainable_variables = tf.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)

        num_true_preds = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabels_eval = sess.run(
                predicted_labels,
                feed_dict={
                    mlp._X: test_data,
                    mlp._real_Y: test_labels
                }
            )
            matches = np.equal(test_plabels_eval, test_labels)
            num_true_preds += np.sum(matches.astype(float))

            if test_data_reader._batch_id == 0:
                break  # Break after 1 epoch
        print('Epoch:', epoch)
        print('Accuracy on test data:', num_true_preds / len(test_data_reader._data))

