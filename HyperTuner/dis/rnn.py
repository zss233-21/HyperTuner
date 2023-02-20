import numpy

import tensorflow as tf
from keras.backend import set_session
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os
import json


def parameter():
    filename = "/home/zss/PycharmProjects/pythonProject/param_list.txt"
    with open(filename, 'r') as f1:
        # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
        # 然后将每个元素中的不同信息提取出来
        lines = f1.readlines()
        # i变量，由于这个txt存储时有空行，所以增只读偶数行，主要看txt文件的格式，一般不需要
        # j用于判断读了多少条，step为画图的X轴
        for line in lines:
            t = line.split(' ')
            d = t[1]
            lr = t[2]
            b = t[3]
            inter = t[5]
            intra = t[6]
            # topwords = t[7]
            # lenth = t[8]
    return float(d), float(lr), int(b), int(inter), int(intra)


dropout_rate, learning_rate, batch_size, inter_op, intra_op = parameter()

session_config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=inter_op,
    intra_op_parallelism_threads=intra_op)
# s = tf.compat.v1.Session(config=session_config)
set_session(tf.compat.v1.Session(config=session_config))

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.1.155:20000"],
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# Wrap data in Dataset objects.
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# The batch size must now be set on the Dataset objects.
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_data = train_data.with_options(options)
val_data = val_data.with_options(options)


def create_rnn(dropout):
    embedding_vecor_length = batch_size
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=dropout))
    model.add(Dense(1, activation='sigmoid'))
    print("rnn")
    return model


with strategy.scope():
    model = create_rnn(dropout_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    h = model.fit(train_data, epochs=1, batch_size=batch_size)
    f = open("/home/zss/PycharmProjects/pythonProject/rnn/test.csv", "a")
    print(h.history['accuracy'], file=f)






