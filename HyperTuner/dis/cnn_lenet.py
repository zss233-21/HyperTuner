import time
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json
from keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

num_workers = 3
num_epochs = 4


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
    return float(d), float(lr), int(b), int(inter), int(intra)


def create_model_lenet():
    # model = tf.keras.applications.VGG16(weights=None, classes=10)
    model = Sequential()

    # 选取6个特征卷积核，大小为5∗5(不包含偏置),得到66个特征图，每个特征图的大小为32−5+1=2832−5+1=28，
    # 也就是神经元的个数由10241024减小到了28∗28=78428∗28=784。
    # 输入层与C1层之间的参数:6∗(5∗5+1)
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))

    # 这一层的输入为第一层的输出，是一个28*28*6的节点矩阵。
    # 本层采用的过滤器大小为2*2，长和宽的步长均为2，所以本层的输出矩阵大小为14*14*6。
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 本层的输入矩阵大小为14*14*6，使用的过滤器大小为5*5，深度为16.本层不使用全0填充，步长为1。
    # 本层的输出矩阵大小为10*10*16。本层有5*5*6*16+16=2416个参数
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))

    # 本层的输入矩阵大小10*10*16。本层采用的过滤器大小为2*2，长和宽的步长均为2，所以本层的输出矩阵大小为5*5*16。
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 本层的输入矩阵大小为5*5*16，在LeNet-5论文中将这一层称为卷积层，但是因为过滤器的大小就是5*5，#
    # 所以和全连接层没有区别。如果将5*5*16矩阵中的节点拉成一个向量，那么这一层和全连接层就一样了。
    # 本层的输出节点个数为120，总共有5*5*16*120+120=48120个参数。
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    # 本层的输入节点个数为120个，输出节点个数为84个，总共参数为120*84+84=10164个 (w + b)
    model.add(Dense(84, activation='relu'))
    # 本层的输入节点个数为84个，输出节点个数为10个，总共参数为84*10+10=850
    model.add(Dense(10, activation='softmax'))
    #    model = tf.keras.applications.VGG16(weights=None, classes=10)
    # model = tf.keras.applications.MobileNetV2(weights=None, classes=10)
    return model


def create_model_vgg(dropout2_rate):
    # 创建序列模型
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same', activation='relu',
                     kernel_initializer='uniform'))
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    #    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout2_rate))
    model.add(Dense(10, activation='softmax'))
    return model


def resize(image, label):
    image = tf.image.resize(image, [28, 28]) / 255.0
    # image = tf.image.resize(image, [224, 224]) / 255.0
    return image, label


os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.1.155:20000", "192.168.1.152:20001", "192.168.1.153:20002" ],
    },
    'task': {'type': 'worker', 'index': 0}
})

dropout_rate, learning_rate, batch_size_per_replica, inter_op, intra_op = parameter()

session_config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=inter_op,
    intra_op_parallelism_threads=intra_op)
# s = tf.compat.v1.Session(config=session_config)
set_session(tf.compat.v1.Session(config=session_config))

strategy = tf.distribute.MultiWorkerMirroredStrategy()
batch_size = batch_size_per_replica * num_workers

dataset = tfds.load("mnist", split=tfds.Split.TRAIN, as_supervised=True)
val_data = tfds.load("mnist", split=tfds.Split.TEST, as_supervised=True)
# dataset = tfds.load("cats_vs_dogs", split=tfds.Split.TRAIN, as_supervised=True)
# data = tf.data.Dataset.from_tensor_slices(dataset)
dataset = dataset.map(resize).shuffle(1024).batch(batch_size)
val_data = val_data.map(resize).shuffle(1024).batch(batch_size)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
dataset = dataset.with_options(options)
val_data = val_data.with_options(options)

with strategy.scope():
    model = create_model_lenet()
    # model = create_model_vgg(dropout_rate)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    h = model.fit(dataset, epochs=num_epochs)
    # f = open("/home/zss/PycharmProjects/pythonProject/lenet/test.csv", "a")
    # print(h.history['accuracy'], file=f)



