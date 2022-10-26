# -*- coding: utf-8 -*-

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
import numpy as np

# dataset_path = 'data/processed_data/gasser_data.txt'
# generated_path = 'data/generated_data/6vae_generation.txt'
dataset_path = '../../data/processed_data/slaac_privacy_addresses_gasser_data.txt'
generated_path = '../../data/generated_data/dnn_vae_generation_slaac_privacy_addresses.txt'

batch_size = 100
original_dim = 32
latent_dim = 32 # 1
intermediate_dim = 16
epochs = 10


# 读取数据集
def load_data(filename):

    f = open(filename, 'r', encoding='utf-8')
    raw_data = f.readlines()
    f.close()

    # 去除末尾换行符
    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i][:-1]

    # 提取地址字符
    word_data = []
    for address in raw_data:
        address_data = []
        for i in range(len(address)):
            address_data.append(address[i])
        word_data.append(address_data)

    # 将地址字符转换为id
    v6dict = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
        '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15
    }
    data = []
    for address in word_data:
        address_data = []
        for bit in address:
            address_data.append(v6dict[bit])
        data.append(address_data)

    target = np.ones(len(raw_data))
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=0)
    return x_train, x_test, y_train, y_test


def generation_store(target_generation, generation_path=generated_path):
    f = open(generation_path, 'w', encoding='utf-8')
    f.writelines(target_generation)
    f.close()


def run_model():
    # 数据处理
    x_train, x_test, y_train, y_test = load_data(dataset_path)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.astype('float32') / 15.
    x_test = x_test.astype('float32') / 15.

    # 构建encoder
    x = keras.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(x)

    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)

    # 重采样
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # 构建decoder
    decoder_h = layers.Dense(intermediate_dim, activation='relu')
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(x, x_decoded_mean)

    # 构建vae_loss
    # xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
    # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    # mnist vae_loss
    xent_loss = keras.metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    # vae.fit(x_train,
    #         shuffle=True,
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         validation_data=(x_test, None))

    # 压缩表示
    encoder = Model(x, z_mean)
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

    # 构建generator
    decoder_input = layers.Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    # # 噪声采样
    # n = 1000
    # address_size = 32
    # grid = norm.ppf(np.linspace(0.05, 0.95, n))
    #
    # # v6地址生成生成
    # v6_generation = []
    # for i, xi in enumerate(grid):
    #     z_sample = np.array([xi])
    #     x_decoded = generator.predict(z_sample)
    #     v6_generation.append(np.rint(x_decoded * 15))
    #
    # target_generation = []
    # for v6_data in v6_generation:
    #     count = 0
    #     address = ''
    #     for bit in v6_data[0]:
    #         count += 1
    #         address += str(hex(int(bit)))[-1]
    #         if count % 4 == 0 and count != 32:
    #             address += ':'
    #         if count == 32:
    #             address += '\n'
    #     target_generation.append(address)
    #
    # generation_store(target_generation)

    def gen():
        r = generator.predict(np.random.randn(1, latent_dim))[0]
        r = np.rint(r * 15)
        # print(r)
        # r = r.argmax(axis=0)
        return r

    class Evaluate(Callback):
        def __init__(self):
            self.log = []

        def on_epoch_end(self, epoch, logs=None):
            self.log.append(gen())
            # print(self.log)
            gen_address = ""
            count = 0
            gen_address_list = [str(hex(int(i)))[-1] for i in self.log[-1]]
            for i in gen_address_list:
                count += 1
                gen_address += i
                if count % 4 == 0:
                    gen_address += ":"
            gen_address = gen_address[:-1]
            print(gen_address)
            # print(u'          %s'%(self.log[-1]))

    evaluator = Evaluate()

    vae.fit(x_train,
            shuffle=True,
            epochs=10,
            batch_size=64,
            callbacks=[evaluator]
            )

    vae.save_weights('../../models/dnn_vae.model')

    for i in range(20):
        r = gen()
        gen_address = ""
        count = 0
        gen_address_list = [str(hex(int(i)))[-1] for i in r]
        for i in gen_address_list:
            count += 1
            gen_address += i
            if count % 4 == 0:
                gen_address += ":"
        gen_address = gen_address[:-1]
        print(gen_address)


if __name__ == '__main__':

    run_model()
