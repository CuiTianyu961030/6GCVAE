import numpy as np
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split

dataset_path = 'data/processed_data/data.txt'
generated_path = 'data/generated_data/6gcvae_generation.txt'
# dataset_path = 'data/processed_data/slaac_privacy_addresses_gasser_data.txt'
# generated_path = 'data/generated_data/6vae_generation_slaac_privacy_addresses.txt'

n = 32
latent_dim = 64
hidden_dim = 64


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


def run_model():

    x_train, x_test, y_train, y_test = load_data(dataset_path)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    # x_train = x_train.astype('float32') / 15.
    # x_test = x_test.astype('float32') / 15.

    class GCNN(Layer):
        def __init__(self, output_dim=None, residual=False, **kwargs):
            super(GCNN, self).__init__(**kwargs)
            self.output_dim = output_dim
            self.residual = residual

        def build(self, input_shape):
            if self.output_dim == None:
                self.output_dim = input_shape[-1]
            self.kernel = self.add_weight(name='gcnn_kernel',
                                          shape=(3, input_shape[-1],
                                                 self.output_dim * 2),
                                          initializer='glorot_uniform',
                                          trainable=True)
            print(self.kernel)

        def call(self, x):
            _ = K.conv1d(x, self.kernel, padding='same')
            print("input", x)
            print("conv", _)
            print("output_dim", self.output_dim)
            _ = _[:, :, :self.output_dim] * K.sigmoid(_[:, :, self.output_dim:])
            print("output", _)
            if self.residual:
                return _ + x
            else:
                return _

    input_sentence = Input(shape=(n,), dtype='int32')
    input_vec = Embedding(16, hidden_dim)(input_sentence)
    h = GCNN(residual=True)(input_vec)
    h = GCNN(residual=True)(h)
    h = GlobalAveragePooling1D()(h)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    decoder_hidden = Dense(hidden_dim * n)
    decoder_cnn = GCNN(residual=True)
    decoder_dense = Dense(16, activation='softmax')

    h = decoder_hidden(z)
    h = Reshape((n, hidden_dim))(h)
    h = decoder_cnn(h)
    output = decoder_dense(h)

    vae = Model(input_sentence, output)

    xent_loss = K.sum(K.sparse_categorical_crossentropy(input_sentence, output), 1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    decoder_input = Input(shape=(latent_dim,))
    _ = decoder_hidden(decoder_input)
    _ = Reshape((n, hidden_dim))(_)
    _ = decoder_cnn(_)
    _output = decoder_dense(_)
    generator = Model(decoder_input, _output)

    def gen():
        r = generator.predict(np.random.randn(1, latent_dim))[0]
        r = r.argmax(axis=1)
        print(r)
        return r

    class Evaluate(Callback):
        def __init__(self):
            self.log = []

        def on_epoch_end(self, epoch, logs=None):
            self.log.append(gen())
            gen_address = ""
            count = 0
            gen_address_list = [str(hex(i))[-1] for i in self.log[-1]]
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
            epochs=3,
            batch_size=64,
            callbacks=[evaluator]
            )

    vae.save_weights('models/gcnn_vae.model')

    for i in range(20):
        r = gen()
        gen_address = ""
        count = 0
        gen_address_list = [str(hex(i))[-1] for i in r]
        for i in gen_address_list:
            count += 1
            gen_address += i
            if count % 4 == 0:
                gen_address += ":"
        gen_address = gen_address[:-1]
        print(gen_address)


if __name__ == "__main__":

    run_model()
