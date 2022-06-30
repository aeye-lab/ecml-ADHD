import os
import keras as ks
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D
from keras.callbacks import EarlyStopping
from keras.metrics import AUC
from keras.initializers import he_normal


class CNN_reg():

    def __init__(self, seq_len, channels):
        self.model = CNN_reg.build_CNN_reg(seq_len,channels)
        self.batch_size = 4

    def train(self, X_train, Y_train, X_val, Y_val):
        # callback for early stopping
        callbacks = [EarlyStopping(monitor='val_loss', patience=30, mode = 'min')]
        print('Pre-train CNN...')

        net = self.model.fit(X_train, Y_train,
                            validation_data=(X_val, Y_val),
                            shuffle=True,
                            batch_size=self.batch_size,
                            epochs=1000,
                            callbacks=callbacks)

        return self.model

    @staticmethod
    def build_CNN_reg(seq_len, channels):

        inputs = Input(shape=[seq_len,channels])
        conv1 = Conv1D(filters=16, kernel_size=9, strides=2, padding = 'same', kernel_initializer='he_normal')(inputs)
        a1 = Activation('relu')(conv1)
        b1 = BatchNormalization(axis=-1)(a1)
        p1 = AveragePooling1D(pool_size=2, strides=2, padding='same')(b1)

        conv2 = Conv1D(filters=32, kernel_size=9, strides=1, padding = 'same', kernel_initializer='he_normal')(p1)
        a2 = Activation('relu')(conv2)
        b2 = BatchNormalization(axis=-1)(a2)
        p2 = AveragePooling1D(pool_size=2, strides=2, padding='same')(b2)

        conv3 = Conv1D(filters=32, kernel_size=9, strides=2, padding = 'same', kernel_initializer='he_normal')(p2)
        a3 = Activation('relu')(conv3)
        b3 = BatchNormalization(axis=-1)(a3)
        p3 = AveragePooling1D(pool_size=2, strides=2, padding='same')(b3)

        conv4 = Conv1D(filters=32, kernel_size=3, strides=1, padding = 'same', kernel_initializer='he_normal')(p3)
        a4 = Activation('relu')(conv4)
        b4 = BatchNormalization(axis=-1)(a4)
        p4 = AveragePooling1D(pool_size=2, strides=2, padding='same')(b4)

        x = Dropout(rate=0.4)(p4)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation=None)(x)
        CNN = Model(inputs=inputs, outputs=output)

        lr = 10/1e4
        # compile
        opt_CNN = ks.optimizers.Adam(lr=lr)
        CNN.compile(optimizer=opt_CNN, loss='mean_squared_error', metrics=['mse'])
        print(CNN.summary())

        return CNN






class CNN_classif():

    def __init__(self, seq_len, channels):
        self.model = CNN_classif.build_CNN_classif(seq_len,channels)
        self.batch_size = 4

    def train(self, X_train, Y_train, X_val, Y_val, pretrain_model, save_weights_flag, fold_counter):
        # callback for early stopping
        callbacks = [EarlyStopping(monitor='val_loss', patience=20, mode = 'min')]
        print('Train CNN...')
        if not pretrain_model:
            print('Train model from scratch...')
            net = self.model.fit(X_train, Y_train,
                                validation_data=(X_val, Y_val),
                                shuffle=True,
                                batch_size=self.batch_size,
                                epochs=1000,
                                callbacks=callbacks)

        else:
            print('Loading pretrained weights for model.')
            self.model.set_weights(pretrain_model.get_weights())

            net = self.model.fit(X_train, Y_train,
                                validation_data=(X_val, Y_val),
                                shuffle=True,
                                batch_size=self.batch_size,
                                epochs=1000,
                                callbacks=callbacks)

        #save model weights
        if save_weights_flag:
            isExist = os.path.exists(f'./results/saved_models_weights/fold_{fold_counter}/')
            if not isExist:
                os.makedirs(path)
            self.model.save(path)

        return net


    @staticmethod
    def build_CNN_classif(seq_len, channels):

        inputs = Input(shape=[seq_len,channels])
        conv1 = Conv1D(filters=16, kernel_size=9, strides=2, padding = 'same', kernel_initializer='he_normal')(inputs)
        a1 = Activation('relu')(conv1)
        b1 = BatchNormalization(axis=-1)(a1)
        p1 = AveragePooling1D(pool_size=2, strides=2, padding='same')(b1)

        conv2 = Conv1D(filters=32, kernel_size=9, strides=1, padding = 'same', kernel_initializer='he_normal')(p1)
        a2 = Activation('relu')(conv2)
        b2 = BatchNormalization(axis=-1)(a2)
        p2 = AveragePooling1D(pool_size=2, strides=2, padding='same')(b2)

        conv3 = Conv1D(filters=32, kernel_size=9, strides=2, padding = 'same', kernel_initializer='he_normal')(p2)
        a3 = Activation('relu')(conv3)
        b3 = BatchNormalization(axis=-1)(a3)
        p3 = AveragePooling1D(pool_size=2, strides=2, padding='same')(b3)

        conv4 = Conv1D(filters=32, kernel_size=3, strides=1, padding = 'same', kernel_initializer='he_normal')(p3)
        a4 = Activation('relu')(conv4)
        b4 = BatchNormalization(axis=-1)(a4)
        p4 = AveragePooling1D(pool_size=2, strides=2, padding='same')(b4)

        x = Dropout(rate=0.4)(p4)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        CNN = Model(inputs=inputs, outputs=output)


        lr = 1/1e4
        # compile
        opt_CNN = ks.optimizers.Adam(lr=lr)

        CNN.compile(optimizer=opt_CNN, loss='binary_crossentropy', metrics=['accuracy', AUC()])
        #print(CNN.summary())

        return CNN
