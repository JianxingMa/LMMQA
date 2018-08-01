from config import Config
from keras import regularizers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import LSTM, merge, Dot
from keras.models import Model
import keras
from keras.regularizers import l1_l2
import numpy as np
from keras.models import load_model
from keras import backend as K
import pickle
import os


def abs_diff(X):
    s = X[0]
    s2 = X[1]
    diff = s - s2
    diff = K.abs(diff)
    return diff


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class Trainer:
    def __init__(self):
        self.config = Config()

    def build_model(self):
        input1 = Input(shape=(self.config.max_sent_len, self.config.word_emb_dim))
        input2 = Input(shape=(self.config.max_sent_len, self.config.word_emb_dim))
        lstm1 = LSTM(300, dropout=0.3, activity_regularizer=l1_l2(0.01), return_sequences=True,
                     activation='relu')(input1)
        lstm2 = LSTM(300, dropout=0.3, activity_regularizer=l1_l2(0.01), return_sequences=True,
                     activation='relu')(input2)
        pool1 = GlobalMaxPooling1D()(lstm1)
        pool2 = GlobalMaxPooling1D()(lstm2)
        diffvec = merge([pool1, pool2], mode=abs_diff, output_shape=(300,))
        product = Dot(-1)([pool1, pool2])
        features = merge([pool1, pool2, diffvec, product], mode='concat', concat_axis=-1)
        out = Dense(2, kernel_regularizer=regularizers.l1_l2(0.01),
                    activity_regularizer=regularizers.l1_l2(0.01), activation='softmax')(features)
        model = Model(inputs=[input1, input2], outputs=[out])
        return model

    def train(self):
        check = keras.callbacks.ModelCheckpoint(self.config.model_save_path, monitor='val_acc',
                                                verbose=1,
                                                save_best_only=True, save_weights_only=False,
                                                mode='auto', period=1)
        self.model = self.build_model()
        self.model.summary()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.RMSprop(lr=1e-4),
                           metrics=['accuracy'])
        self.model.fit([self.q, self.a], self.labels,
                       batch_size=32,
                       epochs=100,
                       shuffle=True,
                       verbose=1,
                       validation_split=0.1, callbacks=[check])

    def load_data(self, q, a, lbl):
        sum(lbl), len(lbl)
        self.q = np.array(q)
        self.a = np.array(a)
        lbl = [[0, 1] if x == 1 else [1, 0] for x in lbl]
        self.labels = np.array(lbl)

    def abs_diff(self, X):
        s = X[0]
        for i in range(1, len(X)):
            s -= X[i]
        s = K.abs(s)
        return s

    def run(self):
        self.train()
        self.evaluate()

    def build_model2(self):
        input = Input(shape=(30, 300))
        conv_output = Conv1D(300, kernel_size=3, strides=1, activation="relu")(
            input)
        lstm_output = LSTM(300, dropout=0.3)(conv_output)
        out = Dense(len(self.class_dict), activity_regularizer=l1_l2(0.01), activation="softmax")(
            lstm_output)
        model = Model(inputs=[input], outputs=[out])
        return model

    def train2(self):
        check = keras.callbacks.ModelCheckpoint('model/answer2.model', monitor='val_acc', verbose=1,
                                                save_best_only=True, save_weights_only=False,
                                                mode='auto', period=1)
        self.model = self.build_model2()

        self.model.summary()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.RMSprop(lr=3e-4),
                           metrics=['accuracy'])
        self.model.fit([self.q], self.a,
                       batch_size=32,
                       epochs=100,
                       shuffle=True,
                       verbose=1,
                       validation_split=0.1, callbacks=[check])

    def run2(self):
        self.train2()

    def load_ner_data(self, q, a):
        self.q = []
        self.a = []
        for i in range(len(a)):
            tags = a[i]
            thisq = q[i]
            tags = [x[1] for x in tags]
            # print 'len1 ',len(tags)
            if all([x == 'O' for x in tags]):
                self.q.append(thisq)
                self.a.append('O')
                continue
            tags = [x for x in tags if x != 'O']
            if tags:
                self.q.append(thisq)
                self.a.append(tags[0])
        print
        len(q)
        all_a = list(set(self.a))
        class_dict = {}
        for i in range(len(all_a)):
            lbl = all_a[i]
            one_hot = [0] * len(all_a)
            one_hot[i] = 1
            class_dict[lbl] = one_hot
        self.class_dict = class_dict
        self.onehot_dict = {}
        for key in class_dict:
            oh = class_dict[key]
            # print oh
            new_key = np.argmax(oh)
            print
            new_key
            self.onehot_dict[new_key] = key

        self.a = [class_dict[x] for x in self.a]
        self.q = np.array(self.q)
        self.a = np.array(self.a)

        with open('onehot.dict', 'wb') as f:
            pickle.dump(self.onehot_dict, f)
        with open('class.dict', 'wb') as f:
            pickle.dump(self.class_dict, f)

    def predict_data(self, q, raw_q, raw_a=None, a=None):
        self.model = load_model('model/answer.model')
        with open('onehot.dict') as f:
            self.onehot_dict = pickle.load(f)
        pred_a = self.model.predict(np.array(q))
        class_a = [x.argsort()[-3:][::-1] for x in pred_a]
        answers = [[self.onehot_dict[x] for x in classes] for classes in class_a]
        type_a = []
        if a:
            for i in range(len(a)):
                tags = a[i]
                tags = [x[1] for x in tags]
                # print 'len1 ',len(tags)
                if all([x == 'O' for x in tags]):
                    type_a.append('O')
                    continue
                tags = [x for x in tags if x != 'O']
                if tags:
                    type_a.append(tags[0])
            print
            np.mean(np.array(answers) == np.array(type_a))
            for i in range(len(a)):
                tp = type_a[i]
                answer = answers[i]
                if tp != answer:
                    print
                    answer, a[i]
        if raw_a:
            for i in range(len(pred_a)):
                print
                pred_a[i], raw_a[i]
        for i in range(len(q)):
            print
            raw_q[i], answers[i]

    def load_dummy(self):
        self.q = np.array([[[0] * 300] * 30] * 100 + [[[1] * 300] * 30] * 100)
        self.a = self.q
        self.labels = [1] * 100 + [0] * 100
        self.labels = np.array([[0, 1] if x == 1 else [1, 0] for x in self.labels])

    def evaluate(self):
        self.p = np.array([[[0] * 300] * 30, [[1] * 300] * 30])
        self.pa = self.p
        rs = self.model.predict([self.p, self.pa])
        print
        rs


if __name__ == '__main__':
    tr = Trainer()
    tr.load_dummy()
    tr.run()
