# -*- coding: utf-8 -*-
import os.path

import numpy
import tensorflow as tf
tf.random.set_seed(87)
numpy.random.seed(2)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, LSTM, Dense, ReLU, Subtract, Multiply, Add
import tensorflow.keras.backend as K
import pandas as pd
from config import Config
import numpy as np

import keras.backend as K


# fix random seed for reproducibility
numpy.random.seed(7)

global sess
global graph

print('tf version:')
print(tf.__version__)
print('python version:')
print('numpy version:')
print(numpy.version.version)


class RNNClassifierChooser:

    def __init__(self):
        self.classifiers = []
        self.confidence = []
        self.evaluation = []
        # self.index = 0

    def create_model(self, training_length):

        # create the model
        # 3458 samples
        # 110 features
        X = Input(shape=(111, 1))
        Y = Input(shape=7)
        Y_true = Input(shape=7)
        first_lstm = LSTM(128, return_sequences=True)(X)
        batchnorm1 = BatchNormalization()(first_lstm)
        relu_activation = ReLU()(batchnorm1)
        second_lstm = LSTM(64)(relu_activation)
        batchnorm2 = BatchNormalization()(second_lstm)
        relu_dense = Dense(20, activation='sigmoid')(batchnorm2)
        predictions = Dense(7, activation='softmax')(relu_dense)
        output = predictions[-7:]

        def custom_loss(y_true, y_pred, train_tensor):
            # # Calculating result
            # res = tf.get_static_value(y_pred)
            # # Printing the result
            # print('res: ', res)

            l1 = K.sum(tf.abs(Subtract()([y_true, y_pred])), keepdims=False)
            # return l1
            print(l1)
            print(K.argmax(y_true, axis=-1))
            max_true = tf.one_hot(tf.math.argmax(y_true, axis=-1), int(y_true.shape[1]))
            max_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), int(y_pred.shape[1]))
            # train_tensor = tf.convert_to_tensor(self.eval_X_train['evaluation'])
            mult1 = Multiply()([train_tensor, max_true])
            mult2 = Multiply()([train_tensor, max_pred])
            l2 = K.sum(tf.abs(Subtract()([mult1, mult2])), keepdims=False)
            calc_loss = Add()([l1, l2])
            # return calc_loss
            return calc_loss

        model = Model(inputs=[X, Y, Y_true], outputs=output)
        model.add_loss(custom_loss(Y_true, output, Y))
        # model.compile(loss=self.custom_loss, optimizer='adam', metrics=['accuracy'])
        model.compile(loss=None, optimizer='adam', metrics=['accuracy'])

        model.summary()

        return model

    def start_classifier_chooser(self, save_dir):

        datasets = Config.DATASET_NAME_LIST_TEST
        for dataset in datasets:
            # self.index = 0
            print('dataset:')
            print(dataset)
            self.classifiers = []
            self.evaluation = []
            x_train, y_train, x_test, y_test = self.train_test_sets(dataset)

            self.x_train = x_train
            self.x_test = x_test

            # convert labels to one hot
            for i in range(len(x_train)):
                converter_train = np.zeros((y_train[i].size, len(Config.CLASSIFIERS)))
                converter_train[np.arange(y_train[i].size), y_train[i]] = 1
                y_train[i] = converter_train
            converter_test = np.zeros((y_test.size, len(Config.CLASSIFIERS)))
            converter_test[np.arange(y_test.size), y_test] = 1
            y_test = converter_test

            self.y_train = y_train
            self.y_test = y_test

            # x_train = x_train.values.reshape(x_train.shape[0], 1, x_train.shape[1])
            # x_test = x_test.values.reshape(x_test.shape[0], 1, x_test.shape[1])
            for i in range(len(x_train)):
                x_train[i] = x_train[i].values.reshape(1, x_train[i].shape[0], x_train[i].shape[1])
            x_test = x_test.values.reshape(1, x_test.shape[0], x_test.shape[1])

            self.model = self.create_model(training_length=len(x_train))

            self.run_meta_model(x_train, y_train, x_test, y_test)
            self.save_csv(save_dir=save_dir, dataset=dataset)

    def exp2_start_classifier_chooser(self):

        self.classifiers = []
        self.confidence = []
        self.evaluation = []
        x_train, y_train = self.train_set()

        self.x_train = x_train
        self.x_test = []

        # convert labels to one hot
        for i in range(len(x_train)):
            converter_train = np.zeros((y_train[i].size, len(Config.CLASSIFIERS)))
            converter_train[np.arange(y_train[i].size), y_train[i]] = 1
            y_train[i] = converter_train

        self.y_train = y_train
        self.y_test = []

        for i in range(len(x_train)):
            x_train[i] = x_train[i].values.reshape(1, x_train[i].shape[0], x_train[i].shape[1])

        self.model = self.create_model(training_length=len(x_train))

        self.exp2_train_meta_model(x_train, y_train)

    def train_test_sets(self, current_dataset):

        # # one file data for training
        # #load data for train
        # x_train, y_train = self.load_data(f"drive/MyDrive/Experiment/train_sets/EXP2_{current_dataset}_train_set.csv")
        # #load data for test
        # x_test, y_test = self.load_data(f"drive/MyDrive/Experiment/test_sets/EXP2_{current_dataset}_AUC_ROC.csv")

        # multiple files data for training on each separately
        x_train = []
        y_train = []
        # load data for test
        x_test, y_test = self.load_data(f"drive/MyDrive/Experiment/test_sets/EXP2_{current_dataset}_AUC_ROC.csv")
        for dataset in Config.DATASET_NAME_LIST_TEST:
            if dataset == current_dataset or dataset is current_dataset:
                continue
            # load data for train
            x_temp_train, y_temp_train = self.load_data(
                f"drive/MyDrive/Experiment/test_sets/EXP2_{dataset}_AUC_ROC.csv")
            x_train.append(x_temp_train)
            y_train.append(y_temp_train)

        return x_train, y_train, x_test, y_test

    def train_set(self):
        x_train = []
        y_train = []

        for dataset in Config.DATASET_NAME_LIST_TEST:
            # load data for train
            x_temp_train, y_temp_train = self.load_data(
                fr"C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\test_sets\EXP2_{dataset}_AUC_ROC.csv")
            x_train.append(x_temp_train)
            y_train.append(y_temp_train)

        return x_train, y_train

    def load_data(self, path):
        dataset = pd.read_csv(path)
        numeric_cols = dataset._get_numeric_data().columns
        categorical_cols = list(set(dataset.columns) - set(numeric_cols))
        for cat_col in categorical_cols:
            dataset[cat_col] = dataset[cat_col].astype('category')
        dataset[categorical_cols] = dataset[categorical_cols].apply(lambda x: x.cat.codes)

        target_col_name = 'best_classifier'
        data = dataset.drop([target_col_name], axis=1)
        label = dataset[target_col_name]

        return data, label

    def exp2_split_data(self, dataset):
        numeric_cols = dataset._get_numeric_data().columns
        categorical_cols = list(set(dataset.columns) - set(numeric_cols))
        for cat_col in categorical_cols:
            dataset[cat_col] = dataset[cat_col].astype('category')
        dataset[categorical_cols] = dataset[categorical_cols].apply(lambda x: x.cat.codes)

        target_col_name = 'best_classifier'
        data = dataset.drop([target_col_name], axis=1)
        label = dataset[target_col_name]

        return data, label

    def run_meta_model(self, X_train, y_train, X_test, y_test, num_of_classifiers=7):

        # training on multiple files separately
        self.eval_X_test = np.split(self.x_test['evaluation'],
                                    range(num_of_classifiers, len(self.x_test['evaluation']), num_of_classifiers))
        for X_train_index in range(len(X_train)):
            self.x_train[X_train_index] = pd.DataFrame(np.squeeze(self.x_train[X_train_index]),
                                                       columns=['iteration', 'exp_id',
                                                                'label_rate', 'num_categorical_cols',
                                                                'num_numeric_cols', 'num_instances',
                                                                'num_labeled_instances',
                                                                'num_unlabeled_instances', 'ada_boost',
                                                                'desicition_tree', 'gaussian_nb', 'gaussian_process',
                                                                'logistic_regression', 'mlp',
                                                                'random_forest', 'initial_auc',
                                                                'cur_next_clfs_ttest_stat', 'cur_next_clfs_ttest_pval',
                                                                'clf_backwards_1_steps_ttest_stat',
                                                                'clf_backwards_1_steps_ttest_pval',
                                                                'clf_backwards_2_steps_ttest_stat',
                                                                'clf_backwards_2_steps_ttest_pval',
                                                                'clf_backwards_3_steps_ttest_stat',
                                                                'clf_backwards_3_steps_ttest_pval',
                                                                'confidence_clf_avg', 'confidence_clf_min',
                                                                'confidence_clf_max', 'confidence_clf_median',
                                                                'confidence_clf_std', 'confidence_clf_stdev',
                                                                'confidence_clf_norm_dist_pval',
                                                                'confidence_clf_skew', 'confidence_clf_shapiro_pval',
                                                                '1_iters_back_1_place_ada_boost',
                                                                '1_iters_back_1_place_desicition_tree',
                                                                '1_iters_back_1_place_gaussian_nb',
                                                                '1_iters_back_1_place_gaussian_process',
                                                                '1_iters_back_1_place_logistic_regression',
                                                                '1_iters_back_1_place_mlp',
                                                                '1_iters_back_1_place_random_forest',
                                                                '1_iters_back_2_place_ada_boost',
                                                                '1_iters_back_2_place_desicition_tree',
                                                                '1_iters_back_2_place_gaussian_nb',
                                                                '1_iters_back_2_place_gaussian_process',
                                                                '1_iters_back_2_place_logistic_regression',
                                                                '1_iters_back_2_place_mlp',
                                                                '1_iters_back_2_place_random_forest',
                                                                '1_iters_back_3_place_ada_boost',
                                                                '1_iters_back_3_place_desicition_tree',
                                                                '1_iters_back_3_place_gaussian_nb',
                                                                '1_iters_back_3_place_gaussian_process',
                                                                '1_iters_back_3_place_logistic_regression',
                                                                '1_iters_back_3_place_mlp',
                                                                '1_iters_back_3_place_random_forest',
                                                                '2_iters_back_1_place_ada_boost',
                                                                '2_iters_back_1_place_desicition_tree',
                                                                '2_iters_back_1_place_gaussian_nb',
                                                                '2_iters_back_1_place_gaussian_process',
                                                                '2_iters_back_1_place_logistic_regression',
                                                                '2_iters_back_1_place_mlp',
                                                                '2_iters_back_1_place_random_forest',
                                                                '2_iters_back_2_place_ada_boost',
                                                                '2_iters_back_2_place_desicition_tree',
                                                                '2_iters_back_2_place_gaussian_nb',
                                                                '2_iters_back_2_place_gaussian_process',
                                                                '2_iters_back_2_place_logistic_regression',
                                                                '2_iters_back_2_place_mlp',
                                                                '2_iters_back_2_place_random_forest',
                                                                '2_iters_back_3_place_ada_boost',
                                                                '2_iters_back_3_place_desicition_tree',
                                                                '2_iters_back_3_place_gaussian_nb',
                                                                '2_iters_back_3_place_gaussian_process',
                                                                '2_iters_back_3_place_logistic_regression',
                                                                '2_iters_back_3_place_mlp',
                                                                '2_iters_back_3_place_random_forest',
                                                                '3_iters_back_1_place_ada_boost',
                                                                '3_iters_back_1_place_desicition_tree',
                                                                '3_iters_back_1_place_gaussian_nb',
                                                                '3_iters_back_1_place_gaussian_process',
                                                                '3_iters_back_1_place_logistic_regression',
                                                                '3_iters_back_1_place_mlp',
                                                                '3_iters_back_1_place_random_forest',
                                                                '3_iters_back_2_place_ada_boost',
                                                                '3_iters_back_2_place_desicition_tree',
                                                                '3_iters_back_2_place_gaussian_nb',
                                                                '3_iters_back_2_place_gaussian_process',
                                                                '3_iters_back_2_place_logistic_regression',
                                                                '3_iters_back_2_place_mlp',
                                                                '3_iters_back_2_place_random_forest',
                                                                '3_iters_back_3_place_ada_boost',
                                                                '3_iters_back_3_place_desicition_tree',
                                                                '3_iters_back_3_place_gaussian_nb',
                                                                '3_iters_back_3_place_gaussian_process',
                                                                '3_iters_back_3_place_logistic_regression',
                                                                '3_iters_back_3_place_mlp',
                                                                '3_iters_back_3_place_random_forest', 'rank',
                                                                'AUC_gap_from_top', 'AUC_gap_from_avg',
                                                                'AUC_gap_from_min',
                                                                'AUC_gap_from_top_1_steps_back',
                                                                'AUC_gap_from_avg_1_steps_back',
                                                                'AUC_gap_from_min_1_steps_back',
                                                                'AUC_gap_from_top_2_steps_back',
                                                                'AUC_gap_from_avg_2_steps_back',
                                                                'AUC_gap_from_min_2_steps_back',
                                                                'AUC_gap_from_top_3_steps_back',
                                                                'AUC_gap_from_avg_3_steps_back',
                                                                'AUC_gap_from_min_3_steps_back', 'evaluation'])
            print('self.x_train[X_train_index]')
            print(self.x_train[X_train_index])
            # print('type(self.x_train[X_train_index])')
            # print(type(self.x_train[X_train_index]))
            self.eval_X_train = np.split(self.x_train[X_train_index]['evaluation'],
                                         range(num_of_classifiers, len(self.x_train[X_train_index]['evaluation']),
                                               num_of_classifiers))
            index = 0
            for i in range(0, len(X_train[X_train_index]), num_of_classifiers):

                try:
                    x = self.eval_X_train[index + 1]
                except:
                    break

                print(index)
                self.model.train_on_batch(x=[X_train[X_train_index][:(index + 1) * num_of_classifiers],
                                             self.eval_X_train[index + 1],
                                             y_train[X_train_index][
                                             (index + 1) * num_of_classifiers: (index + 2) * num_of_classifiers]],
                                          y=y_train[X_train_index][
                                            (index + 1) * num_of_classifiers: (index + 2) * num_of_classifiers],
                                          reset_metrics=False)
                index += 1

        # # training on one file
        # self.eval_X_train = np.split(self.x_train['evaluation'], range(num_of_classifiers, len(self.x_train['evaluation']), num_of_classifiers))
        # self.eval_X_test = np.split(self.x_test['evaluation'], range(num_of_classifiers, len(self.x_test['evaluation']), num_of_classifiers))
        # index = 0
        # for i in range(0, len(X_train[0]), num_of_classifiers):
        #     try:
        #         x = self.eval_X_train[index + 1]
        #     except:
        #         break
        #     # for current pred
        #     # self.model.train_on_batch(x=[X_train[0][i:i+num_of_classifiers],
        #     #                   self.eval_X_train[index],
        #     #                   y_train[i:i+num_of_classifiers]],
        #     #                y=y_train[i:i+num_of_classifiers], reset_metrics=False)
        #     print(index)
        #     # for next step pred
        #     self.model.train_on_batch(x=[X_train[0][:(index+1)*num_of_classifiers],
        #                       self.eval_X_train[index+1],
        #                       y_train[(index+1)*num_of_classifiers: (index+2)*num_of_classifiers]],
        #                    y=y_train[(index+1)*num_of_classifiers: (index+2)*num_of_classifiers], reset_metrics=False)
        #
        #     # for next next step pred
        #     # self.model.train_on_batch(x=[X_train[0][:(index+1)*num_of_classifiers],
        #     #                   self.eval_X_train[index+2],
        #     #                   y_train[(index+2)*num_of_classifiers: (index+3)*num_of_classifiers]],
        #     #                y=y_train[(index+2)*num_of_classifiers: (index+3)*num_of_classifiers], reset_metrics=False)
        #
        #     index += 1

        # test by steps
        start_test_index = 0
        test_index = 1
        while start_test_index < len(X_test[0]):

            try:
                x = self.eval_X_test[test_index]
            except:
                break

            # input: one window, current step prediction
            # TODO: delete try except from here
            # X_test_step = X_test[0][start_test_index: start_test_index+len(Config.CLASSIFIERS)]
            # y_test_step = y_test[start_test_index : start_test_index + len(Config.CLASSIFIERS)]
            # print('start_test_index:')
            # print(start_test_index)
            # print('index:')
            # print(test_index)
            # pred = self.model.predict((X_test_step,
            #                           self.eval_X_test[test_index-1],
            #                           y_test_step))

            # input: one window, next step prediction
            # X_test_step = X_test[0][start_test_index: start_test_index+len(Config.CLASSIFIERS)]
            # y_test_step = y_test[start_test_index + len(Config.CLASSIFIERS): start_test_index + 2 * len(Config.CLASSIFIERS)]
            # print('start_test_index:')
            # print(start_test_index)
            # print('index:')
            # print(test_index)
            # pred = self.model.predict((X_test_step,
            #                           self.eval_X_test[test_index],
            #                           y_test_step))

            # input: all windows up to i, next step prediction
            X_test_step = X_test[0][: start_test_index + len(Config.CLASSIFIERS)]
            y_test_step = y_test[
                          start_test_index + len(Config.CLASSIFIERS): start_test_index + 2 * len(Config.CLASSIFIERS)]
            print('start_test_index:')
            print(start_test_index)
            print('index:')
            print(test_index)
            pred = self.model.predict_on_batch(x=[X_test_step,
                                                  self.eval_X_test[test_index],
                                                  y_test_step])

            test_index += 1

            # print("predict:" + str(pred))
            # print("y_test: "+ str(y_test_step))
            print("prediction step")

            print('index of max prediction prob')
            print([list(i).index(i.max()) for i in pred])
            most_common_classifier = self.most_common([Config.CLASSIFIERS[list(i).index(i.max())] for i in pred])
            print('most_common_classifier')
            print(most_common_classifier)
            for index in range(len(Config.CLASSIFIERS)):
                self.classifiers.append(most_common_classifier)

            start_test_index += len(Config.CLASSIFIERS)
        return pred

    def exp2_train_meta_model(self, X_train, y_train, num_of_classifiers=7):

        # training on multiple files separately

        for X_train_index in range(len(X_train)):
            self.x_train[X_train_index] = pd.DataFrame(np.squeeze(self.x_train[X_train_index]),
                                                       columns=['iteration', 'exp_id',
                                                                'label_rate', 'num_categorical_cols',
                                                                'num_numeric_cols', 'num_instances',
                                                                'num_labeled_instances',
                                                                'num_unlabeled_instances', 'ada_boost',
                                                                'desicition_tree', 'gaussian_nb', 'gaussian_process',
                                                                'logistic_regression', 'mlp',
                                                                'random_forest', 'initial_auc',
                                                                'cur_next_clfs_ttest_stat', 'cur_next_clfs_ttest_pval',
                                                                'clf_backwards_1_steps_ttest_stat',
                                                                'clf_backwards_1_steps_ttest_pval',
                                                                'clf_backwards_2_steps_ttest_stat',
                                                                'clf_backwards_2_steps_ttest_pval',
                                                                'clf_backwards_3_steps_ttest_stat',
                                                                'clf_backwards_3_steps_ttest_pval',
                                                                'confidence_clf_avg', 'confidence_clf_min',
                                                                'confidence_clf_max', 'confidence_clf_median',
                                                                'confidence_clf_std', 'confidence_clf_stdev',
                                                                'confidence_clf_norm_dist_pval',
                                                                'confidence_clf_skew', 'confidence_clf_shapiro_pval',
                                                                '1_iters_back_1_place_ada_boost',
                                                                '1_iters_back_1_place_desicition_tree',
                                                                '1_iters_back_1_place_gaussian_nb',
                                                                '1_iters_back_1_place_gaussian_process',
                                                                '1_iters_back_1_place_logistic_regression',
                                                                '1_iters_back_1_place_mlp',
                                                                '1_iters_back_1_place_random_forest',
                                                                '1_iters_back_2_place_ada_boost',
                                                                '1_iters_back_2_place_desicition_tree',
                                                                '1_iters_back_2_place_gaussian_nb',
                                                                '1_iters_back_2_place_gaussian_process',
                                                                '1_iters_back_2_place_logistic_regression',
                                                                '1_iters_back_2_place_mlp',
                                                                '1_iters_back_2_place_random_forest',
                                                                '1_iters_back_3_place_ada_boost',
                                                                '1_iters_back_3_place_desicition_tree',
                                                                '1_iters_back_3_place_gaussian_nb',
                                                                '1_iters_back_3_place_gaussian_process',
                                                                '1_iters_back_3_place_logistic_regression',
                                                                '1_iters_back_3_place_mlp',
                                                                '1_iters_back_3_place_random_forest',
                                                                '2_iters_back_1_place_ada_boost',
                                                                '2_iters_back_1_place_desicition_tree',
                                                                '2_iters_back_1_place_gaussian_nb',
                                                                '2_iters_back_1_place_gaussian_process',
                                                                '2_iters_back_1_place_logistic_regression',
                                                                '2_iters_back_1_place_mlp',
                                                                '2_iters_back_1_place_random_forest',
                                                                '2_iters_back_2_place_ada_boost',
                                                                '2_iters_back_2_place_desicition_tree',
                                                                '2_iters_back_2_place_gaussian_nb',
                                                                '2_iters_back_2_place_gaussian_process',
                                                                '2_iters_back_2_place_logistic_regression',
                                                                '2_iters_back_2_place_mlp',
                                                                '2_iters_back_2_place_random_forest',
                                                                '2_iters_back_3_place_ada_boost',
                                                                '2_iters_back_3_place_desicition_tree',
                                                                '2_iters_back_3_place_gaussian_nb',
                                                                '2_iters_back_3_place_gaussian_process',
                                                                '2_iters_back_3_place_logistic_regression',
                                                                '2_iters_back_3_place_mlp',
                                                                '2_iters_back_3_place_random_forest',
                                                                '3_iters_back_1_place_ada_boost',
                                                                '3_iters_back_1_place_desicition_tree',
                                                                '3_iters_back_1_place_gaussian_nb',
                                                                '3_iters_back_1_place_gaussian_process',
                                                                '3_iters_back_1_place_logistic_regression',
                                                                '3_iters_back_1_place_mlp',
                                                                '3_iters_back_1_place_random_forest',
                                                                '3_iters_back_2_place_ada_boost',
                                                                '3_iters_back_2_place_desicition_tree',
                                                                '3_iters_back_2_place_gaussian_nb',
                                                                '3_iters_back_2_place_gaussian_process',
                                                                '3_iters_back_2_place_logistic_regression',
                                                                '3_iters_back_2_place_mlp',
                                                                '3_iters_back_2_place_random_forest',
                                                                '3_iters_back_3_place_ada_boost',
                                                                '3_iters_back_3_place_desicition_tree',
                                                                '3_iters_back_3_place_gaussian_nb',
                                                                '3_iters_back_3_place_gaussian_process',
                                                                '3_iters_back_3_place_logistic_regression',
                                                                '3_iters_back_3_place_mlp',
                                                                '3_iters_back_3_place_random_forest', 'rank',
                                                                'AUC_gap_from_top', 'AUC_gap_from_avg',
                                                                'AUC_gap_from_min',
                                                                'AUC_gap_from_top_1_steps_back',
                                                                'AUC_gap_from_avg_1_steps_back',
                                                                'AUC_gap_from_min_1_steps_back',
                                                                'AUC_gap_from_top_2_steps_back',
                                                                'AUC_gap_from_avg_2_steps_back',
                                                                'AUC_gap_from_min_2_steps_back',
                                                                'AUC_gap_from_top_3_steps_back',
                                                                'AUC_gap_from_avg_3_steps_back',
                                                                'AUC_gap_from_min_3_steps_back', 'evaluation'])
            # print('self.x_train[X_train_index]')
            # print(self.x_train[X_train_index])
            # print('type(self.x_train[X_train_index])')
            # print(type(self.x_train[X_train_index]))
            self.eval_X_train = np.split(self.x_train[X_train_index]['evaluation'],
                                         range(num_of_classifiers, len(self.x_train[X_train_index]['evaluation']),
                                               num_of_classifiers))
            index = 0
            for i in range(0, len(X_train[X_train_index]), num_of_classifiers):

                try:
                    x = self.eval_X_train[index + 1]
                except:
                    break

                print(index)
                self.model.train_on_batch(x=[X_train[X_train_index][:(index + 1) * num_of_classifiers],
                                             self.eval_X_train[index + 1],
                                             y_train[X_train_index][
                                             (index + 1) * num_of_classifiers: (index + 2) * num_of_classifiers]],
                                          y=y_train[X_train_index][
                                            (index + 1) * num_of_classifiers: (index + 2) * num_of_classifiers],
                                          reset_metrics=False)
                index += 1

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")


    def exp2_test_meta_model(self, X_test, y_test, num_of_classifiers):
        self.classifiers = []
        self.confidence = []
        self.eval_X_test = np.split(self.x_test['evaluation'],
                                    range(num_of_classifiers, len(self.x_test['evaluation']), num_of_classifiers))
        # test by steps
        start_test_index = 0
        test_index = 1
        # print('X_test[0]')
        # print(X_test[0])
        print('len(X_test[0])')
        print(len(X_test[0]))
        # print('self.eval_X_test[test_index]')
        # print(self.eval_X_test[test_index])
        while start_test_index < len(X_test[0]):

            try:
                x = self.eval_X_test[test_index]
            except:
                break

            # input: all windows up to i, next step prediction
            X_test_step = X_test[0][: start_test_index + len(Config.CLASSIFIERS)]
            y_test_step = y_test[
                          start_test_index + len(Config.CLASSIFIERS): start_test_index + 2 * len(Config.CLASSIFIERS)]
            print('start_test_index:')
            print(start_test_index)
            print('index:')
            print(test_index)
            pred = self.model.predict_on_batch(x=[X_test_step,
                                                  self.eval_X_test[test_index],
                                                  y_test_step])

            test_index += 1

            # print("predict:" + str(pred))
            # print("y_test: "+ str(y_test_step))
            print("prediction step")

            print('index of max prediction prob')
            print([list(i).index(i.max()) for i in pred])
            most_common_classifier = self.most_common([Config.CLASSIFIERS[list(i).index(i.max())] for i in pred])
            confidence = np.max(pred)
            print('most_common_classifier')
            print(most_common_classifier)
            for index in range(len(Config.CLASSIFIERS)):
                self.classifiers.append(most_common_classifier)
                self.confidence.append(confidence)

            start_test_index += len(Config.CLASSIFIERS)
        return pred

    def save_csv(self, save_dir, dataset, index=False):
        meta_dataset = {}
        meta_dataset['chosen_classifier'] = self.classifiers
        meta_dataset['confidence'] = self.confidence
        df = pd.DataFrame.from_dict(meta_dataset)
        df.to_csv(os.path.join(save_dir, f'EXP2_{dataset}_RNN_predictions.csv'), index=index)

    def most_common(self, lst):
        return max(set(lst), key=lst.count)

