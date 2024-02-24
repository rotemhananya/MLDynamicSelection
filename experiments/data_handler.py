import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys

RANDOM_STATE = 42


class DataHandler:
    def __init__(self, target_col_name=None, dataset_name='german_credit'):
        self.dataset_name = dataset_name
        self.split_percentage = 0.1

        filename = '../datasets/{}.csv'.format(dataset_name)
        self.dataset = pd.read_csv(filename)
        self.dataset = pd.read_csv(filename)

        self.original_dataset = self.dataset  # keep the original dataset
        self.handle_categorical_data()

        if target_col_name is not None:
            self.data = self.dataset.drop([target_col_name], axis=1)
            self.label = self.dataset[target_col_name]
        else:
            self.data = self.dataset[self.dataset.columns[:-1]]
            self.label = self.dataset[self.dataset.columns[-1]]
        self.label_encode()
        self.class_ratio = self.classes_ratio_calc()

    def separate_data(self, label_percentage=0.75, batch_length=None):

        if batch_length is None:
            batch_length = int(0.1 * len(self.data))
        # check correctness of dataset:
        # print(self.dataset[self.dataset.columns[-1]][0])
        # print(self.dataset[self.dataset.columns[-1]][7])
        class1_rows = self.dataset.loc[
            self.dataset[self.dataset.columns[-1]] == self.dataset[self.dataset.columns[-1]].iloc[0]]
        class2_rows = self.dataset.loc[
            self.dataset[self.dataset.columns[-1]] != self.dataset[self.dataset.columns[-1]].iloc[0]]
        # print(len(class1_rows[class1_rows.columns[-1]]))
        # print(len(class2_rows[class2_rows.columns[-1]]))
        separated_data = pd.DataFrame(dict())

        flag = "begin"
        while True:
            # if there is not enough rows for current batch: Done
            if class1_rows.shape[0] < label_percentage * batch_length or class2_rows.shape[0] < (
                    1 - label_percentage) * batch_length:
                break

            batch = class1_rows[: int(label_percentage * batch_length) - 1]
            batch = batch.append(class2_rows[: int((1 - label_percentage) * batch_length) - 1])
            # shuffle the rows
            batch = batch.sample(frac=1).reset_index(drop=True)
            if (flag == "begin"):
                separated_data = batch
                flag = "continue"
            else:
                separated_data = separated_data.append(batch)
            class1_rows = class1_rows[int(label_percentage * batch_length):]
            class2_rows = class2_rows[int((1 - label_percentage) * batch_length):]

        # print(type(class2_rows))
        # print(self.data.shape[0])
        self.data = separated_data[separated_data.columns[:-1]]
        self.label = separated_data[separated_data.columns[-1]]
        # print(self.data.shape[0])

    def data_split(self, label_rate, test_rate=0.25, iteration=1, steps=1):
        self.test_rate = test_rate
        self.label_rate = label_rate
        # split data hardcoded
        delta_train_data = int(len(self.data) * steps * self.split_percentage)
        end_train_data_index = int(len(self.data) * iteration * self.split_percentage)
        end_test_data_index = int(len(self.data) * (iteration + 1) * self.split_percentage)
        start_train_data_index = end_train_data_index - delta_train_data
        # start_train_data_index = max(start_train_data_index, 0)
        print("iteration: " + str(iteration))
        print("steps: " + str(steps))
        print("X_train: " + str(start_train_data_index) + " to " + str(end_train_data_index))
        print("X_test: " + str(end_train_data_index + 1) + " to " + str(end_test_data_index))
        self.X_train = self.data[start_train_data_index: end_train_data_index + 1]
        self.y_train = self.label[start_train_data_index: end_train_data_index + 1]
        self.X_test = self.data[end_train_data_index + 1: end_test_data_index + 1]
        self.y_test = self.label[end_train_data_index + 1: end_test_data_index + 1]
        return self.X_train, self.y_train, self.X_test, self.y_test

    def data_split_by_train_window(self, label_rate, test_rate=0.25, train_window_size=4):
        self.test_rate = test_rate
        self.label_rate = label_rate
        # split data hardcoded
        end_train_data_index = int(len(self.data) * train_window_size * self.split_percentage)
        end_test_data_index = int(len(self.data) * (train_window_size+1) * self.split_percentage)
        start_train_data_index = 0
        # start_train_data_index = max(start_train_data_index, 0)

        print("train window size: " + str(train_window_size))
        print("X_train: " + str(start_train_data_index) + " to " + str(end_train_data_index-1))
        print("X_test: " + str(end_train_data_index) + " to " + str(end_test_data_index-1))
        self.X_train = self.data[start_train_data_index: end_train_data_index]
        self.y_train = self.label[start_train_data_index: end_train_data_index]
        self.X_test = self.data[end_train_data_index: end_test_data_index]
        self.y_test = self.label[end_train_data_index: end_test_data_index]
        return self.X_train, self.y_train, self.X_test, self.y_test


    def data_split_by_train_test_windows(self, label_rate, test_rate=0.25, train_window_size=4):
        self.test_rate = test_rate
        self.label_rate = label_rate
        # split data hardcoded
        end_train_data_index = int(len(self.data) * train_window_size * self.split_percentage)
        end_test_data_index = int(len(self.data))
        start_train_data_index = 0
        # start_train_data_index = max(start_train_data_index, 0)

        print("train window size: " + str(train_window_size))
        print("X_train: " + str(start_train_data_index) + " to " + str(end_train_data_index-1))
        print("X_test: " + str(end_train_data_index) + " to " + str(end_test_data_index-1))
        self.X_train = self.data[start_train_data_index: end_train_data_index]
        self.y_train = self.label[start_train_data_index: end_train_data_index]
        self.X_test = self.data[end_train_data_index: end_test_data_index]
        self.y_test = self.label[end_train_data_index: end_test_data_index]
        return self.X_train, self.y_train, self.X_test, self.y_test

    def handle_categorical_data(self):
        self.numeric_cols = self.dataset._get_numeric_data().columns
        self.categorical_cols = list(set(self.dataset.columns) - set(self.numeric_cols))
        for cat_col in self.categorical_cols:
            self.dataset[cat_col] = self.dataset[cat_col].astype('category')
        self.dataset[self.categorical_cols] = self.dataset[self.categorical_cols].apply(lambda x: x.cat.codes)

    def label_encode(self):
        # for binary classes target
        unique_vals = self.label.unique()
        self.len_unique_vals = len(unique_vals)
        self.label = self.label.map({unique_vals[0]: 0, unique_vals[1]: 1, unique_vals[2]: 2}) if len(unique_vals) == 3 \
            else self.label.map({unique_vals[0]: 0, unique_vals[1]: 1})
        # print(self.label)

    def classes_ratio_calc(self):
        return sum(self.label) / len(self.label)
