# -*- coding: utf-8 -*-
from random import randrange
from config import Config
import pandas as pd
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


class ChooseClassifiers:
    def __init__(self, metadata_name):
        self.chosen_classifiers = []
        self.metadata_name = metadata_name
        self.evaluation = []
        self.classifiers = []
        
        
    def calculate_next_step(self, meta_features, iteration = 1):
        
        self.model = RandomForestClassifier()
        #print(meta_features[len(meta_features)-1]['exp_id'])
        list_features = list(list(meta_features[i].values()) for i in range(0, len(meta_features)))

        self.split_data(meta_features, iteration)
        if len(self.X_train) == 0:
            return Config.CLASSIFIERS[len(Config.CLASSIFIERS)-1]
        #train meta_features label=best_classifier 
        #(with/out classifiers for each meta_features)
        #without the models score(evaluation)-this will be our indication for the model
        #test last meta_features
        #exist_clfs = Config.CLASSIFIERS
        
        
        self.X_train['clf_backwards_1_steps_ttest_stat'] = [0.0 if pd.isna(x) else x for x in self.X_train['clf_backwards_1_steps_ttest_stat']]
        self.X_train['clf_backwards_1_steps_ttest_pval'] = [0.0 if pd.isna(x) else x for x in self.X_train['clf_backwards_1_steps_ttest_pval']]
        self.X_test['clf_backwards_1_steps_ttest_stat'] = [0.0 if pd.isna(x) else x for x in self.X_test['clf_backwards_1_steps_ttest_stat']]
        self.X_test['clf_backwards_1_steps_ttest_pval'] = [0.0 if pd.isna(x) else x for x in self.X_test['clf_backwards_1_steps_ttest_pval']]
        
        self.X_train['clf_backwards_2_steps_ttest_stat'] = [0.0 if pd.isna(x) else x for x in self.X_train['clf_backwards_2_steps_ttest_stat']]
        self.X_train['clf_backwards_2_steps_ttest_pval'] = [0.0 if pd.isna(x) else x for x in self.X_train['clf_backwards_2_steps_ttest_pval']]
        self.X_test['clf_backwards_2_steps_ttest_stat'] = [0.0 if pd.isna(x) else x for x in self.X_test['clf_backwards_2_steps_ttest_stat']]
        self.X_test['clf_backwards_2_steps_ttest_pval'] = [0.0 if pd.isna(x) else x for x in self.X_test['clf_backwards_2_steps_ttest_pval']]
        
        self.X_train['clf_backwards_3_steps_ttest_stat'] = [0.0 if pd.isna(x) else x for x in self.X_train['clf_backwards_3_steps_ttest_stat']]
        self.X_train['clf_backwards_3_steps_ttest_pval'] = [0.0 if pd.isna(x) else x for x in self.X_train['clf_backwards_3_steps_ttest_pval']]
        self.X_test['clf_backwards_3_steps_ttest_stat'] = [0.0 if pd.isna(x) else x for x in self.X_test['clf_backwards_3_steps_ttest_stat']]
        self.X_test['clf_backwards_3_steps_ttest_pval'] = [0.0 if pd.isna(x) else x for x in self.X_test['clf_backwards_3_steps_ttest_pval']]

        # print('is nan')
        # print(self.X_train.isnull().sum())
        # print('is nan any')
        # print(self.X_train.isnull().values.any())
        # print('is nan sum')
        # print(self.X_train.isnull().values.sum())
        # print('is infinity')
        # print(np.isinf(self.X_train).values.sum())
        # # print(np.isinf(self.X_train).values)
        # print(np.isnan(self.X_train).values.sum())
        #
        # print('is nan')
        # print(self.y_train.isnull().sum())
        # print('is nan any')
        # print(self.y_train.isnull().values.any())
        # print('is nan sum')
        # print(self.y_train.isnull().values.sum())
        # print('is infinity')
        # print(np.isinf(self.y_train).values.sum())
        # print(np.isnan(self.y_train).values.sum())

        print('is nan')
        print(self.X_test.isnull().sum())
        print('is nan any')
        print(self.X_test.isnull().values.any())
        print('is nan sum')
        print(self.X_test.isnull().values.sum())
        print('is infinity')
        print(np.isinf(self.X_test).values.sum())



        if np.isinf(self.X_test).values.sum():
            self.X_test = self.X_test.replace([np.inf, -np.inf], 0)
        if np.isinf(self.X_train).values.sum():
            self.X_train = self.X_train.replace([np.inf, -np.inf], 0)

        if np.isnan(self.X_test).values.sum():
            self.X_test = self.X_test.replace([np.nan], 0)
        if self.X_test.isnull().values.sum() > 0:
            self.X_test = self.X_test.replace([np.nan], 0)
        if np.isnan(self.X_train).values.sum():
            self.X_train = self.X_train.replace([np.nan], 0)

        # self.X_test[self.X_train < -10000] = 0
        # self.X_test[self.X_train > 10000] = 0
        # self.X_test[self.X_test < -10000] = 0
        # self.X_test[self.X_test > 10000] = 0


        # BootstrapSample
        views_samples = []

        sample = sklearn.utils.resample(self.X_train, self.y_train, n_samples=int(Config.RESAMPLE_LABELED_RATIO*len(self.y_train))
            , random_state=Config.RANDOM_STATE)

        views_samples.append(sample)
        # self.model.fit(*sample)


        # pred = self.model.predict(self.X_test)
        pred = [1,1,1,1,1,1,1]
        score = sklearn.metrics.accuracy_score(self.y_test, pred)
        
        print("predict:" + str(pred))
        print("y_test: "+ str(self.y_test))
        print("score: "+ str(score))
        
        self.evaluation.append(score)
        self.classifiers.append([Config.CLASSIFIERS[i] for i in pred])

    
        # we choose only one classifier for now
        return self.most_common(self.classifiers[len(self.classifiers)-1])
    
    
    def split_data(self, meta_features, iteration=1):
        #self.metadata = pd.read_csv('../meta_datasets/{}_meta_features.csv'.format(self.metadata_name))
        self.metadata = pd.DataFrame.from_dict(meta_features)
        self.original_metadata = self.metadata # keep the original metadata
        
        ##self.handle_categorical_data()
        self.numeric_cols = self.metadata._get_numeric_data().columns
        self.categorical_cols = list(set(self.metadata.columns) - set(self.numeric_cols))
        for cat_col in self.categorical_cols:
            self.metadata[cat_col] = self.metadata[cat_col].astype('category')
        self.metadata[self.categorical_cols] = self.metadata[self.categorical_cols].apply(lambda x: x.cat.codes)

        self.data = self.metadata.drop(['best_classifier'], axis=1)
        self.label = self.metadata['best_classifier'] #y_label


        
        unique_vals = self.label.unique()
        
        # caused bugs:
        #if len(unique_vals)>1:
         #   self.label = self.label.map({unique_vals[0] : 0, unique_vals[1] : 1})

        self.class_ratio = sum(self.label)/len(self.label)
        
        steps = 3
        
        start_train_data_index = int(len(self.data) - len(Config.CLASSIFIERS)*(steps+1))
        start_train_data_index = max(start_train_data_index, 0)
        end_train_data_index = int(len(self.data) - len(Config.CLASSIFIERS) -1)
        end_test_data_index = int(len(self.data)-1)
        #print("Split data classifiers:")
        #print("iteration: "+str(iteration))
        #print("X_train: "+str(start_train_data_index)+" to "+str(end_train_data_index))
        #print("X_test: "+str(end_train_data_index + 1)+" to "+str(end_test_data_index))
        
        self.X_train = self.data[start_train_data_index : end_train_data_index+1]
        #print('int(len(self.data)')
        #print(str(int(len(self.data))))
        #print('len X_train')
        #print(len(self.X_train))
        self.y_train = self.label[start_train_data_index : end_train_data_index+1]
        self.X_test = self.data[end_train_data_index + 1 : end_test_data_index+1]
        #print('len X-test')
        #print(len(self.X_test))
        self.y_test = self.label[end_train_data_index + 1 : end_test_data_index+1]
        return self.X_train, self.y_train, self.X_test, self.y_test
        
        
        
    def most_common(self, lst):
        return max(set(lst), key=lst.count)
        
        
        
        