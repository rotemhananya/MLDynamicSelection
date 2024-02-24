# -*- coding: utf-8 -*-

import glob
import os

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# -*- coding: utf-8 -*-
from config import Config


class FinalClassifierChooser:
    
    def __init__(self):
        self.classifiers = []
        self.evaluation = []
        
    def merge_files(self, test_set_dir, save_dir):
        for dataset in ['ECG200',
                        'electricity-normalized',
                        'FordA',
                        'fri_c0_1000_10',
                        # 'fri_c0_1000_25',
                        'fri_c0_1000_50',
                        'phoneme',
                        'wind',
                        'Yoga',
                        'Strawberry',
                        'HandOutlines',
                        'FordB',
                        'PhalangesOutlinesCorrect',
                        'wafer',
                        'DistalPhalanxOutlineCorrect',
                        'ECGFiveDays',
                        'ItalyPowerDemand',
                        # 'MiddlePhalanxOutlineCorrect',
                        'MoteStrain',
                        'ProximalPhalanxOutlineCorrect',
                        'SonyAIBORobotSurface2',
                        'TwoLeadECG',
                        'Chinatown',
                        'FreezerRegularTrain',
                        # 'GunPointAgeSpan',
                        'PowerCons',
                        # 'CinCECGTorso',
                        'DiatomSizeReduction',
                        'StarLightCurves',
                        'TwoPatterns',
                        'EthanolLevel',
                        'DistalPhalanxTW',
                        'Mallat',
                        'MiddlePhalanxTW',
                        'OSULeaf',
                        'ProximalPhalanxTW',
                        'Symbols',
                        'SyntheticControl',
                        'UWaveGestureLibraryX',
                        'UWaveGestureLibraryY',
                        'UWaveGestureLibraryZ',
                        'MelbournePedestrian',
                        'MedicalImages',
                        'CricketX',
                        'CricketY',
                        'CricketZ',
                        'FacesUCR',
                        'InsectWingbeatSound',
                        'SwedishLeaf',
                        # 'PLAID',
                        'EOGHorizontalSignal',
                        # 'EOGVerticalSignal'
                        ]:
            all_files = glob.glob(os.path.join(test_set_dir, "EXP2_*.csv"))
            all_files.remove(f'{test_set_dir}\\EXP2_{dataset}_AUC_ROC.csv')
            for data in ['fri_c0_1000_25', 'MiddlePhalanxOutlineCorrect', 'GunPointAgeSpan', 'CinCECGTorso' , 'PLAID', 'EOGVerticalSignal']:
                all_files.remove(f'{test_set_dir}\\EXP2_{data}_AUC_ROC.csv')
            df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
            df_merged = pd.concat(df_from_each_file, ignore_index=True)
            numeric_cols = df_merged._get_numeric_data().columns
            categorical_cols = list(set(df_merged.columns) - set(numeric_cols))
            for cat_col in categorical_cols:
                df_merged[cat_col] = df_merged[cat_col].astype('category')
            df_merged[categorical_cols] = df_merged[categorical_cols].apply(lambda x: x.cat.codes)

            try:
                df_merged = df_merged.drop(["Unnamed: 0"], axis=1)
                df_merged = df_merged.drop(["Unnamed: 0.1"], axis=1)
                df_merged = df_merged.drop(["Unnamed: 0.1.1"], axis=1)
                df_merged = df_merged.drop(["Unnamed: 0.1.1.1"], axis=1)
            except:
                print(f'{dataset}: Oh No')
            df_merged.to_csv(f'{save_dir}\\EXP2_{dataset}_train_set.csv')
            # df_merged.to_csv(f'{save_dir}\\EXP2_AllDatasets_train_set.csv')

    def start_classifier_chooser(self, classifier):
        if classifier == 'adaboost':
            self.model = AdaBoostClassifier()
        elif classifier == 'randomforest':
            self.model = RandomForestClassifier()
        elif classifier == 'xgboost':
            self.model = XGBClassifier()
        elif classifier == 'mlp':
            self.model = MLPClassifier()
            
        datasets = Config.DATASET_NAME_LIST_TEST
        for dataset in datasets:
            print('dataset:')
            print(dataset)
            self.classifiers = []
            self.evaluation = []
            x_train, y_train, x_test, y_test = self.train_test_sets(dataset) 
            print('x_train:')
            print(x_train)
            print('y_train:')
            print(y_train)
            self.run_meta_model(x_train, y_train, x_test, y_test)
            self.save_csv(dataset, classifier)
        
    def train_test_sets(self, current_dataset):
        #load data for train
        x_train, y_train = self.load_data("../results/train_sets/EXP2_{}_train_set.csv".format(current_dataset))
        #load data for test
        x_test, y_test = self.load_data("../results/test_sets/EXP2_{}_AUC_ROC.csv".format(current_dataset))
        
        return x_train, y_train, x_test, y_test
               
            
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
        
        
    def run_meta_model(self, X_train, y_train, X_test, y_test):
        views_samples = []
        print(X_train.isnull().values.any())
        print(y_train.isnull().values.any())
        print(X_test.isnull().values.any())
        print(y_test.isnull().values.any())
        sample = sklearn.utils.resample(X_train, y_train, n_samples=int(Config.RESAMPLE_LABELED_RATIO*len(y_train))
            , random_state=Config.RANDOM_STATE)

        views_samples.append(sample)
        self.model.fit(*sample) 
        
        #pred = self.model.predict(X_test) 

        #score = sklearn.metrics.accuracy_score(y_test, pred)
        
        #print("predict:" + str(pred))
        #print("y_test: "+ str(y_test))
        #print("score: "+ str(score))
        
        #self.evaluation.append(score)
        #self.classifiers.append([Config.CLASSIFIERS[i] for i in pred])
        #return pred
    
        # test by steps
        start_test_index = 0
        while start_test_index < len(X_test):
            X_test_step = X_test[start_test_index: start_test_index+len(Config.CLASSIFIERS)]
            y_test_step = y_test[start_test_index: start_test_index+len(Config.CLASSIFIERS)]
            #print('start_test_index')
            #print(start_test_index)
            #print('start_test_index+len(Config.CLASSIFIERS)')
            #print(start_test_index+len(Config.CLASSIFIERS))
            #print('len(test_step)')
            #print(len(X_test_step))
            
            pred = self.model.predict(X_test_step) 
            score = sklearn.metrics.accuracy_score(y_test_step, pred)
            
            print("predict:" + str(pred))
            print("y_test: "+ str(y_test_step))
            print("score: "+ str(score))
            
            self.evaluation.append(score)
            most_common_classifier = self.most_common([Config.CLASSIFIERS[i] for i in pred])
            for index in range(len(Config.CLASSIFIERS)):
                self.classifiers.append(most_common_classifier)
            
            start_test_index += len(Config.CLASSIFIERS)
        return pred


    def save_csv(self, dataset, classifier):
        meta_dataset = {}
        meta_dataset['chosen_classifier'] = self.classifiers
        df =  pd.DataFrame.from_dict(meta_dataset)
        df.to_csv('../results/predictions/EXP2_{}_{}_predictions.csv'.format(dataset, classifier))

        
    def most_common(self, lst):
        return max(set(lst), key=lst.count)
        
        
    