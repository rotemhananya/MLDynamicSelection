from tri_training import TriTraining
from data_handler import DataHandler
import pandas as pd
import os
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
from meta_features_extractor import MetaFeaturesExtracion
from classifier import Classifier
from config import Config
from scipy.stats import ttest_ind, normaltest, skewtest
import itertools
import statistics




class Experiment:
    def __init__(self, ds = Config.DATASET_NAME, label_rate = Config.LABEL_RATE, useModelsScore=False, useModelsEvaluations = False, backwards_best_clf=False):
        self.exp_id = datetime.now()
        self.is_extract_meta_features = True
        self.label_rate = label_rate
        self.exp_results = {}
        self.exp_features = {}
        self.dataset_name = ds
        self.dataset = DataHandler(dataset_name = ds)
        self.dataset_length = len(self.dataset.data)
        self.useModelsEvaluations = useModelsEvaluations
        self.useModelsScore = useModelsScore
        self.backwards_best_clf = backwards_best_clf
        #self.dataset.separate_data()


    def start(self, exp_iteration = 1, steps = 1, chosen_classifiers = []):
        self.iteration = exp_iteration
        L_X, L_y, X_test, y_test = self.dataset.data_split(label_rate=self.label_rate, test_rate=Config.TEST_RATE, iteration=exp_iteration, steps = steps)
        self.classifiers = []
        self.meta_features = []
        self.evaluation = []
        self.acc_score = []
        for i in range (0, len(Config.CLASSIFIERS)):
            self.classifiers.append(Classifier(Config.CLASSIFIERS[i]))
            t_training = TriTraining( self.classifiers[i].get_classifier(), self.is_extract_meta_features)

            if self.is_extract_meta_features:
                metaFeaturesExtracion = MetaFeaturesExtracion()
                self.meta_features.append(metaFeaturesExtracion)
                self.meta_features[i].dataset_based_mf(self.dataset, self.classifiers[i])
                t_training.set_meta_features_extractor(self.meta_features[i])
            t_training.fit(self.dataset)
            self.res = t_training.predict(X_test)
            self.evaluation.append(t_training.score(X_test, y_test))
            self.acc_score.append(accuracy_score(y_test, self.res))


    def export_results(self):
        for i in range (0, len(Config.CLASSIFIERS)):
            self.exp_results[self.dataset_name+"_"+str(self.iteration)+"_"+Config.CLASSIFIERS[i]] = self.evaluation[i]
            print("Accuracy for dataset {} & classifier {}: {}".format(self.dataset_name, Config.CLASSIFIERS[i], self.evaluation[i]))



    def export_meta_features(self, backwards_meta_features = [], classifiers = 'logistic_regression'):


        sorted_clfs_by_score = sorted(range(len(self.evaluation)), key = lambda sub: self.evaluation[sub])[-len(Config.CLASSIFIERS):]
        best_classifier = sorted_clfs_by_score[-1]
        second_place = sorted_clfs_by_score[-2]
        third_place = sorted_clfs_by_score[-3]

        ranks = self.get_ranks(self.evaluation, sorted_clfs_by_score)
        top_auc = max(self.evaluation)
        bottom_auc = min(self.evaluation)
        avg_auc = statistics.mean(self.evaluation)

        list_meta_features = []
        for i in range (0, len(Config.CLASSIFIERS)):
            temp_dict = {}
            # additional data
            temp_dict['iteration'] = self.iteration
            #temp_dict['classifier'] = Config.CLASSIFIERS[i]
            temp_dict['exp_id'] = self.exp_id
            temp_dict['label_rate'] = self.label_rate

            # add meta features
            temp_dict.update(self.meta_features[i].dataset_based_meta_features)
            j = (i + 1) % len(Config.CLASSIFIERS)

            temp_dict.update(self.meta_features[i].extract_proba_features(
                    self.meta_features[i].test_proba,i,
                    self.meta_features[j].test_proba,j))

            for steps in range(1,4):
                temp_dict = self.get_backwards_features(backwards_meta_features, temp_dict, i, steps)

            temp_dict.update(self.meta_features[i].check_distributions(i))
            if self.backwards_best_clf:
                for steps in range(1,4):
                    for place in range(1,4):
                        temp_dict = self.meta_features[i].get_backwards_best_clf(temp_dict, backwards_meta_features, i, steps, place)

            self.meta_features[i].best_classifier = best_classifier
            self.meta_features[i].second_place = second_place
            self.meta_features[i].third_place = third_place
            self.meta_features[i].rank = ranks[i]
            self.meta_features[i].top_auc = top_auc
            self.meta_features[i].bottom_auc = bottom_auc
            self.meta_features[i].avg_auc = avg_auc
            temp_dict = self.meta_features[i].get_ranks_and_auc_gap_backwards(temp_dict, self.evaluation[i], backwards_meta_features)

            if self.useModelsScore:
                temp_dict['accuracy'] = self.acc_score[i]
            if self.useModelsEvaluations:
                temp_dict['evaluation'] = self.evaluation[i]
            temp_dict['best_classifier'] = best_classifier

            list_meta_features.append(temp_dict)




        return list_meta_features, self.meta_features, best_classifier


    def get_ranks(self, evaluations, sorted_clfs_by_score):
        #calculate rank for each model in the iteration
        score = 0
        ranks = [0,0,0,0,0,0,0]
        for i in range(0, len(ranks)):
            if i>0 and evaluations[sorted_clfs_by_score[i]] == evaluations[sorted_clfs_by_score[i-1]]:
                score-=1
            ranks[sorted_clfs_by_score[i]] += score
            score+=1
        return ranks

    def get_backwards_features(self, backwards_meta_features, temp_dict, clf_index,  steps=1):
        if len(backwards_meta_features) > steps-1:
            temp_dict.update(self.meta_features[clf_index].extract_proba_features_backwards(
                self.meta_features[clf_index].test_proba,
                backwards_meta_features[len(backwards_meta_features)-steps][clf_index]
                .test_proba, steps))
        else:
            nan_dict = dict()
            nan_dict['clf_backwards_'+str(steps)+'_steps_ttest_stat'] = 0
            nan_dict['clf_backwards_'+str(steps)+'_steps_ttest_pval'] = 0
            temp_dict.update(nan_dict)
        return temp_dict




    def save_csv(self, list_meta_features, chosen_classifiers= [], classes = 2):

        meta_dataset = pd.DataFrame.from_dict(list_meta_features)

        updated_chosen_classifiers = list(itertools.chain.from_iterable(itertools.repeat(x, len(Config.CLASSIFIERS)) for x in chosen_classifiers))
        meta_dataset['chosen_classifier'] = updated_chosen_classifiers

        num_of_columns = len(meta_dataset.columns)

        # change columns order
        cols = meta_dataset.columns.tolist()

        cols = cols[:num_of_columns-3] + cols[-1:] + [cols[len(cols)-3]] + [cols[len(cols)-2]]

        meta_dataset = meta_dataset[cols]


        #for i in range (0, len(Config.CLASSIFIERS)):
        #    print(list(meta_dataset.columns.values))
        outname = 'EXP2_{}_AUC_ROC.csv'.format(self.dataset_name)
        outdir = '../meta_datasets'

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        fullname = os.path.join(outdir, outname)
        meta_dataset.to_csv(fullname, index=False)

