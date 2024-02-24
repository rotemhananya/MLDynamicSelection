import numpy as np
import sklearn
import math
import pandas as pd
from itertools import combinations, product
from sklearn import preprocessing
from config import Config
import sys
sys.path.append('/path/to/MLWithDynamicEnv/featuresExtruction/MetaLearning')


class TriTraining:
    def __init__(self, classifier, is_extract_meta_features = False):

        self.is_extract_meta_features = is_extract_meta_features
        self.classifier = sklearn.base.clone(classifier)

    def fit(self, dataset):
        if dataset is not None:
            X_label, y_label, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
            self.dataset_exp = dataset
        self.X_label = X_label
        self.y_label = y_label

        # BootstrapSample
        views_samples = []
        
        sample = sklearn.utils.resample(self.X_label, self.y_label, n_samples=int(Config.RESAMPLE_LABELED_RATIO*len(self.y_label))
            , random_state=Config.RANDOM_STATE)
        views_samples.append(sample)
        self.classifier.fit(*sample) 

        # Initial variables 
        classification_error_current = 0.5
        pseudo_label_size = 0
        X_pseudo_label_index = []
        X_pseudo_label_index_current = []
        update = False
        improve = True
        
        X_pseudo_label_index_current = X_pseudo_label_index    
        update = False

        # Confidence score
        test_proba = self.classifier.predict_proba(X_test)

        #    # Get meta features
        if self.is_extract_meta_features:
            self.meta_features_extractor.view_based_mf(X_test, y_test, test_proba)
        


    def predict(self, X):
        pred = self.classifier.predict(X) 
        return pred
        
    def score(self, X, y):
        #return sklearn.metrics.accuracy_score(y, self.predict(X))
        #precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y, self.predict(X))
        #return sklearn.metrics.auc(recall, precision)
        pred = self.predict(X)
        if max(pred) > 1 or max(y) > 1:
            lb = preprocessing.LabelBinarizer()
            lb.fit(y)
            lb.classes_
            y_true = lb.transform(y)
            # print(y_true)
            return sklearn.metrics.roc_auc_score(y_true, self.classifier.predict_proba(X), multi_class='ovr', labels=[i for i in range(3)])
        else:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred)
            return sklearn.metrics.auc(fpr, tpr)

    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        wrong_index =np.logical_and(j_pred != y, k_pred==j_pred)
        return sum(wrong_index)/sum(j_pred == k_pred)

    def set_meta_features_extractor(self, mf_extractor):
        self.meta_features_extractor = mf_extractor

    def top_confidence(self, confidence_list):
        class_1_conf = confidence_list[:,1]
        
        condidates_per_class = int(len(class_1_conf) * Config.TOP_CONFIDENCE_RATIO)
        top_class_0 = np.argsort(class_1_conf)[:condidates_per_class]
        top_class_1 = np.argsort(class_1_conf)[-condidates_per_class:]
        return top_class_0, top_class_1

    def generate_labeling_candidates(self, confidence_list_j, confidence_list_k, agree_list, batch_size):
        # ToDo: add the elements to the agreed: if the agreed>= batch size, than sample. else: add all agreed and add top instances.
        candidates = []
        classes_ratio = self.dataset_exp.class_ratio
        class_1_ratio = int(Config.BATCH_SIZE*classes_ratio)
        class_0_ratio = Config.BATCH_SIZE - class_1_ratio
        
        # Get instances with highest labeling confidence
        candidates_j_class_0, candidates_j_class_1 = self.top_confidence(confidence_list_j)
        candidates_k_class_0, candidates_k_class_1 = self.top_confidence(confidence_list_k)
        agree_class_0, agree_class_1 = np.where(agree_list==0), np.where(agree_list==1)

        agree_candidates = []
        if len(agree_class_1[0]) >= class_1_ratio and len(agree_class_0[0]) >= class_0_ratio:
            for batch_i in range(int(0.1*Config.NUM_BATCHES)):
                candidates_classes = []
                agree_class_1_tmp = np.random.choice(agree_class_1[0].tolist(), class_1_ratio)
                agree_class_0_tmp = np.random.choice(agree_class_0[0].tolist(), class_0_ratio)
                candidates_classes.append(agree_class_0_tmp)
                candidates_classes.append(agree_class_1_tmp)
                agree_candidates.append(candidates_classes)

        if len(agree_candidates) >= Config.NUM_BATCHES:
            candidates = agree_candidates[:Config.NUM_BATCHES]
            return candidates

        for batch_i in range(Config.NUM_BATCHES - len(agree_candidates)):
            candidates_classes = []
            class_1_concat = list(dict.fromkeys(candidates_j_class_1.tolist() + candidates_k_class_1.tolist() + agree_class_1[0].tolist()))
            class_0_concat = list(dict.fromkeys(candidates_j_class_0.tolist() + candidates_k_class_0.tolist() + agree_class_0[0].tolist()))
            
            class_1 = sklearn.utils.resample(class_1_concat
                ,replace=False,n_samples=class_1_ratio, random_state=Config.RANDOM_STATE + batch_i)
            class_0 = sklearn.utils.resample(class_0_concat
                ,replace=False,n_samples=class_0_ratio, random_state=Config.RANDOM_STATE + batch_i)

            candidates_classes.append(class_0)
            candidates_classes.append(class_1)
            candidates.append(candidates_classes)

        if len(agree_candidates)> 0:
            candidates = agree_candidates + candidates

        return np.asarray(candidates)