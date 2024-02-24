import numpy as np
import statistics
from scipy.stats import ttest_ind, normaltest, skewtest, shapiro
from sklearn.cluster import KMeans
from collections import Counter
from classifier import Classifier
from sklearn.metrics import accuracy_score
from config import Config


class MetaFeaturesExtracion:
    def __init__(self):
        self.view_based_meta_features = dict()
        self.instance_based_meta_features = dict()
        self.dataset_based_meta_features = dict()

          

    def view_based_mf(self, X_test, y_test, test_proba):
        
        view_based_meta_features_current = dict()
        self.test_proba = test_proba
       
        # Agreement features
        view_based_meta_features_current['agreement_set_size'] = len(y_test)
        view_based_meta_features_current['agreement_set_class_size'] = np.sum(y_test)

        kmeans = KMeans(n_clusters=1).fit(X_test)
        kmeans_labels = Counter(kmeans.labels_)
        view_based_meta_features_current['agreement_set_cluster'] = kmeans_labels[0]


        self.view_based_meta_features = view_based_meta_features_current


    def instance_based_mf(self, iteration, classifier_number, batches, 
        view_j_presictions_proba, view_k_presictions_proba, view_i_presictions_proba):
        
        instance_based_meta_features_current = dict()
        

        for index, batch in enumerate(batches):
            if isinstance(batch[0], list):
                batch_union = batch[0] + batch[1]
            else:
                batch_union = batch[0].tolist() + batch[1].tolist()
            instances_j = view_j_presictions_proba[batch_union]
            instances_k = view_k_presictions_proba[batch_union]
            instances_i = view_i_presictions_proba[batch_union]
            
            # Descriprive
            instance_based_meta_features_current.update(
                self.descriptive_statistics('batch_confidence_view_j', instances_j[:,1]))
            instance_based_meta_features_current.update(
                self.descriptive_statistics('batch_confidence_view_k', instances_k[:,1]))
            instance_based_meta_features_current.update(
                self.descriptive_statistics('batch_confidence_view_i', instances_i[:,1]))
            
            # T-Test
            ttest_stat, ttest_pval = ttest_ind(instances_j[:,1], instances_k[:,1])
            instance_based_meta_features_current['batch_view_j_k_ttest_stat'] = ttest_stat
            instance_based_meta_features_current['batch_view_j_k_ttest_pval'] = ttest_pval
            ttest_stat, ttest_pval = ttest_ind(instances_j[:,1], instances_i[:,1])
            instance_based_meta_features_current['batch_view_j_i_ttest_stat'] = ttest_stat
            instance_based_meta_features_current['batch_view_j_i_ttest_pval'] = ttest_pval
            ttest_stat, ttest_pval = ttest_ind(instances_i[:,1], instances_k[:,1])
            instance_based_meta_features_current['batch_view_i_k_ttest_stat'] = ttest_stat
            instance_based_meta_features_current['batch_view_i_k_ttest_pval'] = ttest_pval

            self.instance_based_meta_features[iteration][classifier_number][index].update(instance_based_meta_features_current)


    def dataset_based_mf(self, dataset, classifier):

        self.dataset_based_meta_features['num_categorical_cols'] = len(dataset.categorical_cols)
        self.dataset_based_meta_features['num_numeric_cols'] = len(dataset.numeric_cols)
        self.dataset_based_meta_features['num_instances'] = len(dataset.label)
        self.dataset_based_meta_features['num_labeled_instances'] = len(dataset.y_train)
        self.dataset_based_meta_features['num_unlabeled_instances'] = len(dataset.y_test)
 
        # Initial AUC
        temp_classifier = Classifier(classifier.get_classifier_name).get_classifier()

        temp_classifier.fit(dataset.X_train, dataset.y_train)
        self.dataset_based_meta_features['ada_boost'] = 0
        self.dataset_based_meta_features['desicition_tree'] = 0
        self.dataset_based_meta_features['gaussian_nb'] = 0
        self.dataset_based_meta_features['gaussian_process'] = 0
        self.dataset_based_meta_features['logistic_regression'] = 0
        self.dataset_based_meta_features['mlp'] = 0
        self.dataset_based_meta_features['random_forest'] = 0

        self.dataset_based_meta_features[classifier.get_classifier_name()] = 1
        self.dataset_based_meta_features['initial_auc'] = accuracy_score(dataset.y_test, temp_classifier.predict(dataset.X_test))

        # Mean skewness of numeric attributes

        # Classes ratio (test set)

    def check_distributions(self, i):
        temp_dict = dict()
        # Descriptive statistics for each view
        temp_dict.update(
        self.descriptive_statistics('confidence_clf', 
                                                     self.test_proba[:,1]))
        ## Check distributions
        _, temp_dict['confidence_clf_norm_dist_pval'] = normaltest(self.test_proba[:,1])        
        temp_dict['confidence_clf_skew'], _ = skewtest(self.test_proba[:,1])  
        _, temp_dict['confidence_clf_shapiro_pval'] = shapiro(self.test_proba[:,1])
        
        
        return temp_dict
        

    def descriptive_statistics(self, feature_name_prefix, numbers_list):
        desc_features = dict()
        desc_features['{}_avg'.format(feature_name_prefix)] = np.mean(numbers_list)
        desc_features['{}_min'.format(feature_name_prefix)] = np.min(numbers_list)
        desc_features['{}_max'.format(feature_name_prefix)] = np.max(numbers_list)
        desc_features['{}_median'.format(feature_name_prefix)] = np.median(numbers_list)
        desc_features['{}_std'.format(feature_name_prefix)] = np.std(numbers_list)
        try:
            desc_features['{}_stdev'.format(feature_name_prefix)] = statistics.stdev(numbers_list)
        except:
            desc_features['{}_stdev'.format(feature_name_prefix)] = np.std(numbers_list)
        

        return desc_features
    
    
    def extract_proba_features(self, first_test_proba,i, second_test_proba,j):
        classifiers_meta_features = dict()
        # T-test means comparison for each pairs of views
        ttest_stat, ttest_pval = ttest_ind(first_test_proba[:,1], second_test_proba[:,1])
        classifiers_meta_features['cur_next_clfs_ttest_stat'] = ttest_stat
        classifiers_meta_features['cur_next_clfs_ttest_pval'] = ttest_pval
        return classifiers_meta_features
    
    
    def extract_proba_features_backwards(self, current_test_proba, bacwards_test_proba, steps):
        classifiers_meta_features = dict()
        # T-test means comparison for each pairs of views
        ttest_stat, ttest_pval = ttest_ind(current_test_proba[:,1], bacwards_test_proba[:,1])
        #clf_backwards_'+str(steps)+'_steps_ttest_stat
        classifiers_meta_features['clf_backwards_'+str(steps)+'_steps_ttest_stat'] = ttest_stat
        classifiers_meta_features['clf_backwards_'+str(steps)+'_steps_ttest_pval'] = ttest_pval
        return classifiers_meta_features
    
    def get_backwards_best_clf(self, temp_dict, backwards_meta_features, clf_index, steps, place):
        temp_dict['{}_iters_back_{}_place_ada_boost'.format(steps, place)] = 0
        temp_dict['{}_iters_back_{}_place_desicition_tree'.format(steps, place)] = 0 
        temp_dict['{}_iters_back_{}_place_gaussian_nb'.format(steps, place)] = 0
        temp_dict['{}_iters_back_{}_place_gaussian_process'.format(steps, place)] = 0 
        temp_dict['{}_iters_back_{}_place_logistic_regression'.format(steps, place)] = 0 
        temp_dict['{}_iters_back_{}_place_mlp'.format(steps, place)] = 0
        temp_dict['{}_iters_back_{}_place_random_forest'.format(steps, place)] = 0
        if len(backwards_meta_features) > steps-1:
            if place == 1:
                clf = Config.CLASSIFIERS[backwards_meta_features[len(backwards_meta_features)-steps][clf_index].best_classifier]
            elif place == 2:
                clf = Config.CLASSIFIERS[backwards_meta_features[len(backwards_meta_features)-steps][clf_index].second_place]
            elif place == 3:
                clf = Config.CLASSIFIERS[backwards_meta_features[len(backwards_meta_features)-steps][clf_index].third_place]           
            temp_dict['{}_iters_back_{}_place_{}'.format(steps, place, clf)] = 1
        return temp_dict
    
    def get_ranks_and_auc_gap_backwards(self, temp_dict, evaluation, backwards_meta_features):
        temp_dict['rank'] = self.rank
        temp_dict['AUC_gap_from_top'] = evaluation / self.top_auc
        temp_dict['AUC_gap_from_avg'] = evaluation / self.avg_auc 
        temp_dict['AUC_gap_from_min'] = evaluation / self.bottom_auc
        for steps in range(1,4):
            if len(backwards_meta_features) > steps-1:
                top_auc = backwards_meta_features[len(backwards_meta_features)-steps][1].top_auc
                avg_auc = backwards_meta_features[len(backwards_meta_features)-steps][1].avg_auc
                bottom_auc = backwards_meta_features[len(backwards_meta_features)-steps][1].bottom_auc
                           
                temp_dict['AUC_gap_from_top_{}_steps_back'.format(steps)] = evaluation / top_auc
                temp_dict['AUC_gap_from_avg_{}_steps_back'.format(steps)] = evaluation / avg_auc
                temp_dict['AUC_gap_from_min_{}_steps_back'.format(steps)] =  evaluation / bottom_auc
                
            else:
                temp_dict['AUC_gap_from_top_{}_steps_back'.format(steps)] = 0
                temp_dict['AUC_gap_from_avg_{}_steps_back'.format(steps)] = 0
                temp_dict['AUC_gap_from_min_{}_steps_back'.format(steps)] = 0
        return temp_dict
        
        
        
        
        

        
        
        
        
        
        
        