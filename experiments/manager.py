
import RNNClassifierChooser as rnnClassifierChooser
from RNN_EXP2 import RNNClassifierChooser
from sklearn.exceptions import ConvergenceWarning
import warnings
from config import Config
import sys
import numpy as np
from scipy import stats
import Autoencoder as autoencoder
import EncoderDecoderExperiment as EncoderDecoder
import ChooseClassifiers as clf
import Experiment as ex


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
from scipy.stats import ttest_ind


def t_test(x, y, alternative='both-sided'):
    _, double_p = ttest_ind(x, y, equal_var=False)
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p / 2.
        else:
            pval = 1.0 - double_p / 2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p / 2.
        else:
            pval = 1.0 - double_p / 2.
    return pval


if __name__ == "__main__":

    # Temp: New experiment for paper1: lag=3
    RnnClassifierChooser = rnnClassifierChooser.RNNClassifierChooser()
    RnnClassifierChooser.start_classifier_chooser(
        save_dir=r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\new_predictions_3lag')
    sys.exit()


    # Papaer1: FinalClassifierChooser: create RNN train sets
    # print("FinalClassifierChooser")
    # finalClassifierChooser = f.FinalClassifierChooser()
    # finalClassifierChooser.merge_files(test_set_dir=r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\test_sets',
    # save_dir=r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\meta_datasets\RNNTrainSet')
    # sys.exit()


    '''
    #Experiment1: run classifiers
    for ds in Config.DATASET_NAME_LIST_TEST:
        num_steps = 1
        exp = ex.Experiment(ds, label_rate=0.8, useModelsEvaluations=True, backwards_best_clf=True)
        classifiers_choose = clf.ChooseClassifiers(ds)
        list_meta_features = []
        meta_features_ext = []
        chosen_classifiers = []
        real_classifiers = []
        classifier = 'random_forest'
        iteration = 1
        boolean = True
        while boolean or int(exp.dataset_length*(iteration)*0.1)<exp.dataset_length:
           #try:
                print("start")
                exp.start(iteration, steps = num_steps, chosen_classifiers = classifier)

                print("export meta features")
                list_meta_features_clf, meta_features, best_classifier = exp.export_meta_features(meta_features_ext, classifier)
                list_meta_features += list_meta_features_clf
                meta_features_ext.append(meta_features)
                real_classifiers.append(best_classifier)

                print("export results")
                classifier = classifiers_choose.calculate_next_step(list_meta_features, iteration)
                exp.export_results()

                print("chosen classifier: "+classifier)
                chosen_classifiers.append(classifier)
                exp.save_csv(list_meta_features, chosen_classifiers)

                num_steps += 1
                iteration += 1
                boolean = False

                #if num_steps > 5:
                #    num_steps = 5


            #except:
                #print("Failed for dataset: {}".format(ds))


    print("The chosen classifiers are:")
    num_of_match_classifiers = 0
    for i in range(0, len(chosen_classifiers)):
        print("iteration {} : {}".format(i, chosen_classifiers[i]))
        if (Config.CLASSIFIERS[real_classifiers[i]] == chosen_classifiers[i]):
            num_of_match_classifiers+=1
    print("Real Classifiers:")
    print(real_classifiers)
    print("number of matches = "+str(num_of_match_classifiers))
    '''


    # '''
    # RNN for exp2 paper1
    # RnnClassifierChooser = rnnClassifierChooser.RNNClassifierChooser()
    # RnnClassifierChooser.start_classifier_chooser(save_dir=r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\new_predictions')
    # sys.exit()
    # '''

    '''
        # old RNN for paper 1
        # FinalClassifierChooser: run meta model
        finalClassifierChooser = f.FinalClassifierChooser()
        finalClassifierChooser.start_classifier_chooser('adaboost') 

        finalClassifierChooser = f.FinalClassifierChooser()
        finalClassifierChooser.start_classifier_chooser('randomforest') 

        finalClassifierChooser = f.FinalClassifierChooser()
        finalClassifierChooser.start_classifier_chooser('xgboost') 

        finalClassifierChooser = f.FinalClassifierChooser()
        finalClassifierChooser.start_classifier_chooser('mlp') 
        '''
