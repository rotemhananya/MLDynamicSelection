
'''
## convert multiclass to binary class
import csv
import sys

with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\datasets\\MelbournePedestrian.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\datasets\\MelbournePedestrian_new.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            print(row[len(row)-1])
            row[len(row)-1] = int(row[len(row)-1])%2
            spamwriter.writerow(row)

    sys.exit()
'''

'''
## organize dataset with both classes in each view
import csv
import sys

with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\datasets\\FreezerRegularTrain.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    n_count = 0
    p_count = 0
    first_array = []
    second_array = []
    for row in csv_reader:
      print(row[len(row)-1])
      if row[len(row)-1] == '1':
        n_count+=1
        first_array.append(row)
      elif row[len(row)-1] == '2':
        p_count+=1
        second_array.append(row)
    balance = n_count/(n_count+p_count)
    print('N count is '+str(n_count))
    print('P count is ' + str(p_count))
    print('balance is '+str(balance))
    with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\datasets\\FreezerRegularTrain_new.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        first_index = 0
        second_index = 0
        while True:
            if first_index >= len(first_array):
                for j in range(second_index,len(second_array)):
                    spamwriter.writerow(second_array[j])
                break
            elif second_index >= len(second_array):
                for i in range(first_index, len(first_array)):
                    spamwriter.writerow(first_array[i])
                break
            for i in range(first_index, min(first_index + int(balance*0.1*(n_count+p_count)), len(first_array))):
                spamwriter.writerow(first_array[i])
            first_index = min(first_index + int(balance*0.1*(n_count+p_count)), len(first_array))
            print('first_index '+str(first_index))
            for j in range(second_index, min(second_index + int((1-balance)*0.1*(n_count+p_count))+1, len(second_array))):
                spamwriter.writerow(second_array[j])
            second_index = min(second_index + int((1-balance)*0.1*(n_count+p_count))+1, len(second_array))
            print('second_index ' +str(second_index))

sys.exit()

'''

#
# # get top classifier in first iteration
# import csv
# import sys
# for dataset in datasets:
#     with open(r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\test_sets\EXP2_{}_AUC_ROC.csv'.format(dataset)) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#                       'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#         index = 0
#         first_row = True
#         evals = [0.0]*len(models)
#         scores = [0,0,0,0,0,0,0]
#         i = 0
#         for row in csv_reader:
#             if first_row:
#                 first_row = False
#                 continue
#             if i>6:
#                 break
#             #print('evaluation: '+str(row[len(row)-2]))
#             scores[i]+=float(row[len(row)-2])
#             i+=1
#         places = sorted(range(len(scores)), key = lambda sub: scores[sub])[-len(scores):]
#         print('datadet: '+dataset+' '+models[places[len(scores)-1]])
# sys.exit()
#


#
#
# # get AUC average of classifiers in the first iteration
# import csv
# import sys
# for dataset in datasets:
#     with open(r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\test_sets\EXP2_{}_AUC_ROC.csv'.format(dataset)) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#                       'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#         index = 0
#         first_row = True
#         evals = [0.0]*len(models)
#         scores = [0,0,0,0,0,0,0]
#         i = 0
#         for row in csv_reader:
#             if first_row:
#                 first_row = False
#                 continue
#             if i>6:
#                 break
#             #print('evaluation: '+str(row[len(row)-2]))
#             scores[i]+=float(row[len(row)-2])
#             i+=1
#         print(dataset+','+str(sum(scores)/7))
# sys.exit()


# '''
# # get classifiers average of drops from top
# import csv
# import sys
# total_drops_sum = 0
# total_drops_count = 0
# total_same_top_sum = 0
# total_same_top_count = 0
# for dataset in datasets:
#     with open(r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\test_sets\EXP2_{}_AUC_ROC.csv'.format(dataset)) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#                       'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#         index = 0
#         first_row = True
#         first_time = True
#         done = False
#         drop = 0
#         count_drops = 0
#         last_top_index = -1
#         count_same_top = 0
#         count_models_same_top = 0
#         i = 0
#         evals = [0.0] * len(models)
#         scores = [0, 0, 0, 0, 0, 0, 0]
#         for row in csv_reader:
#             if first_row:
#                 first_row = False
#                 continue
#             if i%7 == 0:
#                 if first_time:
#                     first_time = False
#                 else:
#                     places = sorted(range(len(scores)), key=lambda sub: scores[sub])[-len(scores):]
#                     if last_top_index is not -1:
#                         if last_top_index is not places[len(scores) - 1]:
#                             model_current_place = 7 - places.index(last_top_index)
#                             drop += 1 - model_current_place #calculate the drop from top
#                             count_drops+=1
#                             count_models_same_top+=1
#                         else:
#                             count_same_top+=1
#
#                     last_top_index = places[len(scores) - 1]
#                     evals = [0.0] * len(models)
#                     scores = [0, 0, 0, 0, 0, 0, 0]
#             #print('evaluation: '+str(row[len(row)-2]))
#             scores[i%7]+=float(row[len(row)-2])
#             i+=1
#         places = sorted(range(len(scores)), key=lambda sub: scores[sub])[-len(scores):]
#         if last_top_index is not -1:
#             if last_top_index is not places[len(scores) - 1]:
#                 model_current_place = 7 - places.index(last_top_index)
#                 drop += 1 - model_current_place  # calculate the drop from top
#                 count_drops += 1
#                 count_models_same_top+=1
#             else:
#                 count_same_top += 1
#         #for the last top model
#         count_models_same_top+=1
#
#         last_top_index = places[len(scores) - 1]
#         total_drops_sum+=drop
#         total_drops_count+=count_drops
#         total_same_top_sum+=count_models_same_top
#
#         if drop is  not 0:
#             drop /= count_drops
#         #print(dataset +' & '+ '9' +' & '+ str(count_drops) +' & '+ str(count_same_top) +' \\\\ ')
#         print('datadet: '+dataset+' drops '+str(drop))
#         print('datadet: ' + dataset + ' count_drops ' + str(count_drops))
#         print('datadet: ' + dataset + ' count_same_top ' + str(count_same_top))
# print('total_drops_sum ' + str(total_drops_sum))
# print('total_drops_count ' + str(total_drops_count))
# print('total_same_top_sum ' + str(total_same_top_sum))
# # If the model drops, in average it drops by total_drops_sum/total_drops_count
# sys.exit()
#
# '''
#
# '''
#
# # get top classifier in two first iterations
# import csv
# import sys
# with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\results\\test_sets\\EXP2_EOGHorizontalSignal_AUC_ROC.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#               'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#     index = 0
#     first_row = True
#     evals = [0.0]*len(models)
#     scores = [0,0,0,0,0,0,0]
#     i = 0
#     for row in csv_reader:
#         if first_row:
#             first_row = False
#             continue
#         if i>13:
#             break
#         #print('evaluation: '+str(row[len(row)-2]))
#         scores[i%len(models)]+=float(row[len(row)-2])
#         i+=1
#     places = sorted(range(len(scores)), key = lambda sub: scores[sub])[-len(scores):]
#     print(models[places[len(scores)-1]])
#     sys.exit()
#
#
# '''
#
# ## delete irrelevant rows from PAMPAS dataset
# # import csv
# # import sys
#
# # with open('C:\\Users\\Rotem\\Downloads\\PAMAP2_Dataset\\PAMAP2_Dataset\\Protocol\\subject106.csv') as csv_file:
# #    csv_reader = csv.reader(csv_file, delimiter=',')
# #    with open('C:\\Users\\Rotem\\Downloads\\PAMAP2_Dataset\\PAMAP2_Dataset\\Protocol\\fixed-subject106.csv', 'w', newline='') as csvfile:
# #        spamwriter = csv.writer(csvfile, delimiter=',',
# #                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
# #        i=0
# #        for row in csv_reader:
# #            if row[1] is '0':
# #                print(i)
# #                i+=1
# #                continue
# #            spamwriter.writerow(row)
# #            i+=1
#
# #        sys.exit()
#
#
# ## change even target to '0' and odd target to '1'
# # import csv
# # import sys
#
# # with open('C:\\Users\\Rotem\\Downloads\\PAMAP2_Dataset\\PAMAP2_Dataset\\Protocol\\PAMAP2-subject103.csv') as csv_file:
# #    csv_reader = csv.reader(csv_file, delimiter=',')
# #    with open('C:\\Users\\Rotem\\Downloads\\PAMAP2_Dataset\\PAMAP2_Dataset\\Protocol\\pamap_subject103.csv', 'w', newline='') as csvfile:
# #        spamwriter = csv.writer(csvfile, delimiter=',',
# #                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
# #        i=0
# #        for row in csv_reader:
# #            if int(row[1])%2 == 0:
# #                row[1] = '0'
# #            else:
# #                row[1] = '1'
# #            spamwriter.writerow(row)
# #            i+=1
#
# #        sys.exit()
#
# '''
# #Friedman Test
# import csv
# import sys
# from scipy.stats import friedmanchisquare
#
# with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\results\\test_sets\\11_4_2021_Yoga_AUC_ROC.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\results\\predictions\\Yoga_predictions.csv') as prediction_file:
#         prediction_reader = csv.reader(prediction_file, delimiter=',')
#         models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#                   'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#         models_eval = [[],[],[],[],[],[],[]]
#         index = 0
#         first_row = True
#         chosen_classifier = []
#         for row, pred in zip(csv_reader,prediction_reader):
#             if first_row:
#                 first_row = False
#                 print('row[len(row) - 2] ' + row[len(row) - 2])
#                 continue
#             print("line "+ str(index) +" "+ models[index%len(models_eval)] +" "+ str(row[len(row)-1]))
#             print('row[len(row)-1] '+row[len(row)-1])
#             models_eval[index%len(models_eval)].append(float(row[len(row)-1]))
#             print('pred[1]')
#             print(pred[1])
#             print('row[1]')
#             print(row[1])
#             if pred[1] == row[1]:
#                 print('True')
#                 chosen_classifier.append(float(row[len(row)-1]))
#             index+=1
#
#         print()
#         i = 0
#         for model in models:
#             print('model '+model+' scores '+str(models_eval[i]))
#             print(len(models_eval[i]))
#             i+=1
#         print('model chosen_classifier scores '+str(chosen_classifier))
#         print(len(chosen_classifier))
#
#         statistic, pvalue = friedmanchisquare(models_eval[0],models_eval[1],models_eval[2],
#                                               models_eval[3],models_eval[4],models_eval[5],
#                                               models_eval[6],chosen_classifier)
#         print('statistic: ')
#         print(statistic)
#         print('pvalue: ')
#         print(pvalue)
#
#         sys.exit()
# '''
#
# '''
# # ClassifierChooser AVG AUC score EXP1
# import csv
#
# with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\meta_datasets\\EXP1_wafer_AUC_ROC.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#               'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#     models_eval = [0.0]*len(models)
#     index = 0
#     first_row = True
#     sum_of_chosen_classifier = 0
#     for row in csv_reader:
#         if first_row:
#             first_row = False
#             print('row[len(row) - 2] ' + row[len(row) - 2])
#             continue
#         print("line "+ str(index) +" "+ models[index%len(models_eval)] +" "+ str(row[len(row)-2]))
#         print('row[len(row)-2] '+row[len(row)-2])
#         models_eval[index%len(models_eval)]+=float(row[len(row)-2])
#         print('row[-3]')
#         print(row[-3])
#         print('models[index%len(models_eval)]')
#         print(models[index%len(models_eval)])
#         if row[-3] == models[index%len(models_eval)]:
#             print('True')
#             sum_of_chosen_classifier+=float(row[len(row)-2])
#         index+=1
#
#         print()
#         i = 0
#         for model in models:
#             print('model '+model+' score '+str(models_eval[i]/9))
#             i+=1
#         print('model chosen_classifier score '+str(sum_of_chosen_classifier/9))
#     #    min_eval = models_eval[0]
#     #    worst_classifier = ''
#     #    for i in range(0, len(models_eval)):
#     #      print('model '+models[i]+' with score sum '+str(models_eval[i]))
#     #      if (models_eval[i] < min_eval):
#     #        min_eval = models_eval[i]
#     #        worst_classifier = models[i]
#     #    print('worst model: '+worst_classifier)
#     #    print('sum_of_chosen_classifier '+str(sum_of_chosen_classifier))
#     #    print('score '+str((sum_of_chosen_classifier-min_eval)/9))
#
# ##row[len(row)-3] == ChosenClassifier
# ##row[1] == iteration number
# ##row[len(row)-1] == model evaluation
# ## in useEval: row[len(row)-4] == model evaluation
#
# '''
#
#
# '''
# # old: ClassifierChooser AVG AUC score EXP2
# import csv
# import sys
# for dataset in datasets:
#
#     with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\results\\test_sets\\EXP2_{}_AUC_ROC.csv'.format(dataset)) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',', skiprows=7)
#         with open(r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\paper1_next_dense_predictions\EXP2_{}_RNN_predictions.csv'.format(dataset)) as prediction_file:
#             prediction_reader = csv.reader(prediction_file, delimiter=',')
#             models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#                       'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#             classifiers = []
#             models_eval = [0.0]*len(models)
#             index = 0
#             first_row = True
#             sum_of_chosen_classifier = 0
#             for row, pred in zip(csv_reader[7:],prediction_reader):
#               if first_row:
#                 first_row = False
#                 continue
#
#               models_eval[index%len(models_eval)]+=float(row[len(row)-2])
#
#               if pred[1] == models[index%len(models_eval)]:
#                 classifiers.append(index%len(models_eval))
#                 sum_of_chosen_classifier+=float(row[len(row)-2])
#               index+=1
#
#             print()
#             i = 0
#             for model in models:
#                 #print('model '+model+' score '+str(models_eval[i]/9))
#                 i+=1
#             print('dataset: '+dataset)
#             print('model chosen_classifier score '+str(sum_of_chosen_classifier/9))
#             print('chosen classifiers: '+str(classifiers))
#         #    min_eval = models_eval[0]
#         #    worst_classifier = ''
#         #    for i in range(0, len(models_eval)):
#         #      print('model '+models[i]+' with score sum '+str(models_eval[i]))
#         #      if (models_eval[i] < min_eval):
#         #        min_eval = models_eval[i]
#         #        worst_classifier = models[i]
#         #    print('worst model: '+worst_classifier)
#         #    print('sum_of_chosen_classifier '+str(sum_of_chosen_classifier))
#         #    print('score '+str((sum_of_chosen_classifier-min_eval)/9))
# sys.exit()
#
# ##row[len(row)-3] == ChosenClassifier
# ##row[1] == iteration number
# ##row[len(row)-1] == model evaluation
# ## in useEval: row[len(row)-4] == model evaluation
# '''
#
# # AUC average
#
# # ClassifierChooser AVG AUC score EXP2
# import csv
# import sys
# import pandas as pd
datasets = [
    'ECG200',
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
    ]
# final_results = []
# num_of_itrs = 6 #TODO 1step=8itrs, 2steps=7itrs, 3steps=6itrs
# for dataset in datasets:
#     with open(rf"C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\test_sets\EXP2_{dataset}_AUC_ROC.csv") as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         with open(rf'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\new_predictions_3lag\EXP2_{dataset}_RNN_predictions.csv') as prediction_file:
#             prediction_reader = csv.reader(prediction_file, delimiter=',')
#             models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#                       'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#             classifiers = []
#             models_eval = [0.0]*len(models)
#             index = 0
#             first_row = True
#             sum_of_chosen_classifier = 0
#             count_irrelevant_rows = 0
#             for irrelevant_row in csv_reader:
#               if count_irrelevant_rows == 20: #TODO 1step=6rows, 2steps=13rows, 3steps=20rows
#                 break
#               count_irrelevant_rows+=1
#             for row, pred in zip(csv_reader,prediction_reader):
#               if first_row:
#                 first_row = False
#                 # print(row) #TODO
#                 # print(pred)
#                 # sys.exit()
#                 continue
#
#               models_eval[index%len(models_eval)]+=float(row[len(row)-2])
#
#               if pred[1] == models[index%len(models_eval)]:
#                 classifiers.append(index%len(models_eval))
#                 sum_of_chosen_classifier+=float(row[len(row)-2])
#               index+=1
#
#             print()
#             i = 0
#             for model in models:
#                 #print('model '+model+' score '+str(models_eval[i]/num_of_itrs))
#                 i+=1
#             print('dataset: '+dataset)
#             print('model chosen_classifier score '+str(sum_of_chosen_classifier/num_of_itrs))
#             for i in range(len(models_eval)):
#               print('model '+models[i]+' with score '+str(models_eval[i]/num_of_itrs))
#               models_eval[i] = models_eval[i]/num_of_itrs
#             print('chosen classifiers: '+str(classifiers))
#         #    min_eval = models_eval[0]
#         #    worst_classifier = ''
#         #    for i in range(0, len(models_eval)):
#         #      print('model '+models[i]+' with score sum '+str(models_eval[i]))
#         #      if (models_eval[i] < min_eval):
#         #        min_eval = models_eval[i]
#         #        worst_classifier = models[i]
#         #    print('worst model: '+worst_classifier)
#         #    print('sum_of_chosen_classifier '+str(sum_of_chosen_classifier))
#         #    print('score '+str((sum_of_chosen_classifier-min_eval)/num_of_itrs))
#     current_result = [dataset]+models_eval+[str(sum_of_chosen_classifier/num_of_itrs)]
#     final_results.append(current_result)
# final_results_df = pd.DataFrame (final_results, columns= ['dataset']+models+ ['ADEN AUC average'])
# print(final_results_df)
# final_results_df.to_csv(r'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\new_predictions_3lag\AUC_3_steps_forward.csv')
# sys.exit()

# Ranking score

# ClassifierChooser Ranking score paper1 EXP2
import csv
import sys
import pandas as pd

final_results = []
for dataset in datasets:

    with open(rf"C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\test_sets\EXP2_{dataset}_AUC_ROC.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with open(
                rf'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\new_predictions_3lag\EXP2_{dataset}_RNN_predictions.csv') as prediction_file:
            prediction_reader = csv.reader(prediction_file, delimiter=',')
            models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
                      'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']

            count_irrelevant_rows = 0
            for irrelevant_row in csv_reader:
                if count_irrelevant_rows == 20:  # TODO 1step=6rows, 2steps=13rows, 3steps=20itrs
                    break
                count_irrelevant_rows += 1

            models_rank = [0] * len(models)
            index = 0
            first_row = True
            chosen_classifier_eval = 0
            evals = [0.0] * len(models)
            ranking = [0, 0, 0, 0, 0, 0, 0, 0]
            i = 0
            for row, pred in zip(csv_reader, prediction_reader):
                if first_row:
                    first_row = False
                    continue
                if i == 7:

                    ranks = evals.copy()
                    ranks.append(chosen_classifier_eval)
                    places = sorted(range(len(ranks)), key=lambda sub: ranks[sub])[-len(ranks):]

                    score = 7
                    # print('ranks: ')
                    # print(ranks)
                    for j in range(len(ranking) - 1, -1, -1):
                        if j < 7 and ranks[places[j]] == ranks[places[j + 1]]:
                            # print('same score indexes: '+str(places[j])+' '+str(places[j+1]))
                            score += 1
                        # print('ranking in index '+str(places[j])+' get score '+str(score))
                        ranking[places[j]] += score
                        score -= 1
                    i = 0

                evals[index % len(evals)] = float(row[len(row) - 2])
                if pred[1] == models[index % len(evals)]:
                    chosen_classifier_eval = float(row[len(row) - 2])

                index += 1
                i += 1

            # for the last iteration
            ranks = evals.copy()
            ranks.append(chosen_classifier_eval)
            places = sorted(range(len(ranks)), key=lambda sub: ranks[sub])[-len(ranks):]
            score = 7
            # print('ranks: ')
            # print(ranks)
            for j in range(len(ranking) - 1, -1, -1):
                if j < 7 and ranks[places[j]] == ranks[places[j + 1]]:
                    # print('same score indexes: '+str(places[j])+' '+str(places[j+1]))
                    score += 1
                # print('ranking in index '+str(places[j])+' get score '+str(score))
                ranking[places[j]] += score
                score -= 1

            print()
            print('dataset: ' + dataset)
            sum_rankings = sum(ranking)
            for i in range(len(models)):
                print('model ' + models[i] + ' - ranking score ' + str(ranking[i] / sum_rankings))
                ranking[i] = ranking[i] / sum_rankings

            print('model chosen_classifier - ranking score ' + str(ranking[-1] / sum_rankings))
            ranking[-1] = ranking[-1] / sum_rankings
    current_result = [dataset] + ranking
    final_results.append(current_result)

final_results_df = pd.DataFrame(final_results, columns=['dataset'] + models + ['ADEN Ranking average'])
print(final_results_df)
final_results_df.to_csv(rf'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\new_predictions_3lag\Ranking_3_steps_forward.csv')
sys.exit()

# '''
#
# # ClassifierChooser Ranking score paper1 EXP1
# import csv
# import sys
# with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\meta_datasets\\EXP1_wafer_AUC_ROC.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#               'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#     models_rank = [0]*len(models)
#     index = 0
#     first_row = True
#     chosen_classifier_eval = 0
#     evals = [0.0]*len(models)
#     ranking = [0,0,0,0,0,0,0,0]
#     i = 0
#     for row in csv_reader:
#         if first_row:
#             first_row = False
#             continue
#         if i == 7:
#             ranks = evals.copy()
#             ranks.append(chosen_classifier_eval)
#             places = sorted(range(len(ranks)), key = lambda sub: ranks[sub])[-len(ranks):]
#             score = 0
#             print('ranks: ')
#             print(ranks)
#             for i in range(0, len(ranking)):
#                 if i>0 and ranks[places[i]] == ranks[places[i-1]]:
#                     #print('same score indexes: '+str(places[i])+' '+str(places[i-1]))
#                     score-=1
#                 #print('ranking in index '+str(places[i])+' get score '+str(score))
#                 ranking[places[i]] += score
#                 score+=1
#             i = 0
#
#         evals[index%len(evals)] = float(row[len(row)-2])
#         if row[-3] == models[index%len(evals)]:
#             chosen_classifier_eval = float(row[len(row)-2])
#
#         index+=1
#         i+=1
#
#     # for the last iteration
#     ranks = evals.copy()
#     ranks.append(chosen_classifier_eval)
#     places = sorted(range(len(ranks)), key = lambda sub: ranks[sub])[-len(ranks):]
#     score = 0
#     print('ranks: ')
#     print(ranks)
#     for i in range(0, len(ranking)):
#         if i>0 and ranks[places[i]] == ranks[places[i-1]]:
#             #print('same score indexes: '+str(places[i])+' '+str(places[i-1]))
#             score-=1
#         #print('ranking in index '+str(places[i])+' get score '+str(score))
#         ranking[places[i]] += score
#         score+=1
#     i = 0
#
#     for i in range(len(models)):
#         print('model '+models[i]+' - ranking score '+str(ranking[i]))
#         i+=1
#     print('model chosen_classifier - ranking score '+str(ranking[i]))
# '''
#
# '''
# # ClassifierChooser Ranking score paper1 EXP2
# import csv
# import sys
#
# for dataset in datasets:
#
#     with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\results\\test_sets\\EXP2_{}_AUC_ROC.csv'.format(dataset)) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\results\\predictions\\EXP2_{}_RNN5_predictions.csv'.format(dataset)) as prediction_file:
#             prediction_reader = csv.reader(prediction_file, delimiter=',')
#             models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
#                       'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
#             models_rank = [0]*len(models)
#             index = 0
#             first_row = True
#             chosen_classifier_eval = 0
#             evals = [0.0]*len(models)
#             ranking = [0,0,0,0,0,0,0,0]
#             i = 0
#             for row, pred in zip(csv_reader,prediction_reader):
#                 if first_row:
#                     first_row = False
#                     continue
#                 if i == 7:
#                     ranks = evals.copy()
#                     ranks.append(chosen_classifier_eval)
#                     places = sorted(range(len(ranks)), key = lambda sub: ranks[sub])[-len(ranks):]
#                     score = 7
#                     #print('ranks: ')
#                     #print(ranks)
#                     for j in range(len(ranking)-1,-1,-1):
#                         if j<7 and ranks[places[j]] == ranks[places[j+1]]:
#                             #print('same score indexes: '+str(places[j])+' '+str(places[j+1]))
#                             score+=1
#                         #print('ranking in index '+str(places[j])+' get score '+str(score))
#                         ranking[places[j]] += score
#                         score-=1
#                     i = 0
#
#                 evals[index%len(evals)] = float(row[len(row)-2])
#                 if pred[1] == models[index%len(evals)]:
#                     chosen_classifier_eval = float(row[len(row)-2])
#
#
#                 index+=1
#                 i+=1
#
#             # for the last iteration
#             ranks = evals.copy()
#             ranks.append(chosen_classifier_eval)
#             places = sorted(range(len(ranks)), key = lambda sub: ranks[sub])[-len(ranks):]
#             score = 7
#             #print('ranks: ')
#             #print(ranks)
#             for j in range(len(ranking)-1,-1,-1):
#                 if j<7 and ranks[places[j]] == ranks[places[j+1]]:
#                     #print('same score indexes: '+str(places[j])+' '+str(places[j+1]))
#                     score+=1
#                 #print('ranking in index '+str(places[j])+' get score '+str(score))
#                 ranking[places[j]] += score
#                 score-=1
#             i = 0
#             print()
#             print('dataset: '+dataset)
#             for i in range(len(models)):
#                 print('model '+models[i]+' - ranking score '+str(ranking[i]/sum(ranking)))
#                 i+=1
#             print('model chosen_classifier - ranking score '+str(ranking[i]/sum(ranking)))
#
# '''
#
# ## ClassifierChooser real score in Graph
# # import csv
# # import matplotlib.pyplot as plt
#
# # with open('C:\\Users\\Rotem\\featuresFromDistributions\\MLWithDynamicEnv\\meta_datasets\\27_2_2021_abalone_classifier3StepsBack.csv') as csv_file:
# #    csv_reader = csv.reader(csv_file, delimiter=',')
# #    models = ['random_forest', 'logistic_regression', 'ada_boost',
# #     # '50_neighbors', '100_neighbors', '200_neighbors',
# #     'desicition_tree', 'mlp', 'gaussian_nb', 'gaussian_process']
# #    models_eval = [0.0]*len(models)
# #    index = 0
# #    first_row = True
# #    sum_of_chosen_classifier = 0
# #    for row in csv_reader:
# #        if first_row:
# #            first_row = False
# #            continue
# #        print("line "+ str(index) +" "+ models[index%len(models_eval)] +" "+ str(row[len(row)-1]))
# #        models_eval[index%len(models_eval)]+=float(row[len(row)-1])
# #        #print('row[2] '+ row[2])
# #        if row[len(row) - 3] == row[2]:
# #            #print('True')
# #            sum_of_chosen_classifier+=float(row[len(row)-1])
# #        index+=1
#
# #    models.append('chosen classifier')
# #    models_eval.append(sum_of_chosen_classifier)
# #    print(models_eval)
# #    models_eval[:] = [x / 9 for x in models_eval]
# #    print(models_eval)
# #    print(models)
#
# #    # x-coordinates of left sides of bars
# #    left = [1, 2, 3, 4, 5, 6, 7, 8]
# #    # plotting a bar chart
# #    plt.bar(left, models_eval, tick_label=models,
# #            width=0.8, color=(['purple','purple','purple','purple','purple','purple','purple','orange']))
# #    plt.xticks(rotation=90)
#
# #    for index, value in enumerate(models_eval):
# #        plt.text(index+0.7,value, str('%.3f'%(value)))
#
# #    # naming the x-axis
# #    plt.xlabel('Models')
# #    # naming the y-axis
# #    plt.ylabel('Mean Score')
# #    # plot title
# #    plt.title('abalone_classifier3StepsBack')
# #    # function to show the plot
# #    plt.show()
#
#
# #
# # # paper1 EXP2: calc.csv AUC score for random selection
# # import csv
# # import sys
# #
# # for dataset in datasets:
# #
# #     with open(r'C:\Users\rotem\OneDrive\מסמכים\Thesis\Paper1 results\test_sets\EXP2_{}_AUC_ROC.csv'.format(dataset)) as csv_file:
# #         csv_reader = csv.reader(csv_file, delimiter=',')
# #         with open(r'C:\Users\rotem\OneDrive\מסמכים\Thesis\Paper1 results\predictions\EXP2_{}_RNN4_predictions.csv'.format(dataset)) as prediction_file:
# #             prediction_reader = csv.reader(prediction_file, delimiter=',')
# #             models = ['ada_boost', 'desicition_tree', 'gaussian_nb',
# #                       'gaussian_process', 'logistic_regression', 'mlp', 'random_forest']
# #             classifiers = []
# #             models_eval = [0.0]*len(models)
# #             index = 0
# #             first_row = True
# #             sum_of_chosen_classifier = 0
# #             random_models = []
# #             for row in csv_reader:
# #
# #               if first_row:
# #                 first_row = False
# #                 #print('row[len(row) - 2] ' + row[len(row) - 2])
# #                 continue
# #
# #               if index % len(models_eval) == 0: # choose random model
# #                   next_rand = random.randint(0,6)
# #                   random_models.append(models[next_rand])
# #
# #               models_eval[index%len(models_eval)]+=float(row[len(row)-2])
# #
# #               if models[next_rand] == models[index%len(models_eval)]:
# #                 classifiers.append(models[index%len(models_eval)])
# #                 sum_of_chosen_classifier+=float(row[len(row)-2])
# #               index+=1
# #
# #             print()
# #             i = 0
# #             for model in models:
# #                 #print('model '+model+' score '+str(models_eval[i]/9))
# #                 i+=1
# #             print('dataset: '+dataset)
# #             print('model chosen_classifier score '+str(sum_of_chosen_classifier/9))
# #             print('chosen classifiers: '+str(classifiers))
# #             print('random models: '+str((random_models)))
# #         #    min_eval = models_eval[0]
# #         #    worst_classifier = ''
# #         #    for i in range(0, len(models_eval)):
# #         #      print('model '+models[i]+' with score sum '+str(models_eval[i]))
# #         #      if (models_eval[i] < min_eval):
# #         #        min_eval = models_eval[i]
# #         #        worst_classifier = models[i]
# #         #    print('worst model: '+worst_classifier)
# #         #    print('sum_of_chosen_classifier '+str(sum_of_chosen_classifier))
# #         #    print('score '+str((sum_of_chosen_classifier-min_eval)/9))
# # sys.exit()


# # # old: paper1 EXP2: prepare test sets for next step prediction
# # import pandas as pd
# #
# #
# # for dataset in datasets:
# #     test_set = pd.read_csv(rf'C:\Users\rotem\OneDrive\מסמכים\Thesis\Paper1 results\test_sets\EXP2_{dataset}_AUC_ROC.csv')
# #     test_set['best_classifier'] = test_set['best_classifier'].shift(-7)
# #     test_set = test_set.iloc[:-7, :]
# #     test_set['best_classifier'] = test_set['best_classifier'].astype(int)
# #     file_name = fr'C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\next_step_test_sets\EXP2_{dataset}_AUC_ROC.csv'
# #     test_set.to_csv(file_name)
#
#
