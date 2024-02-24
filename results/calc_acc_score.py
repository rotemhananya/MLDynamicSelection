#Autoencoders acc score comparing to GT
import pandas as pd
from sklearn.metrics import accuracy_score


datasets = [
                'CBF_edited',
                'ChlorineConcentration_edited',
                'GesturePebbleZ1_edited',
                'GesturePebbleZ2_edited',
                'InsectEPGRegularTrain_mixed_edited',
                'MixedShapesRegularTrain_mixed_edited'
                ]
experiments = [
                'AutoEnc1',
                'AutoEnc2',
                'AutoEnc3',
                'AutoEnc4'
]
dict = {}
for dataset in datasets:
    autoencoders_scores = {}
    for experiment in experiments:
        dataset_GT = pd.read_csv(
            rf"C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\datasets\3_classes\{dataset}_Source.csv")
        dataset_autoencoder = pd.read_csv(
            rf"C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\datasets\3_classes\EXP2_{dataset}_{experiment}.csv")
        labels_GT = dataset_GT.iloc[:,-1:]
        # normalize values
        labels_GT_series = labels_GT.squeeze()
        unique_vals_GT = labels_GT_series.unique()
        labels_GT_series = labels_GT_series.map({unique_vals_GT[0]: 0, unique_vals_GT[1]: 1, unique_vals_GT[2]: 2})

        labels_autoencoder = dataset_autoencoder.iloc[:,-1:]
        labels_autoencoder_series = labels_autoencoder.squeeze()
        score = accuracy_score(labels_GT_series, labels_autoencoder_series)
        autoencoders_scores[experiment] = [score]

    df = pd.DataFrame.from_dict(autoencoders_scores)
    df.to_csv(fr"C:\Users\rotem\PycharmProjects\MLWithDynamicEnv\results\2nd_experiment\Autoencoders\{dataset}_autoencoders_scores.csv")
