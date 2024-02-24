import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from time import time

class featureSelection:
    def selectFeaturesImportance(self, dataset_name):


        filename = '../results/test_sets/EXP2_{}_AUC_ROC.csv'.format(dataset_name)
        self.dataset = pd.read_csv(filename)

        #print(self.dataset.columns[0])
        #print(self.dataset.columns[1])
        #print(self.dataset.columns[2])
        self.dataset[self.dataset.columns[1]] = 0*len(self.dataset)
        self.data = self.dataset[self.dataset.columns[:-1]]
        self.label = self.dataset[self.dataset.columns[-1]]

        #print(self.label)
        #print(self.dataset.columns.names)
        #print(self.dataset.columns)

        lasso = LassoCV().fit(self.data, self.label)
        importance = np.abs(lasso.coef_)
        feature_names = np.array(self.dataset.columns[:-1])
        self.feature_names = feature_names
        #plt.bar(height=importance, x=feature_names)
        #plt.title("Feature importances via coefficients")
        #plt.show()

        sorted_indices = np.argsort(importance)
        sorted_indices = sorted_indices[::-1]
        #print(sorted_indices)

        names = [feature_names[i] for i in sorted_indices[:6]]
        features_selection = [importance[i] for i in sorted_indices[:6]]

        #plt.bar(height=features_selection, x=names)
        #plt.title("Feature importances via coefficients")
        #plt.show()

        y_pos = range(len(names))
        plt.bar(y_pos, features_selection)
        # Rotation of the bars names
        #for name in names:
        #    print(name[:(len(name)/2)])
        #    print(name[(len(name)/2):])
        names = [name[:int(len(name)/3)]+'\n'+name[int(len(name)/3):int(2*len(name)/3)]+'\n'+name[int(2*len(name)/3):] if len(name)>10 else name for name in names]
        plt.xticks(y_pos, names, fontsize = 7)
        #plt.show()
        #plt.savefig('../results/features_importance/{}.png'.format(dataset_name))

        #threshold = np.sort(importance)[-6] + 0.01

        #tic = time()
        #sfm = SelectFromModel(lasso, threshold=threshold).fit(self.data, self.label)
        #toc = time()
        #print("Features selected by SelectFromModel: "
        #      f"{feature_names[sfm.get_support()]}")
        #print(f"Done in {toc - tic:.3f}s")
        return importance

