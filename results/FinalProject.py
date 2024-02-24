import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier, LassoCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve

import random
import os


def calcualte_score(y_true, prediction):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    score = 32.7 * tp - 6.05 * fp
    return score


class practice:

    def features_selection(self, model, file_path):
        # file_path = r"C:\Users\rotem\Downloads\פרוייקט מסכם\reviews_training_with_buyer.csv"
        self.dataset = pd.read_csv(file_path)

        # self.dataset[self.dataset.columns[1]] = 0 * len(self.dataset)
        self.data = self.dataset[self.dataset.columns[:-1]]
        self.label = self.dataset[self.dataset.columns[-1]]

        print(self.dataset.columns[-1])
        # print(self.dataset.columns.names)
        # print(self.dataset.columns)
        if model is 'logistic_regression':
            selector = SelectFromModel(estimator=LogisticRegression()).fit(self.data, self.label)
        else:
            clf = ExtraTreesClassifier(n_estimators=50)
            selector = SelectFromModel(estimator=clf).fit(self.data, self.label)
        # print(selector.estimator_.coef_)
        # print('threshold: '+str(selector.threshold_))
        selected_features = self.data.columns[selector.get_support()]
        df = pd.DataFrame(np.squeeze(selector.transform(self.data)), columns=selected_features)
        df['ID'] = self.data['ID']
        df['rating'] = self.label
        # df.to_csv(r"C:\Users\rotem\Downloads\פרוייקט מסכם\reviews_table_features_selection_ET.csv")
        print('hi')
        return df

    def predict_review_type(self, train_df, test_df, output_name):
        target_col_name = 'rating'
        # train_set = pd.read_csv(train_path)
        # test_set = pd.read_csv(test_path)
        train_set = train_df
        test_set = test_df

        numeric_cols = train_set._get_numeric_data().columns
        categorical_cols = list(set(train_set.columns) - set(numeric_cols))
        for cat_col in categorical_cols:
            train_set[cat_col] = train_set[cat_col].astype('category')
        train_set[categorical_cols] = train_set[categorical_cols].apply(lambda x: x.cat.codes)

        numeric_cols = test_set._get_numeric_data().columns
        categorical_cols = list(set(test_set.columns) - set(numeric_cols))
        for cat_col in categorical_cols:
            test_set[cat_col] = test_set[cat_col].astype('category')
        test_set[categorical_cols] = test_set[categorical_cols].apply(lambda x: x.cat.codes)

        train_set_data = train_set.drop([target_col_name], axis=1)
        label = train_set[target_col_name]

        self.X_train = train_set_data
        self.y_train = label
        self.create_model()
        self.model.fit(train_set_data, label)
        prediction = self.model.predict(test_set)
        prediction_df = pd.DataFrame(prediction, columns=['REVIEW_TYPE'])
        test_set['REVIEW_TYPE'] = prediction_df['REVIEW_TYPE']
        test_set.to_csv(f"C:\\Users\\rotem\\Downloads\\פרוייקט מסכם\\{output_name}.csv")

    def start_experiment(self, random_seed, df):
        self.target_col_name = 'BUYER_FLAG'
        self.dataset = df
        self.handle_categorical_data()
        self.data = self.dataset.drop([self.target_col_name], axis=1)

        self.label = self.dataset[self.target_col_name]
        self.label_encode()
        self.data_split(train_size=0.7, test_size=0.2)
        self.create_model(random_seed=random_seed)
        self.model.fit(self.X_train, self.y_train)

        # prediction = self.model.predict(self.X_test)
        # score_train = calcualte_score(self.y_test, prediction)
        preds = self.model.predict_proba(self.X_test)[:, 1]
        optimal_threshold = self.calculate_optimal_teshold(preds)
        predict_with_treshhold = np.where(preds > optimal_threshold, 1, 0)
        score_treshold = calcualte_score(self.y_test, predict_with_treshhold)
        max_th = self.choose_best_treshold_by_defined_cost_function(preds)
        print(f'Finished train model  with th {score_treshold}')
        print(optimal_threshold)
        self.model.fit(self.X_holdout, self.y_holdout)
        preds_holdout = self.model.predict_proba(self.X_holdout)[:, 1]
        predict_holdout_with_roc_treshhold = np.where(preds_holdout > optimal_threshold, 1, 0)
        predict_holdout_with_max_treshhold = np.where(preds_holdout > max_th, 1, 0)
        score_roc_treshhold = calcualte_score(self.y_holdout, predict_holdout_with_roc_treshhold)
        score_max_treshhold = calcualte_score(self.y_holdout, predict_holdout_with_max_treshhold)
        print(f'roc th model score  {score_roc_treshhold}')
        print(f'max th model score  {score_max_treshhold}')

        return score_treshold

    def choose_best_treshold_by_defined_cost_function(self, preds):
        fpr, tpr, thresholds = roc_curve(self.y_test, preds)
        max = 0
        max_th = 0
        for t in thresholds:
            # for tt in range(100):
            #     t=tt*0.01
            predict_with_treshhold = np.where(preds > t, 1, 0)
            score_treshold = calcualte_score(self.y_test, predict_with_treshhold)
            if score_treshold > max:
                max = score_treshold
                max_th = t
        print("max: ", max)
        return max_th

    def calculate_optimal_teshold(self, preds):
        fpr, tpr, thresholds = roc_curve(self.y_test, preds)
        optimal_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_index]
        print(f'Optimal Treshold: {optimal_threshold}')
        return optimal_threshold

    def handle_categorical_data(self):
        self.numeric_cols = self.dataset._get_numeric_data().columns
        self.categorical_cols = list(set(self.dataset.columns) - set(self.numeric_cols))
        for cat_col in self.categorical_cols:
            self.dataset[cat_col] = self.dataset[cat_col].astype('category')
        self.dataset[self.categorical_cols] = self.dataset[self.categorical_cols].apply(lambda x: x.cat.codes)

    def label_encode(self):
        unique_vals = self.label.unique()
        self.len_unique_vals = len(unique_vals)
        self.label = self.label.map({unique_vals[0]: 0, unique_vals[1]: 1})

    def data_split(self, train_size, test_size):
        self.X_train = self.data[:int(train_size * len(self.data))]
        self.y_train = self.label[:int(train_size * len(self.data))]
        self.X_test = self.data[int(train_size * len(self.data)):int((test_size + train_size) * len(self.data))]
        self.y_test = self.label[int(train_size * len(self.data)):int((test_size + train_size) * len(self.data))]
        self.X_holdout = self.data[int((test_size + train_size) * len(self.data)):]
        self.y_holdout = self.label[int((test_size + train_size) * len(self.data)):]

        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_holdout, self.y_holdout


    def create_model(self):
        # self.model = AdaBoostClassifier(random_state=random_seed)
        # self.model = DecisionTreeClassifier(max_depth=5)
        crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)
        ada = AdaBoostClassifier()

        # search_grid={'n_estimators':[10,50,250,1000],'learning_rate':[0.01,0.05,0.1,0.5,1]}
        search_grid = {'n_estimators': [250], 'learning_rate': [0.05]}
        cost_func = make_scorer(calcualte_score, greater_is_better=False)
        model = GridSearchCV(estimator=ada, param_grid=search_grid,
                             scoring=cost_func, cv=crossvalidation)
        model.fit(self.X_train, self.y_train)
        print(model.best_score_)  # Highest AUC
        print(model.best_params_)  # Best tune
        self.model = AdaBoostClassifier(**model.best_params_)

        def create_model2(self, random_seed):
            rf = RandomForestClassifier()
            search_grid = {'n_estimators': [10, 50, 250, 1000]}

            self.model = GridSearchCV(estimator=rf, param_grid=search_grid, scoring='roc_auc', cv=crossvalidation)

        def create_model3(self, random_seed):
            param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                          "base_estimator__splitter": ["best", "random"],
                          'n_estimators': [10, 50, 250, 1000],
                          'learning_rate': [0.01, 0.2, 0.5]
                          }

            DTC = DecisionTreeClassifier(random_state=11, max_features="auto", max_depth=None)

            ABC = AdaBoostClassifier(base_estimator=DTC)
            self.model = GridSearchCV(ABC, param_grid=param_grid, scoring='roc_auc')

        def custom_loss(y_true, y_pred):
            # y_true  y_pred  calc.csv
            #
            # 1       1       32
            #
            # 1       0       -32
            #
            # 0       1       -6
            #
            # 0       0       6

            calc_loss1 = np.multiply(np.multiply(y_true, y_pred), 32)  # for 1 1
            calc_loss2 = np.multiply(np.maximum(np.subtract(y_true, y_pred), 0), -32)  # for 1 0
            calc_loss3 = np.multiply(np.maximum(np.subtract(y_pred, y_true), 0), -6)  # for 0 1
            calc_loss4 = np.multiply(np.logical_not(np.logical_or(y_true, y_pred)), 6)  # for 0 0
            calc_loss = calc_loss1 + calc_loss2 + calc_loss3 + calc_loss4
            return calc_loss


    def plot_correlations(self, df):
        column_names = ['FARE_L_Y1', 'FARE_L_Y2', 'FARE_L_Y3', 'FARE_L_Y4', 'FARE_L_Y5']
        dfExtended = df
        dfExtended['FareTotal'] = df[column_names].sum(axis=1)
        dfExtended['FareMean'] = df[column_names].mean(axis=1)
        column_names = ['POINTS_L_Y1', 'POINTS_L_Y2', 'POINTS_L_Y3', 'POINTS_L_Y4', 'POINTS_L_Y5']
        dfExtended['PointsTotal'] = df[column_names].sum(axis=1)
        dfExtended['PointsMean'] = df[column_names].mean(axis=1)

        dfExtended['Y1_score'] = df["FARE_L_Y1"] * df["POINTS_L_Y1"]
        dfExtended['Y2_score'] = df["FARE_L_Y2"] * df["POINTS_L_Y2"]
        dfExtended['Y3_score'] = df["FARE_L_Y3"] * df["POINTS_L_Y3"]
        dfExtended['Y4_score'] = df["FARE_L_Y4"] * df["POINTS_L_Y4"]
        dfExtended['Y5_score'] = df["FARE_L_Y5"] * df["POINTS_L_Y5"]

        column_names = ["FareTotal",
                        "FareMean", "PointsTotal", "PointsMean", "STATUS_SILVER", "LAST_DEAL", "ADVANCE_PURCHASE",
                        "SERVICE_FLAG", "RECSYS_FLAG", "BUYER_FLAG"]
        df2 = df[column_names]

        correlation_mat = df2.corr()
        sns.heatmap(correlation_mat, annot=True, fmt='.2f')

        fig = plt.gcf()  # or by other means, like plt.subplots
        figsize = fig.get_size_inches()
        fig.set_size_inches(figsize * 2.5)
        plt.show()

    def not_mandatory_work(self):
        train_df = practice().features_selection(
            model='ExtraTreesClassifier',
            file_path=r"text_training.csv")
        test_df = pd.read_csv(r"reviews_rollout.csv")
        test_df = test_df[train_df.columns[:-1]]  # without rating column
        practice().predict_review_type(train_df=train_df,
                                       test_df=test_df,
                                       output_name='reviews_rollout_table_features_selection_ET')

    def merge_files(self):
        df = pd.read_csv(r'reviews_rollout_table_features_selection_ET.csv')
        df2 = pd.read_csv(r"ffp_rollout_X.csv")
        output1 = pd.merge(df, df2,
                           on='ID',
                           how='inner')
        # output1 = output1.fillna(0)
        output1.to_csv(r"merged_reviews_rollout_table_FS_ET.csv")


# df = pd.read_csv(r"C:\Users\rotem\Downloads\פרוייקט מסכם\merged_training_table_FS_LR.csv")
# print(df["BUYER_FLAG"].value_counts())
# p = practice()
# score = p.start_experiment(1,df)

# practice().not_mandatory_work()
practice().merge_files()
# practice().custom_loss(y_true=[1, 0, 1], y_pred=[1, 1, 0])
