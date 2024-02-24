from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from config import Config

class Classifier:
    def __init__(self, classifier_name = None):
        self.classifier_name = classifier_name        
        if classifier_name == 'random_forest':
            self.classifier = RandomForestClassifier()
        elif classifier_name == 'ada_boost':
            self.classifier = AdaBoostClassifier()
        elif classifier_name == 'gaussian_process': # not working
            self.classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
        elif classifier_name == 'gaussian_nb':
            self.classifier = GaussianNB()
        elif classifier_name == '50_neighbors':
            self.classifier = KNeighborsClassifier(n_neighbors = 50)
        elif classifier_name == '100_neighbors':
            self.classifier = KNeighborsClassifier(n_neighbors = 100)
        elif classifier_name == '200_neighbors':
            self.classifier = KNeighborsClassifier(n_neighbors = 200)
        elif classifier_name=='autoML':
            pass
        elif classifier_name == 'desicition_tree':
            self.classifier = tree.DecisionTreeClassifier()
        elif classifier_name == 'logistic_regression':
            self.classifier = LogisticRegression()
        elif classifier_name == 'mlp':
            self.classifier = MLPClassifier(random_state=1)
        elif classifier_name == 'quadratic_discriminant_analysis':
            self.classifier = QuadraticDiscriminantAnalysis()

        else:
            self.classifier = LogisticRegression()

    def get_classifier_name(self):
        return self.classifier_name

    def get_classifier(self):
        return self.classifier