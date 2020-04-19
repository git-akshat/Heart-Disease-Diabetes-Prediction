import os
import pickle
from sklearn.externals import joblib

config = {
    'heart': {
        'SVC': 'production/svc_model.pkl',
        'LogisticRegression': 'production/Logistic_regression_model.pkl',
        'NaiveBayes': 'production/naive_bayes_model.pkl',
        'DecisionTree': 'production/decision_tree_model.pkl',
        'scalar_file': 'production/standard_scalar.pkl',
        'KNN': 'production/knn.pkl',
        'NeuralNetwork': 'production/neural_network.pkl',
    }}

dir = os.path.dirname(__file__)


def GetJobLibFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return joblib.load(os.path.join(dir, filepath))
    return None


def GetPickleFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return pickle.load(open(os.path.join(dir, filepath), "rb"))
    return None


def GetStandardScalarForHeart():
    return GetPickleFile(config['heart']['scalar_file'])


def GetAllClassifiersForHeart():
    return (GetSVCClassifierForHeart(), GetLogisticRegressionClassifierForHeart(),
            GetNaiveBayesClassifierForHeart(), GetDecisionTreeClassifierForHeart(),
            GetKNeighborsClassifierForHeart(), GetNeuralNetworkClassifierForHeart())


def GetSVCClassifierForHeart():
    return GetJobLibFile(config['heart']['SVC'])


def GetLogisticRegressionClassifierForHeart():
    return GetJobLibFile(config['heart']['LogisticRegression'])


def GetNaiveBayesClassifierForHeart():
    return GetJobLibFile(config['heart']['NaiveBayes'])


def GetDecisionTreeClassifierForHeart():
    return GetJobLibFile(config['heart']['DecisionTree'])


def GetKNeighborsClassifierForHeart():
    return GetJobLibFile(config['heart']['KNN'])


def GetNeuralNetworkClassifierForHeart():
    return GetJobLibFile(config['heart']['NeuralNetwork'])
