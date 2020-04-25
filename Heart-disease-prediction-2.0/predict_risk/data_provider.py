import os
import pickle
from sklearn.externals import joblib

config = {
    'heart': {
        'SVC': 'production/svc_model.pkl',
        'LogisticRegression': 'production/Logistic_regression_model.pkl',
        'NaiveBayes': 'production/naive_bayes_model.pkl',
        'DecisionTree':'production/decision_tree_model.pkl',
        'scalar_file': 'production/standard_scalar.pkl',
        'KNN': 'production/knn.pkl',
        'NeuralNetwork': 'production/neural_network.pkl',
    },
    'diabetes': {
        'scalar_file': 'production/diabetes/standard_scalar.pkl',
        'LinearSVC': 'production/diabetes/LogisticRegression.pkl',
        'LogisticRegression': 'production/diabetes/LogisticRegression.pkl',
        'NaiveBayes': 'production/diabetes/NaiveBayes.pkl',
        'KNeighbors': 'production/diabetes/diabetes_knn.pkl',
        'NeuralNetwork': 'production/diabetes/NN.pkl',
        'DecisionTree' : 'production/diabetes/diabetes_decision_tree_model.pkl'
    }
    }

dir = os.path.dirname(__file__)

def GetJobLibFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return joblib.load(os.path.join(dir, filepath))
    return None

def GetPickleFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return pickle.load( open(os.path.join(dir, filepath), "rb" ) )
    return None

def GetStandardScalarForHeart():
    return GetPickleFile(config['heart']['scalar_file'])

def GetAllClassifiersForHeart():
    return (GetSVCClassifierForHeart(),GetLogisticRegressionClassifierForHeart(),GetNaiveBayesClassifierForHeart(),GetDecisionTreeClassifierForHeart(),GetKNeighborsClassifierForHeart(),GetNeuralNetworkClassifierForHeart())

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

def GetAllClassifiersForDiabetes():
    return (GetLinearSVCClassifierForDiabetes(), GetLogisticRegressionClassifierForDiabetes(), GetNaiveBayesClassifierForDiabetes(), GetNeuralNetworkClassifierForDiabetes(),GetKNeighborsClassifierForDiabetes(), GetDecisionTreeClassifierForDiabetes())

def GetStandardScalarForDiabetes():
    return GetPickleFile(config['diabetes']['scalar_file'])

def GetLinearSVCClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['LinearSVC'])

def GetLogisticRegressionClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['LogisticRegression'])

def GetNaiveBayesClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['NaiveBayes'])

def GetKNeighborsClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['KNeighbors'])

def GetNeuralNetworkClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['NeuralNetwork'])

def GetDecisionTreeClassifierForDiabetes():
    return GetJobLibFile(config['diabetes']['DecisionTree'])
