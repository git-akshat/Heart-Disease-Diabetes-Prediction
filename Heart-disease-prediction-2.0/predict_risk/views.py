import csv,io
from django.shortcuts import render
from .forms import Predict_Form,Predict_Diabetes_Form
from predict_risk.data_provider import *
from accounts.models import UserProfileInfo
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.decorators import login_required,permission_required
from django.urls import reverse
from django.contrib import messages


@login_required(login_url='/')
def heart(request,pk):
    predicted = False
    predictions={}
    if request.session.has_key('user_id'):
        u_id = request.session['user_id']

    if request.method == 'POST':
        form = Predict_Form(data=request.POST)
        profile = get_object_or_404(UserProfileInfo, pk=pk)

        if form.is_valid():
            features = [[ form.cleaned_data['age'], form.cleaned_data['sex'], form.cleaned_data['cp'], form.cleaned_data['resting_bp'], form.cleaned_data['serum_cholesterol'],
            form.cleaned_data['fasting_blood_sugar'], form.cleaned_data['resting_ecg'], form.cleaned_data['max_heart_rate'], form.cleaned_data['exercise_induced_angina'],
            form.cleaned_data['st_depression'], form.cleaned_data['st_slope'], form.cleaned_data['number_of_vessels'], form.cleaned_data['thallium_scan_results']]]

            standard_scalar = GetStandardScalarForHeart()
            features = standard_scalar.transform(features)
            SVCClassifier,LogisticRegressionClassifier,NaiveBayesClassifier,DecisionTreeClassifier,KNeighborsClassifier,NeuralNetworkClassifier=GetAllClassifiersForHeart()


            predictions = {'SVC': str(SVCClassifier.predict(features)[0]),
            'LogisticRegression': str(LogisticRegressionClassifier.predict(features)[0]),
             'NaiveBayes': str(NaiveBayesClassifier.predict(features)[0]),
             'DecisionTree': str(DecisionTreeClassifier.predict(features)[0]),
             'KNN': str(KNeighborsClassifier.predict(features)[0]),
             'NeuralNetwork': str(NeuralNetworkClassifier.predict(features)[0]),
              }
            pred = form.save(commit=False)

            l=[predictions['SVC'],predictions['LogisticRegression'],predictions['NaiveBayes'],predictions['DecisionTree'],predictions['KNN'],predictions['NeuralNetwork']]
            count=l.count('1')

            result=False

            if count>=2:
                result=True
                pred.num=1
            else:
                pred.num=0

            pred.profile = profile

            pred.save()
            predicted = True

            colors={}
            if predictions['SVC']=='0':
                predictions['SVC']='Negative'
                colors['SVC']="table-success"
            elif predictions['SVC']=='1':
                predictions['SVC']='Positive'
                colors['SVC']="table-danger"

            if predictions['LogisticRegression']=='0':
                predictions['LogisticRegression']='Negative'
                colors['LR']="table-success"
            else:
                predictions['LogisticRegression']='Positive'
                colors['LR']="table-danger"

            if predictions['NaiveBayes']=='0':
                predictions['NaiveBayes']='Negative'
                colors['NB']="table-success"
            else:
                predictions['NaiveBayes']='Positive'
                colors['NB']="table-danger"

            if predictions['DecisionTree']=='0':
                predictions['DecisionTree']='Negative'
                colors['DT']="table-success"
            else:
                predictions['DecisionTree']='Positive'
                colors['DT']="table-danger"

            if predictions['KNN']=='0':
                predictions['KNN']='Negative'
                colors['KNN']="table-success"
            else:
                predictions['KNN']='Positive'
                colors['KNN']="table-danger"

            if predictions['NeuralNetwork']=='0':
                predictions['NeuralNetwork']='Negative'
                colors['NeuralNetwork']="table-success"
            else:
                predictions['NeuralNetwork']='Positive'
                colors['NeuralNetwork']="table-danger"

    if predicted:
        return render(request, 'predict.html',
                      {'form': form,'predicted': predicted,'user_id':u_id,'predictions':predictions,'result':result,'colors':colors})

    else:
        form = Predict_Form()

        return render(request, 'predict.html',
                      {'form': form,'predicted': predicted,'user_id':u_id,'predictions':predictions})


def description(request,pk):
    return render(request, 'accounts/description.html', {})


@login_required(login_url='/')
def diabetes(request,pk):
    predicted = False
    predictions={}
    if request.session.has_key('user_id'):
        u_id = request.session['user_id']

    if request.method == 'POST':
        form = Predict_Diabetes_Form(data=request.POST)
        profile = get_object_or_404(UserProfileInfo,pk=pk)

        if form.is_valid():
            features = [[ form.cleaned_data['pregnancies'], form.cleaned_data['glucose'], form.cleaned_data['bloodpressure'], form.cleaned_data['skinthickness'], form.cleaned_data['insulin'],
                          form.cleaned_data['bmi'], form.cleaned_data['pedigree'], form.cleaned_data['age']]]

            standard_scalar = GetStandardScalarForDiabetes()
            features_standard = standard_scalar.transform(features)

            LinearSVCClassifier, LogisticRegressionClassifier, NaiveBayesClassifier, NeuralNetworkClassifier, KNeighborsClassifier, GetDecisionTreeClassifier= GetAllClassifiersForDiabetes()


            predictions = {'LinearSVC': str(LinearSVCClassifier.predict(features_standard)[0]),
                           'LogisticRegression': str(LogisticRegressionClassifier.predict(features_standard)[0]),
                           'NaiveBayes': str(NaiveBayesClassifier.predict(features_standard)[0]),
                           'NeuralNetwork': str(NeuralNetworkClassifier.predict(features_standard)[0]),
                           'KNeighbors': str(KNeighborsClassifier.predict(features_standard)[0]),
                           'DecisionTree': str(GetDecisionTreeClassifier.predict(features_standard)[0])
                           }
            pred = form.save(commit=False)

            l=[predictions['LinearSVC'],predictions['LogisticRegression'],predictions['NaiveBayes'],predictions['NeuralNetwork'],predictions['KNeighbors'],predictions['DecisionTree']]
            count=l.count('1')

            result=False

            if count>=2:
                result=True
                pred.num=1
            else:
                pred.num=0

            pred.profile = profile

            pred.save()
            predicted = True

            colors={}

            if predictions['LinearSVC']=='0':
                predictions['LinearSVC']='Negative'
                colors['SVC']="table-success"
            else:
                predictions['LinearSVC']='Positive'
                colors['SVC']="table-danger"

            if predictions['LogisticRegression']=='0':
                predictions['LogisticRegression']='Negative'
                colors['LR']="table-success"
            else:
                predictions['LogisticRegression']='Positive'
                colors['LR']="table-danger"

            if predictions['NaiveBayes']=='0':
                predictions['NaiveBayes']='Negative'
                colors['NB']="table-success"
            else:
                predictions['NaiveBayes']='Positive'
                colors['NB']="table-danger"

            if predictions['DecisionTree']=='0':
                predictions['DecisionTree']='Negative'
                colors['DT']="table-success"
            else:
                predictions['DecisionTree']='Positive'
                colors['DT']="table-danger"

            if predictions['KNeighbors']=='0':
                predictions['KNeighbors']='Negative'
                colors['KNeighbors']="table-success"
            else:
                predictions['KNeighbors']='Positive'
                colors['KNeighbors']="table-danger"

            if predictions['NeuralNetwork']=='0':
                predictions['NeuralNetwork']='Negative'
                colors['NeuralNetwork']="table-success"
            else:
                predictions['NeuralNetwork']='Positive'
                colors['NeuralNetwork']="table-danger"

    if predicted:
        return render(request, 'diabetes.html',
                      {'form': form,'predicted': predicted,'user_id':u_id,'predictions':predictions,'result':result,'colors':colors})

    else:
        form = Predict_Diabetes_Form()

        return render(request, 'diabetes.html',
                      {'form': form,'predicted': predicted,'user_id':u_id,'predictions':predictions})