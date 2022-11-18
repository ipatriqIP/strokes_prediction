#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:58:21 2022

@author: patrickilunga
"""

# chargement de packages isuels
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# chargement des outils de Modélisation
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE   # pour equilibre la base
import pickle                              # pour sauvegarder le model


# 1. Données

# 1.1. Importation des données
url = "https://assets-datascientest.s3-eu-west-1.amazonaws.com/de/total/strokes.csv"
df = pd.read_csv(url, index_col = 0)


# Imputation

bmi_mean = df["bmi"].mean()
df["bmi"].fillna(bmi_mean, inplace = True)

# Variation cible
df['stroke'] = df['stroke'].astype('category')

# On supprime les variables jugées inutiles
df_new = df.drop(['gender', 'smoking_status', 'Residence_type'], axis = 1)

# Variable age : Jeune pour moins de 40 ans et adulte pour plus de 40 ans
age_max = df_new["age"].max()
df_new["age"] = pd.cut(x = df_new['age'], bins = [0, 40, age_max], labels = ["jeune", "adulte"])

# variable bmi : classe 1 moins de 18, classe 2 entre 18 et 25 et classe 3 plus 25
bmi_max = df_new["bmi"].max()
df_new["bmi"] = pd.cut(x = df_new['bmi'], bins = [0, 18, 25, bmi_max], labels = ['trop_mince', 'normale', 'surpoids'])

# variable avg_glucose_level : (120 et 180) classe 1 moins de 120, classe 2 entre 120 et 25 et classe 3 plus 25
glucose_max = df_new["avg_glucose_level"].max()
df_new["avg_glucose_level"] = pd.cut(x = df_new['avg_glucose_level'], 
                                     bins = [0, 120, 180, glucose_max], 
                                     labels = ['faible', 'normale', 'elevee'])
                                     
# Définition de la fonction qui va opérer :
def my_recode(var):
    if var == "Private":
        return "Private"
    elif var == "Govt_job":
        return "Govt_job"
    elif var == "Self-employed":
        return "Self-employed"
    else:
        return "No_Worked"

df_new["work_type"] = df_new["work_type"].apply(my_recode)

# Dichotonomisation des variables
df_new = df_new.join(pd.get_dummies(df_new["age"], prefix = 'age'))
df_new = df_new.join(pd.get_dummies(df_new["hypertension"], prefix = 'hypt'))
df_new = df_new.join(pd.get_dummies(df_new["heart_disease"], prefix = 'heart'))
df_new = df_new.join(pd.get_dummies(df_new["ever_married"], prefix = 'married'))
df_new = df_new.join(pd.get_dummies(df_new["work_type"], prefix = 'work'))
df_new = df_new.join(pd.get_dummies(df_new["avg_glucose_level"], prefix = 'glucos'))
df_new = df_new.join(pd.get_dummies(df_new["bmi"], prefix = 'bmi'))

# Ecrire un nouveau fichier
df_new.to_csv('strokes_dich.csv')

# Separtaion des features X et de la variable cible Y
X = df_new.iloc[:, 8:25]
y = df_new["stroke"]

# Echantillonage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1986)

# Equilibrage des données : SMOTE
columns = X_train.columns
# print(columns)


# On instatncie l'algorithme avec les paramètres par défauts
oversample = SMOTE()
X_SMOTEd, y_SMOTEd = oversample.fit_resample(X_train, y_train)

X_SMOTEd = pd.DataFrame(data = X_SMOTEd, columns = columns)
y_SMOTEd = pd.DataFrame(data = y_SMOTEd, columns = ['stroke'])

# Entrainement du modèle avec les paramètres ci-dessus
model_Final = LogisticRegression(C = 0.01, 
                                 fit_intercept = True, 
                                 intercept_scaling = 1, 
                                 max_iter = 100, 
                                 penalty = 'l2', 
                                 solver = 'lbfgs',
                                 tol = 0.0001)

model_Final.fit(X_SMOTEd.values, y_SMOTEd.values)


# Enregistrement du model entrainé
filename = 'stroke_model.pkl'
pickle.dump(model_Final, open(filename, 'wb'))


# Prédiction
y_pred = model_Final.predict(X_test)

