import math
import pandas as pd
import numpy as np
from more_itertools import distinct_permutations
from scipy.optimize import minimize
import shap
from kernel_shap import KernelShapModel
import time

# shap_values1 = KernelShapModel(neural_net.predict_proba).run(x[:50], np.array(x_explain).reshape(1,2), 1)

df = pd.read_csv('winequality-red.csv') # Load the data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
# The target variable is 'quality'.
Y = df['quality']
X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
#X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides']]
# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Build the model with the random forest regression algorithm:
model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, Y_train)

explainer = shap.KernelExplainer(model.predict, X_train)
start = time.time()
shap_values_1 = explainer.shap_values(X_train.iloc[0:5])
print("python shap execution time: {0}".format(time.time() - start))
instance = X_train.iloc[0:5].values

start = time.time()
coalition_depth = len(X.columns)-1
shap_values_2 = KernelShapModel(model.predict).run(X_train.values, instance, coalition_depth)
print("python shap execution time: {0}".format(time.time() - start))
print(shap_values_2)

shap_values_2 = np.array(shap_values_2)[:, 1:]
# don't include phi0, because that's not part of the shap_values returned by shap.KernelExplainer
M, N = shap_values_2.shape
max_diff = 0
for i in range(0, M):
    for j in range(0, N):
        max_diff = max(shap_values_1[i, j] - shap_values_2[i, j], max_diff)

print(max_diff)

## verify shap_values_1 == shap_values_2




