import math
import pandas as pd
import numpy as np
import shap
from kernel_shap import KernelShapModel
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
shap.initjs()
# Testing a RandomForest model
df = pd.read_csv('winequality-red.csv') # Load the data

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
shap_values_1 = explainer.shap_values(X_test.iloc[0])
# shap.summary_plot(shap_values_1, X_test.iloc[0:5])
shap.decision_plot(explainer.expected_value, shap_values_1, X_test.iloc[0])
p = shap.force_plot(explainer.expected_value, shap_values_1, X_test.iloc[0])

print("python shap execution time: {0}".format(time.time() - start))
instance = X_test.iloc[0:50].values

start = time.time()
coalition_depth = len(X.columns)-1
num_train_samples = X_train.values.shape[0]
weights = np.ones(num_train_samples)/num_train_samples
shap_values_2 = KernelShapModel(model.predict).run(X_train.values, weights, instance, coalition_depth, use_mp=True)
print(time.time() - start)
f = plt.figure()
f.savefig("/summary_plot1.png", bbox_inches='tight', dpi=600)
print("python shap execution time: {0}".format(time.time() - start))
print(shap_values_2)
# don't include phi0, because that's not part of the shap_values returned by shap.KernelExplainer
shap_values_2 = np.array(shap_values_2)[:, 1:]

M, N = shap_values_2.shape
max_diff = 0
for i in range(0, M):
    for j in range(0, N):
        max_diff = max(shap_values_1[i, j] - shap_values_2[i, j], max_diff)

print(max_diff)

## verify shap_values_1 == shap_values_2




