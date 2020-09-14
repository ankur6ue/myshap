import math
import pandas as pd
import numpy as np
import shap
from kernel_shap import KernelShapModel
from kernel_shap_distributed import KernelShapModelDistributed
from kernel_shap_distributed_no_dask import KernelShapModelDistributed_NoDask
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from dask.distributed import Client, progress
from prefect import task, Flow, Parameter
import prefect
from prefect.engine.executors import DaskExecutor

#shap.initjs()
# the _impl versions can be called directly, while the ones without can only be called by Prefect
def etl_impl(name):
    df = pd.read_csv(name)  # Load the data
    return df


@task(name="etl")
def etl(name):
    return etl_impl(name)


def create_model_impl(df):
    # Testing a RandomForest model
    # The target variable is 'quality'.
    Y = df['quality']
    X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
    #X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides']]
    # Split the data into train and test data:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    # Build the model with the random forest regression algorithm:
    model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    model.fit(X_train, Y_train)
    return {'model':model, 'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}


@task(name="create_model")
def create_model(df):
    return create_model_impl(df)


def run_default_shap_impl(model_state, data_to_explain):
    model = model_state['model']
    X_train = model_state['X_train']
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(data_to_explain)
    phi0 = explainer.expected_value
    shap_values_ = np.insert(shap_values, 0, np.ones(shap_values.shape[0]) * phi0, axis=1)
    return shap_values_


@task(name="run_default_shap")
def run_default_shap(model_state, data_to_explain):
    start = time.time()
    shap_values = run_default_shap_impl(model_state, data_to_explain)
    end = time.time()
    logger = prefect.context.get("logger")
    logger.info("default shap execution time: {0}".format(end - start))
    return {'shap_values': shap_values, 'exec_time': end - start}


def run_my_shap_impl(model_state, data_to_explain):
    model = model_state['model']
    X_train = model_state['X_train']
    coalition_depth = len(X_train.columns) - 1
    num_train_samples = X_train.values.shape[0]
    weights = np.ones(num_train_samples) / num_train_samples
    kernelShapModel = KernelShapModel(model.predict)
    shap_values = kernelShapModel.run(X_train.values, weights, data_to_explain.values,
                                      coalition_depth)
    return shap_values


@task(name="run_my_shap")
def run_my_shap(model_state, data_to_explain):
    start = time.time()
    shap_values = run_my_shap_impl(model_state, data_to_explain)
    end = time.time()
    logger = prefect.context.get("logger")
    logger.info("my shap execution time: {0}".format(end - start))
    return {'shap_values': shap_values, 'exec_time': end - start}


def run_distributed_shap_impl(model_state, data_to_explain):
    model = model_state['model']
    X_train = model_state['X_train']
    coalition_depth = len(X_train.columns) - 1
    num_train_samples = X_train.values.shape[0]
    weights = np.ones(num_train_samples) / num_train_samples
    kernelShapModel = KernelShapModelDistributed_NoDask(model.predict)
    shap_values = kernelShapModel.run(X_train.values, weights, data_to_explain.values,
                                      coalition_depth, num_cpus=5)
    return shap_values


@task(name="run_distributed_shap")
def run_distributed_shap(model_state, data_to_explain):
    start = time.time()
    shap_values = run_distributed_shap_impl(model_state, data_to_explain)
    end = time.time()
    logger = prefect.context.get("logger")
    logger.info("my shap (distributed) execution time: {0}".format(end - start))
    return {'shap_values': shap_values, 'exec_time': end - start}


def get_data_to_explain_impl(model_state, start_index, end_index):
    data = model_state.get('X_test')
    return data.iloc[start_index: end_index]


@task(name="get_data_to_explain")
def get_data_to_explain(model_state, start_index, end_index):
    return get_data_to_explain_impl(model_state, start_index, end_index)


def compare_results_impl(default_shap_vals, my_shap_vals):

    my_shap_vals = np.array(my_shap_vals)

    M, N = my_shap_vals.shape
    max_diff = 0
    for i in range(0, M):
        for j in range(0, N):
            max_diff = max(my_shap_vals[i, j] - default_shap_vals[i, j], max_diff)

    return not(max_diff > 1e-5)


@task(name='compare_results')
def compare_results(default_shap, my_shap):
    default_shap_vals = default_shap['shap_values']
    my_shap_vals = my_shap['shap_values']
    return compare_results_impl(default_shap_vals, my_shap_vals)


## verify shap_values_1 == shap_values_2


def test(client):
    df = etl_impl('winequality-red.csv')
    model_state = create_model_impl(df)
    data_to_explain = get_data_to_explain_impl(model_state, 0, 5)
    my_shap_distributed = run_distributed_shap_impl(model_state, data_to_explain)

    default_shap = run_default_shap_impl(model_state, data_to_explain)
    match = compare_results_impl(default_shap, my_shap_distributed)
    if match is True:
        print("Results match!")
    else:
        print("Results don't match!")
    print('done')


distributed = True
if __name__ == '__main__':
    client = Client(threads_per_worker=10, n_workers=1)
    cluster = client.cluster
    serv_address = cluster.scheduler.address
    # test(client)

    with Flow("shap pipeline") as flow:
        name = Parameter('name')
        # load data from CSV and get a dataframe
        df = etl(name)
        # Train randomforest model
        model_state = create_model(df)
        # get data to explain: returns test dataframe rows from start to end index
        data_to_explain = get_data_to_explain(model_state, 0, 5)
        # Run my serial (non-distributed) implementation of shap
        my_shap = run_my_shap(model_state, data_to_explain)
        # Run the distributed version
        my_shap_distributed = run_distributed_shap(model_state, data_to_explain)
        # Run the default shap python library implementation
        default_shap = run_default_shap(model_state, data_to_explain)
        # compare results of the distributed version with the default python implementation
        match = compare_results(default_shap, my_shap_distributed)

    # flow.visualize()
    executor = DaskExecutor(address=serv_address)
    start = time.time()
    state = flow.run(executor=executor, name='winequality-red.csv')
    end = time.time()
    print("Total flow execution time: {0}".format(end - start))
    task_ref = flow.get_tasks(name='compare_results')[0]
    res = state.result[task_ref].result
    if res is True:
        print("Results match!")
    else:
        print("Results don't match!")




