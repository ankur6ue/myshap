import pandas as pd
import numpy as np
import shap
from kernel_shap import KernelShapModel
from kernel_shap_distributed_ray import KernelShapModelDistributedRay
from connectors.s3 import read_from_s3, set_session_creds
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import ray
import logging
import argparse

logger = logging.getLogger(__name__)

#shap.initjs()
# the _impl versions can be called directly
def etl_impl(name, bucket):
    if bucket is None: # read local csv file
        df = pd.read_csv(name)  # Load the data
    else:
        df = read_from_s3(bucket, name)
    return df


@ray.remote
def etl(name, bucket=None):
    return etl_impl(name, bucket)


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


@ray.remote
def create_model(df):
    return create_model_impl(df)

@ray.remote
def call_default_shap(explainer, instance_chunks):
    shap_values = explainer.shap_values(instance_chunks)
    phi0 = explainer.expected_value
    shap_values_ = np.insert(shap_values, 0, np.ones(shap_values.shape[0]) * phi0, axis=1)
    return shap_values_


def run_default_shap_impl(model_state, data_to_explain, num_workers=8):
    model = model_state['model']
    X_train = model_state['X_train']
    explainer = shap.KernelExplainer(model.predict, X_train)
    if isinstance(data_to_explain, pd.DataFrame):
        data_to_explain = data_to_explain.values
    instance_chunks = np.array_split(data_to_explain, min(num_workers, len(data_to_explain)), axis=0)
    num_splits = len(instance_chunks)

    futures = []
    for i in range(0, num_splits):
        print("Submitting:", i)
        future = call_default_shap.remote(explainer, instance_chunks[i])
        futures.append(future)

    shap_vals = []
    for future in futures:
        result = ray.get(future)
        for s in result:
            shap_vals.append(s)
    return np.array(shap_vals)


@ray.remote
def run_default_shap(model_state, data_to_explain):
    start = time.time()
    shap_values = run_default_shap_impl(model_state, data_to_explain)
    end = time.time()
    # logger = prefect.context.get("logger")
    logger = logging.getLogger(__name__)
    logger.warning("default shap execution time: {0}".format(end - start))
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


@ray.remote
def run_my_shap(model_state, data_to_explain):
    start = time.time()
    shap_values = run_my_shap_impl(model_state, data_to_explain)
    end = time.time()
    # logger = prefect.context.get("logger")
    logger = logging.getLogger(__name__)
    logger.warning("my shap execution time: {0}".format(end - start))
    return {'shap_values': shap_values, 'exec_time': end - start}


def run_distributed_shap_impl(model_state, data_to_explain):
    model = model_state['model']
    X_train = model_state['X_train']
    coalition_depth = len(X_train.columns) - 1
    num_train_samples = X_train.values.shape[0]
    weights = np.ones(num_train_samples) / num_train_samples
    kernelShapModel = KernelShapModelDistributedRay(model.predict)
    shap_values = kernelShapModel.run(X_train.values, weights, data_to_explain.values,
                                      coalition_depth, num_workers=12)
    return shap_values


@ray.remote
def run_distributed_shap(model_state, data_to_explain):
    start = time.time()
    shap_values = run_distributed_shap_impl(model_state, data_to_explain)
    end = time.time()
    # logger = prefect.context.get("logger")
    # According to https://stackoverflow.com/questions/55272066/how-can-i-use-the-python-logging-in-ray
    # you should create a new logger inside of the worker because the worker
    # runs on a different Python process. If you try to use a logger that you created
    # outside of the worker within the worker, then Ray will try to pickle the logger and
    # send it to the worker process, and Python loggers typically do not behave correctly
    # when pickled and unpickled.
    logger = logging.getLogger(__name__)
    logger.warning("my shap (distributed) execution time: {0}".format(end - start))
    return {'shap_values': shap_values, 'exec_time': end - start}


def get_data_to_explain_impl(model_state, start_index, end_index):
    data = model_state.get('X_test')
    return data.iloc[start_index: end_index]


@ray.remote
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


@ray.remote
def compare_results(default_shap, my_shap):
    default_shap_vals = default_shap['shap_values']
    my_shap_vals = my_shap['shap_values']
    return compare_results_impl(default_shap_vals, my_shap_vals)


def test():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    df = etl_impl(curr_dir + '/../winequality-red.csv')
    # etl_impl('winequality-red.csv', 'shap-data')
    model_state = create_model_impl(df)
    data_to_explain = get_data_to_explain_impl(model_state, 0, 240)
    my_shap_distributed = run_distributed_shap_impl(model_state, data_to_explain)
    match = True
    # default_shap = run_default_shap_impl(model_state, data_to_explain)
    # match = compare_results_impl(default_shap, my_shap_distributed)
    if match is True:
        print("Results match!")
    else:
        print("Results don't match!")
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='kernel_shap_test_ray', usage='%(prog)s [options]')
    parser.add_argument('--local', type=bool, default=True, help='if True, Ray creates a new cluster, otherwise '
                                                                   'Ray attempts to connect to an existing cluster')

    parser.add_argument('--use_local_mode', type=bool, default=False, help='if True, Ray is run in local mode. '
                                                                             'Doing so forces Ray to use a single '
                                                                             'process, making it easier to debug')
    parser.add_argument('--use_s3', type=bool, default=False,
                        help='if True, local copy of winequality-red.csv is used. Otherwise, copy stored in '
                             'shap-data S3 bucket is used. You must have a iamroles.txt file in your '
                             'root directory specifying the IAM role to be assumed to access the S3 bucket')

    parser.add_argument('--num_rows', type=int, default=10,
                        help='Number of rows of the input data for which SHAP values are calculated. Larger '
                             'this number, longer the computation time')

    args = parser.parse_args()

    # test()
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # When running locally, i.e., not connecting to a existing cluster, we get the STS credentials here, because they'll
    # not be provided in the environment variables during docker run.
    if args.local and args.use_s3:
        file_loc = os.path.join(curr_dir, '../iamroles.txt')
        if os.path.isfile(file_loc):
            with open(file_loc, "r") as file:
                role = file.readline()
                print('Creating temporary credentials using IAM role {0}'.format(role))
                set_session_creds(role)

    if args.local:
        if args.use_local_mode:
            ray.init(local_mode=True)
        else:
            # Will use 6 CPUs
            ray.init(num_cpus=6)
    else:
        try:
            ray.init(address='auto', _redis_password="password", log_to_driver=True, logging_level=logging.DEBUG)
            # ray.init()
        except:
            # in older version of ray, redis_password doesn't have the leading underscore
            ray.init(address='auto', redis_password="password", log_to_driver=True, logging_level=logging.DEBUG)

    if args.use_s3:
        # if loading data from s3 bucket, second argument is the S3 bucket and csv filename is used as the object key
        df = etl.remote('winequality-red.csv', 'shap-data')
    else:
        # load data from CSV and get a dataframe. If file is read from disk, full path is provided and second argument is
        # left empty
        print('reading {0} from local file'.format('winequality-red.csv'))
        df = etl.remote(curr_dir + '/../winequality-red.csv')


     # Train randomforest model
    model_state = create_model.remote(df)
    # get data to explain: returns test dataframe rows from start to end index
    data_to_explain = get_data_to_explain.remote(model_state, 0, args.num_rows)
    # Run my serial (non-distributed) implementation of shap
    my_shap = run_my_shap.remote(model_state, data_to_explain)
    # Run the distributed version
    my_shap_distributed = run_distributed_shap.remote(model_state, data_to_explain)
    # Run the default shap python library implementation
    default_shap = run_default_shap.remote(model_state, data_to_explain)
    # compare results of the distributed version with the default python implementation
    match = compare_results.remote(default_shap, my_shap_distributed)

    start = time.time()
    print("Starting DAG execution")
    res = ray.get(match)
    print("Finished DAG execution")
    end = time.time()
    if res is True:
        print("Results match!")
    else:
        print("Results don't match!")
    print("Total DAG execution time: {0}".format(end - start))
    print('done')


