import math
import numpy as np
from more_itertools import distinct_permutations
from sklearn.cluster import KMeans
import concurrent.futures
import os
import types


class KernelShapModel:

    def __init__(self, predictor):
        assert callable(predictor), \
            "predictor must be a callable"
        self.predictor = predictor
        self.coalitions = None

    def run(self, data: np.ndarray, weights: np.ndarray, new_data: np.ndarray, coalition_depth:int=1, max_cpus:int=4) \
            -> np.ndarray:
        """ Generates KernelSHAP values for data points in data using the Kernel Shap algorithm

                Parameters
                ----------
                data : numpy.array
                    Matrix of training data samples (# samples x # features). Can be original data
                    or summarized data (output of running kmeans on original data)

                weights : numpy.array
                    weights for each data point. Typically returned by running kmeans on the original data.
                    If original data is being passed, use 1/(num of data points) as the weights

                 new_data: numpy.array
                    new data for which shapley values should be computed (# samples x # features)
                    
                 coalition_depth : int
                    coalition depth. This parameter in-turn controls how many coalitions are considered
                    during shapley value computation. Eg., if coalition_depth = 2, and number of features is 
                    4, then number of coalitions considered = C(4,1) + C(4,2) = 10. Note that no features present
                    corresponds to phi0 and all features present correspond to phi_n
                    
                 max_cpus : int  
                    maximum number of cpus to use during parallel computation of shapley values for multiple data
                    points 


                Returns
                -------
                Matrix of shapley values for new_data points [new_data samples x num_features]
                """
        if (isinstance(data, np.ndarray) and isinstance(weights, np.ndarray) and
            isinstance(new_data, np.ndarray)):
            # assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"
            # data = X.reshape((1, X.shape[0]))
            self.num_features = data.shape[1]
            coalitions = self._generate_coalitions(self.num_features, coalition_depth)
            # num_coalitions = len(self.coalitions)
            if len(new_data.shape) == 1:
                new_data = new_data.reshape(1,-1)

            fx = self.predictor(new_data)
            Ef = np.average(self.predictor(data), weights=weights)
            pi = self._generate_pi(coalitions)
            futures = []
            shap_vals = []

            num_cores = min(max_cpus, os.cpu_count())
            instance_chunks = np.array_split(new_data, num_cores, axis=0)
            fx_chunks = np.array_split(fx, num_cores, axis=0)
            num_splits = len(instance_chunks)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(0, num_splits):
                    futures.append(executor.submit(lambda p: self._shap(*p),
                                    [Ef, fx_chunks[i], data, weights, coalitions, pi, instance_chunks[i]]))

            for future in futures:
                for s in future.result():
                    shap_vals.append(s)
            return shap_vals
        else:
            raise ValueError('input variables must be np.ndarray')

    def _shap(self, Ef, fx, data, weights, coalitions, pi, new_data):
        numnew_datas = new_data.shape[0]
        shap_vals = []
        for i in range(0, numnew_datas):
            Y = self._f_hxz(data, weights, coalitions, new_data[i])
            F = data.shape[1]
            W = np.diag(pi)
            M = np.array(coalitions)
            Mn = M[:, -1]  # last column
            Y = Y - Ef - (fx[i] - Ef) * Mn
            M = M - np.repeat(np.expand_dims(Mn, axis=1), F, axis=1)
            MM = M[:, 0: F-1]
            left = np.linalg.inv(np.dot(np.dot(MM.transpose(), W), MM))
            right = np.dot(np.dot(MM.transpose(), W), Y)
            shap_vals_ = np.dot(left, right)
            phi0 = Ef
            phi_n = fx - phi0 - np.sum(shap_vals_)
            shap_vals.append([])
            shap_vals[i].append(phi0)
            for val in shap_vals_:
                shap_vals[i].append(val)
            shap_vals[i].append(phi_n)

        return np.array(shap_vals)

    def _coalition_weight(self, num_features, z):
        def comb(n, m):
            return math.factorial(n)/(math.factorial(m)*math.factorial(n-m))
        return (num_features-1)/(comb(num_features, z)*z*(num_features-z))

    def _generate_coalitions(self, num_features, coalition_depth):
        coalition = []
        zeros = np.zeros(num_features)

        for i in range(coalition_depth):
            zeros[i] = 1
            perms = distinct_permutations(zeros)
            for p in perms:
                coalition.append(p)

        return coalition

    def _f_hxz(self, data, weights, coalitions, instance):
        N = data.shape[0]
        # M: number of coalitions
        M = len(coalitions)
        # Repeat data M times
        points = np.tile(data, (M, 1))
        for i in range(0, M):
            mask = coalitions[i]
            nz_i = np.nonzero(mask)
            points[i*N: (i+1)*N, nz_i] = instance[nz_i]

        evals = self.predictor(points)
        eY = []
        for i in range(0, N * M, N):
            eY.append(np.average(evals[i: i + N], weights=weights))
        return np.array(eY)

    def _generate_pi(self, coalitions):
        weights = []
        for coal in coalitions:
            z = np.count_nonzero(coal)
            weights.append(self._coalition_weight(self.num_features, z))
        return weights

    def kmeans(self, X, k, round_values=True):
        """ Summarize a dataset with k mean samples weighted by the number of data points they
        each represent.

        Parameters
        ----------
        X : numpy.array
            Matrix of data samples to summarize (# samples x # features)

        k : int
            Number of means to use for approximation.

        round_values : bool
            For all i, round the ith dimension of each mean sample to match the nearest value
            from X[:,i]. This ensures discrete features always get a valid value.

        Returns
        -------
        Cluster centers, cluster weights.
        """

        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

        if round_values:
            for i in range(k):  # for each cluster
                for j in range(X.shape[1]):  # find closest data point whose feature j is closest to cluster i
                    ind = np.argmin(np.abs(X[:, j] - kmeans.cluster_centers_[i, j]))
                    kmeans.cluster_centers_[i, j] = X[ind, j]  # set the j'th feature of cluster i to the closest
                    # matching data point
        # The weights of a cluster are proportional to the number of elements in the cluster
        weights = 1.0 * np.bincount(kmeans.labels_)
        weights /= np.sum(weights)
        return kmeans.cluster_centers_, weights


