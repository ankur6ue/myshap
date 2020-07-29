import math
import numpy as np
from more_itertools import distinct_permutations
import concurrent.futures
import os

class KernelShapModel:

    def __init__(self, predictor):
        self.predictor = predictor
        self.coalitions = None

    def run(self, data, _instance, coalition_depth=1):
        # assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"
        # data = X.reshape((1, X.shape[0]))
        self.num_features = data.shape[1]
        coalitions = self._generate_coalitions(self.num_features, coalition_depth)
        # num_coalitions = len(self.coalitions)
        if len(_instance.shape) == 1:
            _instance = _instance.reshape(1,-1)

        fx = self.predictor(_instance)
        Ef = np.mean(self.predictor(data))
        pi = self._generate_pi(coalitions)
        futures = []
        shap_vals = []

        num_cores = min(5, os.cpu_count())
        instance_chunks = np.array_split(_instance, num_cores, axis=0)
        fx_chunks = np.array_split(fx, num_cores, axis=0)
        num_splits = len(instance_chunks)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, num_splits):
                futures.append(executor.submit(lambda p: self._shap(*p),
                                [Ef, fx_chunks[i], data, coalitions, pi, instance_chunks[i]]))

        for future in futures:
            for s in future.result():
                shap_vals.append(s)
        return shap_vals

    def _shap(self, Ef, fx, data, coalitions, pi, _instance):
        num_instances = _instance.shape[0]
        shap_vals = []
        for i in range(0, num_instances):
            Y = self._f_hxz(data, coalitions, _instance[i])
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

    def _f_hxz(self, data, coalitions, instance):
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
            eY.append(np.mean(evals[i: i + N]))
        return np.array(eY)

    def _generate_pi(self, coalitions):
        weights = []
        for coal in coalitions:
            z = np.count_nonzero(coal)
            weights.append(self._coalition_weight(self.num_features, z))
        return weights




