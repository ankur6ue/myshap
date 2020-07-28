import math
import numpy as np
from more_itertools import distinct_permutations


class KernelShapModel:

    def __init__(self, predictor):
        self.predictor = predictor
        self.coalitions = None

    def run(self, data, _instance, coalition_depth=1):
        self.num_features = data.shape[1]
        self.coalitions = self._generate_coalitions(self.num_features, coalition_depth)
        # num_coalitions = len(self.coalitions)
        self.instance = _instance
        model_out = self.predictor(_instance)
        fx = model_out[0]
        Y = self._f_hxz(data)
        pi = self._generate_pi()
        W = np.diag(pi)
        X = np.array(self.coalitions)
        Xn = X[:, -1]  # last column
        Ef = np.mean(self.predictor(data))
        Y = Y - Ef - (fx - Ef)*Xn
        X = X - np.repeat(np.expand_dims(Xn, axis=1), self.num_features, axis=1)
        XX = X[:, 0:self.num_features-1]
        left = np.linalg.inv(np.dot(np.dot(XX.transpose(), W), XX))
        right = np.dot(np.dot(XX.transpose(), W), Y)
        shap_vals_ = np.dot(left, right)
        phi0 = Ef
        phi_n = fx - phi0 - np.sum(shap_vals_)
        shap_vals = []
        shap_vals.append(phi0)
        for val in shap_vals_:
            shap_vals.append(val)
        shap_vals.append(phi_n)
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

    def _f_hxz(self, data):
        N = data.shape[0]
        # M: number of coalitions
        M = len(self.coalitions)
        points = np.zeros((N*M, self.num_features), np.float)
        for i in range(0, M):
            mask = self.coalitions[i]
            for j in range(0, N):
                points[j + i*N] = data[j]
                nz_i = np.nonzero(mask)
                points[j + i*N, nz_i] = self.instance[0][nz_i]
        evals = self.predictor(points)
        eY = []
        for i in range(0, N*M, N):
            eY.append(np.mean(evals[i: i+N]))
        return np.array(eY)

    def _generate_pi(self):
        weights = []
        for coal in self.coalitions:
            z = np.count_nonzero(coal)
            weights.append(self._coalition_weight(self.num_features, z))
        return weights




