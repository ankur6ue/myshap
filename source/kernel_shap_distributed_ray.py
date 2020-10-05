import math
import numpy as np
import ray
from kernel_shap import KernelShapModel, generate_coalitions


@ray.remote
def call_shap(self, Ef_, fx_chunks, data_, weights_, coalitions_, pi_, instance_chunks):
    return self._shap(Ef_, fx_chunks, data_, weights_, coalitions_, pi_, instance_chunks)


class KernelShapModelDistributedRay(KernelShapModel):
    def __init__(self, predictor):
        super().__init__(predictor)

    def run(self, data: np.ndarray, weights: np.ndarray, new_data:np.ndarray, coalition_depth:int=1,
            num_workers:int=5)-> np.ndarray:
        """
        Generates KernelSHAP values for data points in data using the KernelShap algorithm.
        This version of KernelShap distributes out the shapley value calculation over N cpus.
                N = min(max_cpus, number of cpus on your machine).

        Parameters
        ----------
        data : numpy.array
            Matrix of training data samples (# samples x # features). Can be original data
            or summarized data (output of running kmeans on original data)

        weights : numpy.array
            weights for each data point. Typically returned by running kmeans on the original data.
            If original data is being passed, use 1/(num of data points) as the weights

         new_data: numpy.array
            data for which shapley values should be computed (# samples x # features)

         coalition_depth : int
            coalition depth. This parameter controls number of coalitions considered
            during shapley value computation. Eg., if coalition_depth = 2, and number of features is
            4, then number of coalitions considered = C(4,1) + C(4,2) = 10.

         num_workers : int
            number of chunks to split training data samples into. These chunks are processed in
            parallel on a distributed ray cluster

        Returns
        -------
        Matrix of shapley values for new_data points [new_data samples x num_features + 1]. The
        + 1 is because phi0 (no features present) is also returned
        """
        if (isinstance(data, np.ndarray) and isinstance(weights, np.ndarray) and
                isinstance(new_data, np.ndarray)):
            # assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"
            # data = X.reshape((1, X.shape[0]))
            self.num_features = data.shape[1]

            coalitions = generate_coalitions(self.num_features, coalition_depth)
            # num_coalitions = len(self.coalitions)
            if len(new_data.shape) == 1:
                new_data = new_data.reshape(1,-1)

            fx = self.predictor(new_data)
            Ef = np.average(self.predictor(data), weights=weights)
            pi = self._generate_pi(coalitions)
            futures = []
            shap_vals = []

            instance_chunks = np.array_split(new_data, num_workers, axis=0)
            fx_chunks = np.array_split(fx, num_workers, axis=0)
            num_splits = len(instance_chunks)

            for i in range(0, num_splits):
                print("Submitting:", i)
                future = call_shap.remote(self,
                                       Ef, fx_chunks[i], data, weights, coalitions, pi, instance_chunks[i])
                futures.append(future)

            # for future in as_completed(futures): This is sometimes more efficient but may result in out-of-order
            # computation of shap values, and hence requires implementation of re-ordering logic so the shap values
            # line up with the order of data instances.
            for future in futures:
                result = ray.get(future)
                for s in result:
                    shap_vals.append(s)

            return np.array(shap_vals)
        else:
            raise ValueError('input variables must be np.ndarray')



