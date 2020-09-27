# My KernelSHAP implementation

This code implements the KernelSHAP algorithm as described in this [paper](https://arxiv.org/pdf/1705.07874.pdf) 

The Python [shap library](https://github.com/slundberg/shap) already provides an implementation of KernelSHAP, but the implementation is very complex and difficult to understand. My implementation is much shorter and easier to understand. 

For a detailed write up about how Kernel SHAP works, see this [blog post](https://www.telesens.co/2020/09/17/kernel-shap/)

## Prefect + DASK based distributed implementation
I also provide a distributed implementation that uses Python multiprocessing library to distribute out the Shap value computation for multiple data points across multiple processes. I use a RandomForest model trained on the winequality-red dataset to show that the Shap values computed by my implementation matches the default Python implementation exactly.

I use the Prefect workflow orchestration system running over a local DASK cluster to run the ETL task, my serial KernelSHAP implementation, my distributed KernelSHAP implementation, the default Python Shap library implementation and a results comparison tasks. The three KernelSHAP variants are executed in parallel, as shown in the table below.


Total flow execution time (sec) | Distributed implementation | Serial implementation | Default Python implementation 
--- | --- | --- | ---
47.8 | 6.9 | 31.4 | 41.8

To run the code, simply run kernel_shap_test.py from a debugger or command line. No arguments are necessary. You'll have to install DASK, shap package, sklearn and other required libraries if you don't already have them. 

Here's how the Prefect flow looks like:
![](images/prefect_flow.png)

## Distributed implementation using ray
I recently learnt about [Ray](https://docs.ray.io/en/master/ray-overview/index.html), another system for building distributed applications. I was able to set up a distributed cluster consisting of my main computer and a second Ubuntu laptop and run a test script (source/kernel_shap_test_ray.py) on this cluster. 

You can set up the cluster by setting up an identical conda environment on the master and worker nodes and launching ray on each. See setup_conda.sh for details about this approach. Alternatively, you can create a docker container image containing ray and shap code and launch this container on the master and worker nodes. See setup_docker.sh for details about this approach. 

### Instructions 
1. Run ray_source/build_ray_docker_images.sh. This script will clone ray and build the rayproject/ray and other docker images
2. Run `docker build -t ray/myshap .` from the project root directory to build the myshap container image
3. Create a workers.txt with the IP address(es) of your worker node. If you just want to set up a single node ray cluster, you can leave the file blank
4. Run setup_conda.sh for the conda environment approach (easier) or setup_docker.sh for the docker approach (trickier).
5. Once the cluster is running, activate the conda environment and run python source/kernel_shap_test_ray.py in the environment. Note that your application code and ray processes running on the master and worker nodes much use identical conda environment. For the docker approach, use `docker exec` to run kernel_shap_test_ray.py in the docker container running on the master node 