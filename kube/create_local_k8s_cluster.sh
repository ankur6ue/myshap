# The steps shown below are for the most part the same as described in the instructions on how to deploy
# a local kubernetes cluster using kubeadm here:
# https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/
# turn off swap
sudo swapoff -a
# reset current state
yes| sudo kubeadm reset
# We use two custom settings in kubeadm:
# 1. initialize kubeadm with a custom setting for horizontal pod autoscaler (hpa) downscale (lower downscale time from
# default (5 min) to 15s)) so that pod downscaling occurs faster
# 2. Specify a custom pod-network-cidr, which is needed by Flannel
sudo kubeadm init --config custom-kube-controller-manager-config.yaml
# To see default kubeadm configs, use:
# kubeadm config print init-defaults

mkdir -p $HOME/.kube
yes| sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

kubectl apply -f https://github.com/coreos/flannel/raw/master/Documentation/kube-flannel.yml

# to get rid of old cni plugins, delete cni files in /etc/cni/net.d/
# Wait for all pods to come up
sleep 20s
# verify all kube-system pods are running
kubectl get pods -n kube-system
# By default, your cluster will not schedule Pods on the control-plane node for security reasons.
# If you want to be able to schedule Pods on the control-plane node, for example for a single-machine
# Kubernetes cluster for development, run:
kubectl taint nodes --all node-role.kubernetes.io/master-

# to see token list
kubeadm token list

############ CLUSTER SET UP COMPLETE #####################

# Next steps: install metrics-server
kubectl create -f metrics-server/components.yaml
# Note the --metric-resolution=15s setting
# Now you can use kubectl top pods to see CPU/Memory utilization for a pod

# create a namespace for ray
kubectl apply -f ray-namespace.yaml

# run aws configure
# create secret. Note: must have run aws configure --profile <profile_name> before this step
. ../get_session_creds.sh
# need to go one directory up because iamroles.txt file needed by get_session_creds script is in top level directory
cd ..
get_session_creds
cd kube

# create our deployment
kubectl apply -f ray-cluster.yaml -n ray
# give some time for the pods to come up
sleep 3s

# Create horizontal Autoscaler
kubectl delete -f ray-autoscaler.yaml -n ray
kubectl create -f ray-autoscaler.yaml -n ray
# to get/describe HPAs in the ray namespace
kubectl get hpa -n ray
# to get info about a specific HPA
# kubectl get ray-autoscaler.yaml -n ray

# Run our program
# --local keyword instructs Ray to run this program on an existing cluster,
# rather than create a new one
# --use-s3 tells the program to load the input data to train the RandomForest model from S3 rather than from a local
# file.
# num_rows is the number of rows for which shap values are calculated. Increasing this number will make the program
# run for longer
kubectl exec deployments/ray-head -n ray -- python source/kernel_shap_test_ray.py --local=0 --use_s3=1 --num_rows=32

