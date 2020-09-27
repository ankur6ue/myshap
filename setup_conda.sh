update_master_node() {
  # Master node
  # cd to project folder directory if necessary
  # update conda environment (create if necessary)
  NUM_CPUS=$1
  MASTER_IP=$2
  PORT=$3
  conda env update --file envs/environment.yml --name shap_py_env ./envs --prune
  # conda activate has issues from within a shell script, workaround is to use source activate
  source activate shap_py_env
  # -f option forcibly stops ray processes
  ray stop -f
  ray start --head --num-cpus=$NUM_CPUS --redis-password="password" --port=$PORT
}

update_worker_node() {
  # Master node
  # cd to project folder directory if necessary
  # update conda environment (create if necessary)
  NUM_CPUS=$1
  MASTER_IP=$2
  PORT=$3
  ROOT_DIR=$4
  # change directory to where we copied files and update and then activate the conda environment
  # this is necessary because the master and workers must have the same conda environment for multi-node execution
  # to work
  cd $ROOT_DIR
  conda env update --file envs/environment.yml --name shap_py_env ./envs --prune
  source activate shap_py_env
  # -f option forcibly stops ray processes
  ray stop -f
  # start ray on worker node
  ray start --num-cpus=$NUM_CPUS --address=$MASTER_IP:$PORT --redis-password="password"
}

# This script sets up a conda environment on the master node and starts ray. It then copies
# application source files to the worker nodes, sets up the same conda environment on the worker nodes
# and starts ray worker processes which join the ray processes launched on the master node

# Get IP address of master node
master_node_address="$(hostname --ip-address)"
# number of ray worker processes to launch on the master node. Should be less than the number of CPUs
NUM_CPUS=2
# number of ray worker processes to launch on the worker nodes
NUM_CPUS_WORKERS=6
# Ray port on the master node
PORT=6378
# location of the ssh keys to connect with your worker nodes.
# See https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-on-ubuntu-1604
# for more info
SSH_KEY_LOCATION=~/.ssh/home_ubuntu_laptop/id_rsa
# IP addresses of the worker nodes
input="workers.txt"
# Root directory for the myshap application on the worker nodes
ROOT_DIR=~/dev/apps/ML/Interpretability/myshap/

# step 1: copy data from head to worker node
. ./copy_data.sh
copy_data $ROOT_DIR $SSH_KEY_LOCATION

if [ "$1" != "copy_only" ]; then
  # step 2: run ray on master node
  update_master_node $NUM_CPUS $master_node_address $PORT
  # step 3: attach worker nodes to master. Worker IPs are specified in workers.txt
  while IFS= read -r line
  do
    echo "$line"
    worker_ip=$line
    # This line opens a SSH connection with each worker and calls update_worker_node with
    # the provided arguments
    ssh ankur@$worker_ip -i $SSH_KEY_LOCATION "$(typeset -f update_worker_node); \
    update_worker_node $NUM_CPUS_WORKERS $master_node_address $PORT $ROOT_DIR"
  done < "$input"
fi
