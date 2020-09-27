update_master_node() {
  # Master node
  # cd to project folder directory if necessary
  # update conda environment (create if necessary)
  NUM_CPUS=$1
  PORT=$2
  IMAGE_NAME=$3
  CONTAINER_NAME=$4

  docker build -t $IMAGE_NAME:latest .

  # stop running containers with name CONTAINER_NAME
  docker ps --filter "name=$CONTAINER_NAME" -aq | xargs docker stop | xargs docker rm

  # First run our docker container based on the image we just created, then run ray on it. Must be
  # done in this sequence for this to work..
  # -itd runs the container in interactive mode in the background
  # --rm removes the container when it is stopped
  # --network=host lets workers connect to ray master using the master machine's IP
  # /bin/bash lets the container running so we can run exec commands
  docker run -itd --rm --name=$CONTAINER_NAME --shm-size=2G --network=host $IMAGE_NAME /bin/bash
  # -f option forcibly stops ray processes
  docker exec $CONTAINER_NAME \
    ray start --head --num-cpus=$NUM_CPUS --redis-password="password" --port=$PORT
}


update_worker_node() {
  # Worker node
  # cd to project folder directory if necessary
  # update conda environment (create if necessary)

  NUM_CPUS=$1
  PORT=$2
  MASTER_IP=$3
  IMAGE_NAME=$4
  CONTAINER_NAME=$5
  ROOT_DIR=$6
  # change directory to the location of Dockerfile
  cd $ROOT_DIR
  docker build -t $IMAGE_NAME:latest .
   # stop running containers with name CONTAINER_NAME
  docker ps --filter "name=$CONTAINER_NAME" -aq | xargs docker stop | xargs docker rm

  # First run our docker container based on the image we just created, then run ray on it. Must be
  # done in this sequence for this to work..
  # -itd runs the container in interactive mode in the background
  # --rm removes the container when it is stopped
  # --network=host lets workers connect to ray master using the master machine's IP
  # /bin/bash lets the container run so we can run exec commands later
  docker run -itd --rm --name=$CONTAINER_NAME --shm-size=2G --network=host $IMAGE_NAME /bin/bash

  docker exec $CONTAINER_NAME \
  ray start --num-cpus=$NUM_CPUS --redis-password="password" --address=$MASTER_IP:$PORT
}

# This script launches the application docker container on the master node and starts ray. It then copies
# application source files to the worker nodes and launches the application docker container on the worker nodes
# and starts ray worker processes which join the ray processes launched on the master node

# see setup_docker.sh for comments about the parameters below
master_node_address="$(hostname --ip-address)"
NUM_CPUS=2
NUM_CPUS_WORKERS=8
PORT=6378
SSH_KEY_LOCATION=~/.ssh/home_ubuntu_laptop/id_rsa
IMAGE_NAME=ray/myshap
CONTAINER_NAME=myshap
input="workers.txt"
ROOT_DIR=~/dev/apps/ML/Interpretability/myshap/

# step 1: copy data from head to worker node
copy_data $ROOT_DIR $SSH_KEY_LOCATION

# step 2: run ray on head node
update_master_node $NUM_CPUS $MASTER_IP $PORT $IMAGE_NAME $CONTAINER_NAME

if [ "$1" != "copy_only" ]; then
  # step 2: run head node
  # update_master_node $NUM_CPUS $master_node_address $PORT
  # step 3: attach worker nodes to master. Worker IPs are specified in cluster_config.txt
  while IFS= read -r line
  do
    echo "$line"
    worker_ip=$line

    # This line opens a SSH connection with each worker and calls update_slave_node with the arguments provided on the worker node

    ssh ankur@$worker_ip -i $SSH_KEY_LOCATION "$(typeset -f update_worker_node); \
    update_worker_node $NUM_CPUS_WORKERS $PORT $master_node_address $IMAGE_NAME $CONTAINER_NAME $ROOT_DIR"
  done < "$input"
fi
