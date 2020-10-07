update_master_node() {
  # Master node
  # cd to project folder directory if necessary
  # update conda environment (create if necessary)
  NUM_CPUS=$1
  PORT=$2
  IMAGE_NAME=$3
  CONTAINER_NAME=$4
  ROOT_DIR=$5

  # Build Docker image
  docker build -t $IMAGE_NAME:latest --target shap-image .

  # stop running containers with name CONTAINER_NAME
  docker ps --filter "name=$CONTAINER_NAME" -aq | xargs docker stop | xargs docker rm

  # First run our docker container based on the image we just created, then run Ray on it. Must be
  # done in this sequence for this to work..
  # -itd runs the container in interactive mode in the background
  # --rm removes the container when it is stopped
  # --network=host lets workers connect to ray master using the master machine's IP
  # /bin/bash lets the container running so we can run exec commands
  docker run -itd --rm --name=$CONTAINER_NAME --cpus=0.000 --shm-size=12G -v $ROOT_DIR/source:/app/source \
  --network=host --env-file ./env.list $IMAGE_NAME /bin/bash
  # -f option forcibly stops ray processes
  ray stop -f
  docker exec $CONTAINER_NAME \
    ray start --head --num-cpus=$NUM_CPUS --redis-password="password" --port=$PORT
}

update_worker_node() {
  # Worker node
  # cd to project folder directory if necessary
  NUM_CPUS=$1
  PORT=$2
  MASTER_IP=$3
  IMAGE_NAME=$4
  CONTAINER_NAME=$5
  ROOT_DIR=$6
  # change directory to the location of Dockerfile
  cd $ROOT_DIR
  docker build -t $IMAGE_NAME:latest --target shap-image .
   # stop running containers with name CONTAINER_NAME
  docker ps --filter "name=$CONTAINER_NAME" -aq | xargs docker stop | xargs docker rm

  # First run our docker container based on the image we just created, then run ray on it. Must be
  # done in this sequence for this to work..
  # -itd runs the container in interactive mode in the background
  # --rm removes the container when it is stopped
  # --cpus=0.000 means no limit on number of cpus the container can use
  # --network=host lets workers connect to ray master using the master machine's IP
  # /bin/bash lets the container run so we can run exec commands later
  docker run -itd --rm --name=$CONTAINER_NAME --cpus=0.000 --shm-size=12G -v $ROOT_DIR/source:/app/source \
  --network=host --env-file ./env.list $IMAGE_NAME /bin/bash
   # -f option forcibly stops any ray processes running on the master node
  ray stop -f
  docker exec $CONTAINER_NAME \
  ray start --num-cpus=$NUM_CPUS --redis-password="password" --address=$MASTER_IP:$PORT
}

# This script:
# 1. Builds and launches the application docker containers on the master and worker nodes and starts ray.
# 2. Copies application source and data files to the worker nodes
# 3. Uses AWS STS (Secure Token Service) to retrieve temporary credentials, and saves those to a file. These credentials
# are passed to the application throug -env-file option in the docker run command. The application uses these credentials
# to make Boto calls to get input data from AWS S3

# # Get IP address of master node
master_node_address="$(hostname --ip-address)"
# Number of CPUs to use on the master node
NUM_CPUS=8
# Number of CPUs to use on the worker node
NUM_CPUS_WORKERS=10
# Ray port on the master node
PORT=6378
# location of the ssh keys to connect with your worker nodes.
# See https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-on-ubuntu-1604
# for more info
SSH_KEY_LOCATION=~/.ssh/home_ubuntu_laptop/id_rsa
# Name of the image
IMAGE_NAME=ray/myshap
# Name of the container
CONTAINER_NAME=myshap
# File containing IP address(es) of the worker nodes
input="workers.txt"
# Root directory
ROOT_DIR=~/dev/apps/ML/Interpretability/myshap

# Create an env.list file containing the temporary AWS credentials
# This syntax allows to call a function (get_session_creds) declared in another shell file
. ./get_session_creds.sh
# echo "Reading iamroles.txt"
creds=$(get_session_creds)

# step 1: copy data from head to worker node
. ./copy_data.sh
copy_data $ROOT_DIR $SSH_KEY_LOCATION

# step 2: run ray on head node
update_master_node $NUM_CPUS $PORT $IMAGE_NAME $CONTAINER_NAME $ROOT_DIR

# step 3: attach worker nodes to master. Worker IPs are specified in workers.txt
while IFS= read -r line
do
  echo "$line"
  worker_ip=$line

  # This line opens a SSH connection with each worker and calls update_worker_node with the arguments provided on the worker node

  ssh ankur@$worker_ip -i $SSH_KEY_LOCATION "$(typeset -f update_worker_node); \
  update_worker_node $NUM_CPUS_WORKERS $PORT $master_node_address $IMAGE_NAME $CONTAINER_NAME $ROOT_DIR"
done < "$input"

