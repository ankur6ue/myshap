# This script copies necessary files from the master to the worker nodes
copy_data(){
    ROOT_DIR=$1
    SSH_KEY_LOCATION=$2

    while IFS= read -r line
    do
      echo "$line"
      # Get the IP address of the worker
      worker_ip=$line
      # Sync the contents of the source directory
      rsync -e "ssh -i $SSH_KEY_LOCATION" -a --relative ./source ankur@$worker_ip:$ROOT_DIR
      # Sync the script that builds the docker image on the worker
      rsync -e "ssh -i $SSH_KEY_LOCATION" -a --relative ./ray_source/build_ray_docker_images.sh ankur@$worker_ip:$ROOT_DIR
      # Copy dockerignore so we are not sending unnecessary data to the docker daemon
      scp -i $SSH_KEY_LOCATION -r ./.dockerignore ankur@$worker_ip:$ROOT_DIR/
      # Conda environment file, not important when using docker images
      scp -i $SSH_KEY_LOCATION ./envs/environment.yml ankur@$worker_ip:$ROOT_DIR/envs/environment.yml
      # csv file containing our data. Not necessary if file is being read from AWS S3
      scp -i $SSH_KEY_LOCATION ./winequality-red.csv ankur@$worker_ip:$ROOT_DIR/winequality-red.csv
      # Requirements.txt
      scp -i $SSH_KEY_LOCATION ./requirements.txt ankur@$worker_ip:$ROOT_DIR/
      # AWS keys, passed to the docker run command as file environment variable
      scp -i $SSH_KEY_LOCATION ./env.list ankur@$worker_ip:$ROOT_DIR/
      # Dockerfile to build images
      scp -i $SSH_KEY_LOCATION ./Dockerfile ankur@$worker_ip:$ROOT_DIR/

    done < "$input"
}
