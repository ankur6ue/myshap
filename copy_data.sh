copy_data(){
    ROOT_DIR=$1
    SSH_KEY_LOCATION=$2

    while IFS= read -r line
    do
      echo "$line"
      worker_ip=$line
      # step 2: copy any conda env and source files from master to worker nodes. This is obviously project specific
      rsync -e "ssh -i $SSH_KEY_LOCATION" -a --relative ./source ankur@$worker_ip:$ROOT_DIR
      rsync -e "ssh -i $SSH_KEY_LOCATION" -a --relative ./ray_source/build_ray_docker_images.sh ankur@$worker_ip:$ROOT_DIR
      scp -i $SSH_KEY_LOCATION -r ./.dockerignore ankur@$worker_ip:$ROOT_DIR/
      scp -i $SSH_KEY_LOCATION ./envs/environment.yml ankur@$worker_ip:$ROOT_DIR/envs/environment.yml
      scp -i $SSH_KEY_LOCATION ./winequality-red.csv ankur@$worker_ip:$ROOT_DIR/winequality-red.csv
      scp -i $SSH_KEY_LOCATION ./requirements.txt ankur@$worker_ip:$ROOT_DIR/
      scp -i $SSH_KEY_LOCATION ./Dockerfile ankur@$worker_ip:$ROOT_DIR/

    done < "$input"
}
