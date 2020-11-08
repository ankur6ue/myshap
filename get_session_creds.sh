get_session_creds() {
  # Read IAMRole from file. This IAMRole is attached to the bucket policy and allows read access to the bucket
  role=$(head -n 1 iamroles.txt)
  # Get the Access keys for your AWS dev profile. Change this to your profile name
  PROFILE_NAME=ankur-dev
  unset AWS_ACCESS_KEY_ID
  unset AWS_SECRET_ACCESS_KEY
  # This is important, otherwise the sts assume-role command below can fail
  unset AWS_SESSION_TOKEN
  AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile=$PROFILE_NAME)
  AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile=$PROFILE_NAME)
  export AWS_ACCESS_KEY_ID
  export AWS_SECRET_ACCESS_KEY
  # echo $AWS_SECRET_ACCESS_KEY
  # echo $AWS_ACCESS_KEY_ID
  # see: https://github.com/rik2803/aws-sts-assumerole/blob/master/assumerole
  # Get temporary creds that can be used to assume this role
  creds=$(aws sts assume-role --role-arn ${role} --role-session-name AWSCLI-Session)
  AWS_ACCESS_KEY_ID_SESS=$(echo ${creds} | jq --raw-output ".Credentials[\"AccessKeyId\"]")
  AWS_SECRET_ACCESS_KEY_SESS=$(echo ${creds} | jq --raw-output ".Credentials[\"SecretAccessKey\"]")
  AWS_SESSION_TOKEN=$(echo ${creds} | jq --raw-output ".Credentials[\"SessionToken\"]")

  # Write the creds to a file. This file can be passed as argument to a Docker run command using the --env-file option
  env_file='./env.list'

  if [ -e $env_file ]; then
    echo "AWS_ACCESS_KEY_ID_SESS="$AWS_ACCESS_KEY_ID_SESS >$env_file
    echo "AWS_SECRET_ACCESS_KEY_SESS="$AWS_SECRET_ACCESS_KEY_SESS >>$env_file
    echo "AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN >>$env_file
  fi

  # Create kubernetes secret to pass SSO tokens to the pod
  # first delete any existing secrets
  kubectl delete secret session-token -n ray
  # now create a secret. The use of generic means an opaque secret is created
  kubectl create secret generic session-token -n ray --from-literal=AWS_ACCESS_KEY_ID_SESS=$AWS_ACCESS_KEY_ID_SESS \
    --from-literal=AWS_SECRET_ACCESS_KEY_SESS=$AWS_SECRET_ACCESS_KEY_SESS \
    --from-literal=AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN

  # to see if secret was created:
  #kubectl edit secret session-token -n ray
  # or
  # kubectl get secret/session-token -n ray -o jsonpath='{.data}'
}
