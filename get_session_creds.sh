get_session_creds() {
  # Read IAMRole from file
  role=$(head -n 1 iamroles.txt)
  AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile=ankur-dev)
  AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile=ankur-dev)
  export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
  export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
  # see: https://github.com/rik2803/aws-sts-assumerole/blob/master/assumerole
  # Get temporary creds that can be used to assume this role
  creds=$(aws sts assume-role --role-arn ${role} --role-session-name AWSCLI-Session)
  AWS_ACCESS_KEY_ID_SESS=$(echo ${creds} | jq --raw-output ".Credentials[\"AccessKeyId\"]")
  AWS_SECRET_ACCESS_KEY_SESS=$(echo ${creds} | jq --raw-output ".Credentials[\"SecretAccessKey\"]")
  AWS_SESSION_TOKEN=$(echo ${creds} | jq --raw-output ".Credentials[\"SessionToken\"]")

  env_file='./env.list'

  if [ -e $env_file ]
  then
      echo  "AWS_ACCESS_KEY_ID_SESS="$AWS_ACCESS_KEY_ID_SESS > $env_file
      echo  "AWS_SECRET_ACCESS_KEY_SESS="$AWS_SECRET_ACCESS_KEY_SESS >> $env_file
      echo  "AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN >> $env_file
  fi

}
