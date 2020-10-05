import os
import boto3
import pandas as pd
import io
import logging

# sudo AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
# spark-ec2/spark-ec2
# -k keypair --identity-file=keypair.pem --region=us-west-2 --zone=us-west-2a --copy-aws-credentials --instance-type t2.micro --worker-instances 1 launch project-launch

# aws_id = os.popen('aws configure get aws_access_key_id --profile=ankur-dev').read()
# get your credentials from environment variables


def set_session_creds(role):
    sts_client = boto3.client('sts')

    # Call the assume_role method of the STSConnection object and pass the role
    # ARN and a role session name.

    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_id = os.environ['AWS_ACCESS_KEY_ID']
    print('aws_id: {0}'.format(aws_id) )
    print('aws_secret: {0}'.format(aws_secret))
    assumed_role_object = sts_client.assume_role(
        RoleArn=role,
        RoleSessionName="S3AccessAssumeRoleSession"
    )


    # From the response that contains the assumed role, get the temporary
    # credentials that can be used to make subsequent API calls
    credentials = assumed_role_object['Credentials']
    os.environ['AWS_ACCESS_KEY_ID_SESS'] = credentials['AccessKeyId']
    os.environ['AWS_SECRET_ACCESS_KEY_SESS'] = credentials['SecretAccessKey']
    os.environ['AWS_SESSION_TOKEN'] = credentials['SessionToken']


def write_to_s3(bucket_name, csv_file, parent_dir):


    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_id = os.environ['AWS_ACCESS_KEY_ID']
    client = boto3.client('s3', aws_access_key_id=aws_id,
    aws_secret_access_key=aws_secret)

    df = pd.read_csv(os.path.join(parent_dir, csv_file))
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name, csv_file).put(Body=csv_buffer.getvalue())


def read_from_s3(bucket_name, object_key):

    logger = logging.getLogger(__name__)
    aws_secret = os.environ['AWS_SECRET_ACCESS_KEY_SESS']
    aws_id = os.environ['AWS_ACCESS_KEY_ID_SESS']
    token = os.environ['AWS_SESSION_TOKEN']
    logger.warning('aws_id')
    logger.warning(aws_id)
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=token
    )

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=token
    )
    # Use the Amazon S3 resource object that is now configured with the
    # credentials to access your S3 buckets.
    # for bucket in s3_resource.buckets.all():
    #    print(bucket.name)
    print('reading {0} from S3'.format(object_key))
    obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df



# write_to_s3()
