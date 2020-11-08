import os

aws_secret = os.environ['AWS_SECRET_ACCESS_KEY_SESS']
aws_id = os.environ['AWS_ACCESS_KEY_ID_SESS']
token = os.environ['AWS_SESSION_TOKEN']

print(aws_secret)
print(aws_id)