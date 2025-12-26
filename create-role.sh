#!/bin/bash

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

set -e

STACK_NAME=${STACK_NAME:-"y-search"}
[ -n "$1" ] && STACK_NAME="$1"

ROLE_NAME="${STACK_NAME}-lambda-role"

AWS_PROFILE_OPTION=""
[ -n "$AWS_PROFILE" ] && AWS_PROFILE_OPTION="--profile $AWS_PROFILE"

echo "Creating Lambda execution role: $ROLE_NAME"

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

aws iam create-role $AWS_PROFILE_OPTION \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --no-cli-pager 2>/dev/null || echo "Role exists, continuing..."

# Attach policies
for POLICY in \
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
    "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
do
    aws iam attach-role-policy $AWS_PROFILE_OPTION \
        --role-name "$ROLE_NAME" \
        --policy-arn "$POLICY"
done

echo "✅ Role setup complete: $ROLE_NAME"
