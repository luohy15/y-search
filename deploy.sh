#!/bin/bash

set -e

# Load local secrets
if [ -f ".env.local" ]; then
    export $(grep -v '^#' .env.local | xargs)
elif [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

AWS_PROFILE=${AWS_PROFILE:-default}

uv export --format=requirements-txt --no-hashes | grep -v "^-e \." > requirements.txt

sam build

if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &>/dev/null; then
    echo "AWS credentials not found or expired. Run: aws sso login --profile $AWS_PROFILE"
    exit 1
fi

add_param() {
    local param_name="$1"
    local param_value="$2"
    if [ -n "$param_value" ]; then
        if [ -n "$PARAM_OVERRIDES" ]; then
            PARAM_OVERRIDES="$PARAM_OVERRIDES $param_name=$param_value"
        else
            PARAM_OVERRIDES="$param_name=$param_value"
        fi
    fi
}

PARAM_OVERRIDES=""
add_param "DatabaseUrl" "$DATABASE_URL"
add_param "LambdaSecurityGroupId" "$LAMBDA_SECURITY_GROUP_ID"
add_param "SubnetIds" "$SUBNET_IDS"

if [ -n "$PARAM_OVERRIDES" ]; then
    sam deploy --profile "$AWS_PROFILE" --parameter-overrides $PARAM_OVERRIDES
else
    sam deploy --profile "$AWS_PROFILE"
fi
