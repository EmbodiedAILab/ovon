#!/bin/bash
tag="${1:-}"

if [[ -z "$tag" ]]; then
  echo "Usage: $0 <tag>"
  exit 1
fi

cd ../../
docker build --no-cache --build-arg github_access_token=${GITHUB_TOKEN} -f ./scripts/docker/Dockerfile.update -t swr.cn-east-3.myhuaweicloud.com/meadow/ovon:dev-cu113_$tag  .
cd scripts/docker

