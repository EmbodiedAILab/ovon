#!/bin/bash

# 使用：./build_update.sh test fsq_develop_1

tag="${1:-}"           # 镜像 tag
branch_tag="${2:-}"    # OVON 分支/标签（传给 Docker --build-arg branch_tag）

if [[ -z "$tag" || -z "$branch_tag" ]]; then
  echo "Usage: $0 <image-tag> <ovon-branch-tag>"
  echo "Example: $0 v0.4 feature/abc"
  exit 1
fi

# 检查环境变量是否存在
if [ -z "$GITHUB_USERNAME" ] || [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_USERNAME or GITHUB_TOKEN environment variable is not set."
    exit 1
fi

echo $GITHUB_TOKEN

cd ../../
docker build --no-cache --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} --build-arg GITHUB_USERNAME=${GITHUB_USERNAME}  --build-arg OVON_BRANCH_TAG=${branch_tag} -f ./scripts/docker/Dockerfile.update -t swr.cn-east-3.myhuaweicloud.com/meadow/ovon:dev-cu113_$tag  .
cd scripts/docker
