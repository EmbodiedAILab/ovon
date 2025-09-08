#!/bin/bash
tag="${1:-}"           # 镜像 tag
branch_tag="${2:-}"    # OVON 分支/标签（传给 Docker --build-arg branch_tag）

if [[ -z "$tag" || -z "$branch_tag" ]]; then
  echo "Usage: $0 <image-tag> <ovon-branch-tag>"
  echo "Example: $0 v0.4 feature/abc"
  exit 1
fi

cd ../../
docker build --no-cache --build-arg github_access_token=${GITHUB_TOKEN}  --build-arg OVON_BRANCH_TAG=${branch_tag} -f ./scripts/docker/Dockerfile.update -t swr.cn-east-3.myhuaweicloud.com/meadow/ovon:dev-cu113_$tag  .
cd scripts/docker