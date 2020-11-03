#!/usr/bin/env bash
# shellcheck disable=SC2155
set -euo pipefail

readonly APP_NAME=ml-deep-learning-final
readonly IMAGE_TAG=${IMAGE_TAG:-latest}
readonly DOCKER_FILE="Dockerfile.inference"
readonly GIT_SHA=$(git rev-parse HEAD)
readonly IMAGE_NAME="${APP_NAME}"
readonly ECR_URL="$(aws ecr get-authorization-token --output text --query 'authorizationData[].proxyEndpoint' | awk -Fhttps:// '{print $2}')"

parse_args() {

  usage=$(cat <<END
  Usage: build_inference [OPTIONS]
  Options:
    -e  --environment string Environment (development|qualification|staging|production)
    -n, --no-cache           Do not use cache when building the image. Optional
    -p, --push               Push the image to ECR. Optional
END
  )

  if [ $# -lt 1 ]; then
    echo "${usage}"
    exit 1
  fi

  opts=
  push=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--environment) environment="$2"; shift; shift;;
        -n|--no-cache) opts="--no-cache"; shift;;
        -p|--push) push=true; shift;;
        *) echo "${usage}" && exit 1; shift;;
    esac
  done

  if [[ (-z "${environment}") || ("${environment}" != "development" && "${environment}" != "qualification" && "${environment}" != "staging" && "${environment}" != "production") ]]; then
      echo "${usage}"
      exit 1
  fi

}

build_image() {
  echo "Building Dockerfile: ${IMAGE_NAME}"
  docker build ${opts} -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${DOCKER_FILE}" .
}

push_image() {
  # Login to ECR
  local login_status="$(aws ecr get-login-password | docker login -u AWS --password-stdin https://"${ECR_URL}")"
  echo "${login_status}"
  if [ "${login_status}" != "Login Succeeded" ]; then exit 1; fi

  # Tag and push the image to ECR
  docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ecr_url}/${IMAGE_NAME}:${IMAGE_TAG}"
  docker push "${ecr_url}/${IMAGE_NAME}:${IMAGE_TAG}"

  # Retag with git-sha: https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-retag.html
  local manifest=$(aws ecr batch-get-image --repository-name "${IMAGE_NAME}" --image-ids imageTag="${IMAGE_TAG}" --query 'images[].imageManifest' --output text)
  aws ecr put-image --repository-name "${IMAGE_NAME}" --image-tag "${GIT_SHA}" --image-manifest "${manifest}"

  echo "Successfully created and pushed the new Docker image"
  echo "${ecr_url}/${IMAGE_NAME}:${IMAGE_TAG}"
}

main() {
  parse_args "$@"
  build_image
  if [ "${push}" = true ]; then push_image; fi
}

main "$@"