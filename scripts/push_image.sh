#!/bin/bash

export $(cat .env | xargs)

DEPLOYMENT_NAME=torchserve

VERSION=v1

az acr login --name ${AZURE_CONTAINER_REGISTRY_NAME}

docker tag ${DEPLOYMENT_DOCKER_IMAGE} ${AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/${DEPLOYMENT_DOCKER_IMAGE}:${VERSION}

docker push ${AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/${DEPLOYMENT_DOCKER_IMAGE}:${VERSION}
