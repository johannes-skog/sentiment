#!/bin/bash
export $(cat .env | xargs)

# az group create --name ${AZURE_RESOURCE_GROUP} --location northeurope

az acr create --resource-group ${AZURE_RESOURCE_GROUP} \
  --name ${AZURE_CONTAINER_REGISTRY_NAME} --sku Basic

az acr login --name ${AZURE_CONTAINER_REGISTRY_NAME}
