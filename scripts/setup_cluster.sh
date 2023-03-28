#!/bin/bash
export $(cat .env | xargs)

az aks create --resource-group ${AZURE_RESOURCE_GROUP} --name computecluster --node-vm-size Standard_DC2s_v2 --node-count 1 --generate-ssh-keys
