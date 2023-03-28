# Train XLMR for sentiment analysis

## Locally 

### Setup the container 

```
cd dockercontext
make build 
make enter
```

### Create the training datasets and upload to azureml datasets
```
python src/setup.py --dataset_name twitter-sentiment
```

### Train the model on the dataset (downloaded from azureml), upload the model checkpoint and a traced JIT version to azureml models, also upload the tokenizer
```
python train.py --dataset twitter-sentiment --lock_embedding --lock_first_n_layers 7 --batch_size 25 --iterations 50000
```

### Package the model and requirements into a mar-file to make it available for torchserve, fetch the model from azureml models
```
python package.py --model_name xlmr_sentiment_traced --version 1 --handler src/handler.py --requirements dockercontext_torchserve/requirements.txt
```

## On a remote with Reacher

### Setup reacher

```
from reacher.reacher import RemoteClient, Reacher
from dotenv import dotenv_values

config = dotenv_values()  # take environment variables from .env.

reacher = Reacher(
    build_name="pytorch_base",
    image_name="pytorch_base",
    build_context="dockercontext",
    host=config["HOST"],
    user=config["USER"],
    password=config["PASSWORD"],
    ssh_key_filepath=config["SSH_KEY_PATH"],
)

reacher.build()

reacher.setup(
    ports=[8888, 6666],
    envs=dotenv_values(".env") 
)
```

### execute the different pipeline steps on the remote

```
reacher.execute(
    context_folder="src",
    file="src/setup.py",
    command="python setup.py --dataset_name twitter-sentiment",
    named_session="setup_twitter_dataset",
)

reacher.execute(
    context_folder="src",
    file="src/train.py",
    command="python train.py --dataset twitter-sentiment --lock_embedding --lock_first_n_layers 7 --batch_size 25 --iterations 50000",
    named_session="train_sentiment_model",
)

reacher.execute(
    file="dockercontext_torchserve/requirements.txt",
    context_folder="src",
    command="python package.py --model_name xlmr_sentiment_traced --version 1 --handler handler.py --requirements requirements.txt",
    named_session="package_sentiment_model",
)
```

### and then fetch the mar-file 

```
reacher.get_artifact("sentiment.mar", "artifacts")
```

# Deployment on a azure kubernetes cluster with torchserve

## Setup K8s cluster on Azure 

Export envs from .env
```
export $(cat .env | xargs)
```

Create the cluster
```
az aks create --resource-group ${AZURE_RESOURCE_GROUP} --name ${COMPUTE_CLUSTER_NAME} --node-vm-size Standard_DC2s_v2 --node-count 1 --generate-ssh-keys
```

Intall necessary tools i.e. kubectl

```
az aks install-cli
```

Setup the credentials, will push them to ~/.kube/config so that kubectl can be used

```
az aks get-credentials --resource-group ${AZURE_RESOURCE_GROUP}  --name ${COMPUTE_CLUSTER_NAME}
```

## Configure Cluster

#### If gpu powered nod?
```
kubectl apply -f k8s/nvidia-device-plugin-ds.yaml
```

### Configure storage

#### Create a storage class for files across the nodes
```
kubectl apply -f k8s/Azure_file_sc.yaml
```

#### Create a PersistentVolume where we will store mar-files and configs 
```
kubectl apply -f k8s/AKS_pv_claim.yaml
```

#### Create a pod, with PersistentVolume, for copying mar-files and config files 
```
kubectl apply -f k8s/model_store_pod.yaml
```

#### Check the running pods
```
kubectl get pods -o wide
```

### Upload mar/config files to k8s

#### Create the folder on the pod with PersistentVolume where we will upload mar/config files to
```
kubectl exec --tty pod/model-store-pod -- mkdfir /mnt/azure/model-store/
```

#### Copy the mar file to the pod with PersistentVolume
```
kubectl cp dockercontext_torchserve/model_store/sentiment.mar model-store-pod:/mnt/azure/model-store/sentiment.mar
```

#### Copy config files
```
kubectl exec --tty pod/model-store-pod -- mkdir /mnt/azure/config/
kubectl cp dockercontext_torchserve/config.properties model-store-pod:/mnt/azure/config/config.properties
```

#### Check if all the files have been uploaded 
```
kubectl exec --tty pod/model-store-pod -- find /mnt/azure/
```

## Deployment for public access

We will setup a torchserve deployment and a service that forward the inference/management/metric ports to the load balancer for external access
```
kubectl create -f k8s/torchserve_public.yaml
```

#### Get information about torchserve service

```
kubectl describe service torchserve
```

and to get information about all services

```
kubectl get service -A
```

#### Use the external IP together with the correct port to start the sentiment model on the deployment

```
curl -v -X POST "http://<EXTERNAL_IP>:<EXTERNAL_MGM_PORT>/models?initial_workers=1&batch_size=5&maxWorkers=5&max_batch_delay=1000&synchronous=true&url=sentiment.mar"
```

Here we will setup one inital worker and allow torchserve to scale up to 5, if needed. At maximum will we allow a batchsize of 5, if request are arriving within 1000ms torchserve will group them in batches of the maximum size.

#### Delete the deployment and service
```
kubectl delete deployment torchserve
kubectl delete service torchserve
```

#### Inference on the model

```
curl -X POST -H "Content-Type: application/json" -d '["The movie was so good", "The movie was so bad"]' http://<EXTERNAL_IP>>:<EXTERNAL_INFERENCE_PORT>/v1/models/sentiment:predict

[
  0.9991590976715088,
  0.0054536801762878895
]
```

The model is certain about the first text describing something with a postive sentiment and the second with a bad sentiment.

## Deployment for private access

We will only setup a pod with the torchserve image
```
kubectl create -f k8s/torchserve_private.yaml
```

#### Access the torchserve pod 
```
kubectl exec --stdin --tty torchserve -- /bin/bash
```

We can now setup and test torchserve in a similar way as for the public deployment, change <EXTERNAL_IP> to 127.0.0.1 and the ports to the those specified in config.properties.

