
# Setting up a Service Principle for a azure ML workspace

AUTH_NAME=authv2
### Add extention 
```
az extension add -n azure-cli-ml
```
### Create the Service Principal with az after login in
```
az ad sp create-for-rbac --sdk-auth --name ${AUTH_NAME}
```
### Capture the "objectId" using the clientID
```
az ad sp show --id xxxxxxxx-3af0-4065-8e14-xxxxxxxxxxxx
```
The object id is called "id"

### Assign the role to the new Service Principal for the given Workspace, Resource Group and User objectId
```
az ml workspace share -w <WORKSPACE_NAME> -g <RESOURCE_GROUP> --user xxxxxxxx-cbYYYYdb-YYY-089f-xxxxxxxxxxxx --role owner
```

### Loggin in with an SP 

###  Note it's the APP_ID not the "id" or object id
```
az login --service-principal --username appID --password PASSWORD --tenant tenantID
```
```
svc_pr = ServicePrincipalAuthentication(
    tenant_id=config['AZURE_TENANT_ID'],
    service_principal_id=config['AZURE_CLIENT_ID'],
    service_principal_password=config['AZURE_CLIENT_SECRET']
)
```

#### query the password
```
az ad sp create-for-rbac --name ai-auth --query password -o tsv
```