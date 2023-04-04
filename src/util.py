
from azure.ai.ml import MLClient
from azureml.core.authentication import ServicePrincipalAuthentication
import os
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils

def get_latest_data_version(name: str):

    ml_client = get_ml_client()

    version = max(
        [int(m.version) for m in ml_client.data.list(name=name)]
    )

    return version

def get_latest_model_version(name: str):

    ml_client = get_ml_client()

    version = max(
        [int(m.version) for m in ml_client.models.list(name=name)]
    )

    return version

def get_ml_client():

    svc_pr = ServicePrincipalAuthentication(
        tenant_id=os.getenv('AZURE_TENANT_ID'),
        service_principal_id=os.getenv('AZURE_CLIENT_ID'),
        service_principal_password=os.getenv('AZURE_CLIENT_SECRET')
    )

    ml_client = MLClient(
        svc_pr,
        subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
        workspace_name=os.getenv('AZURE_WORKSPACE_NAME')
    )

    return ml_client

def download_dataset(
    ml_client,
    name: str,
    destination: str,
    version: str = None,
):
    if version is None:
        version = get_latest_data_version(name)

    data_info = ml_client.data.get(name=name, version=str(version))

    artifact_utils.download_artifact_from_aml_uri(
        uri=data_info.path,
        destination=destination,
        datastore_operation=ml_client.datastores
    )


def create_traced_model(tokenizer, model):

    dd = tokenizer(
        ["This is a test...", "Detta är ett test..."],
        return_tensors="pt",
        padding=True
    )

    model.eval()

    jit_model = torch.jit.trace(
        model,
        [
            dd["input_ids"].to(model._device),
            dd["attention_mask"].to(model._device)
        ]
    )

    return jit_model