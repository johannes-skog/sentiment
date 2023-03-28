import os
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
from util import get_ml_client, get_latest_model_version, get_latest_data_version
import argparse

DESTINATION_FOLDER = "artifacts"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name', type=str, default="xlmr_sentiment_traced",
        help="name of the model as called in az"
    )
    parser.add_argument(
        '--version', type=int, default=1,
        help="version of the model that is going to be packaged",
    )
    parser.add_argument(
        '--tokenizer', type=str, default="tokenizer",
        help="name of the tokenizer as called in az"
    )
    parser.add_argument(
        '--handler', type=str, default="handler.py",
        help="name of handler module"
    )
    parser.add_argument(
        '--requirements', type=str, default="requirements.txt",
        help="path of the requirements file"
    )

    args = parser.parse_args()

    ml_client = get_ml_client()

    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)
    
    artifact_utils.download_artifact_from_aml_uri(
        uri=ml_client.models.get(
            name=args.tokenizer,
            version=get_latest_model_version(args.tokenizer),
        ).path,
        destination=DESTINATION_FOLDER,
        datastore_operation=ml_client.datastores
    )
    artifact_utils.download_artifact_from_aml_uri(
        uri=ml_client.models.get(
            name=args.model_name,
            version=get_latest_model_version(args.model_name)
        ).path,
        destination=DESTINATION_FOLDER,
        datastore_operation=ml_client.datastores
    )

    cmd = f"""torch-model-archiver \
    --force\
    --model-name sentiment\
    --version {args.version}\
    --serialized-file {DESTINATION_FOLDER}/traced.pt\
    --handler {args.handler}\
    --export-path {DESTINATION_FOLDER}\
    --requirements-file {args.requirements}\
    --extra-files {DESTINATION_FOLDER}/special_tokens_map.json,{DESTINATION_FOLDER}/tokenizer_config.json,{DESTINATION_FOLDER}/tokenizer.json"""

    os.system(cmd)