from datasets import load_dataset
import argparse
from azureml.core.run import Run
from azureml.core.dataset import Dataset
# from utils import _setup_model
from azureml.core.model import Model
from model import SentimentXLMRModel, TrainingLoopStructure
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import get_ml_client


def _setup_dataset(tokenizer, path: str = "temp_twitter"):

    dataset = load_dataset("carblacac/twitter-sentiment-analysis", "None")

    dataset = dataset.map(
        lambda e: tokenizer(
            e["text"],
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding='max_length'
        ),
        batched=True
    )

    dataset = dataset.rename_column("feeling", "label")

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    dataset.save_to_disk(path)

    return dataset
   
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name', type=str, default="twitter-sentiment",
        help="name of the dataset as will be called in az"
    )

    args = parser.parse_args()

    ml_client = get_ml_client()

    model = SentimentXLMRModel(
        model_folder="./tmpmodel"
    )   

    print("working with the dataset")
    dataset_local_path = "./tmpdata"
    dataset_az_path = "datasets/twitter/sentiment"

    dataset_hg = _setup_dataset(model._tokenizer, dataset_local_path)

    dataset = Data(
        path=dataset_local_path,
        type=AssetTypes.URI_FOLDER,
        description="Twitter sentiment data",
        name=args.dataset_name,
    )

    ml_client.data.create_or_update(dataset)

if __name__ == "__main__":
    main()