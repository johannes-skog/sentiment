
from typing import Dict, List, Tuple
import numpy as np
from io import BytesIO
import os
from ts.torch_handler.base_handler import BaseHandler
import torch
from transformers import XLMRobertaTokenizerFast


class SentimentHandler(BaseHandler):
    
    def __init__(self):

        super().__init__()
        self.initialized = False

        self._received_size = None
        
    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        self.manifest = context.manifest
        print(context.manifest)
        
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self._context = context

        self.model = torch.jit.load(
            os.path.join(
                properties.get("model_dir"),
                "traced.pt",
            )
        )
        
        self.model.eval()

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(".")
        
        print(self.model)
        print(self.tokenizer)
        
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        
        self.model.to(self.device)
        
        self.initialized = True

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """

        probs = data.flatten().tolist()

        i = 0

        ret = []

        for _, s in enumerate(self._received_size):
            d = []
            for _ in range(s):
                d.append(probs[i])
                i += 1                
            ret.append(d)

        return ret

    def preprocess(self, requests: List[Dict[str, bytearray]]):
        """
        Function to prepare data from the model
        
        :param requests:
        :return: tensor of the processed shape specified by the model
        """
        input_ids_batch = None
        attention_mask_batch = None

        self._received_size = []

        input_texts = []

        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            print("INPUT_TEXT", input_text)

            self._received_size.append(len(input_text))

            input_texts.extend(input_text)

        inputs = self.tokenizer(
            input_texts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True
        )

        input_ids_batch = inputs["input_ids"].to(self.device)
        attention_mask_batch = inputs["attention_mask"].to(self.device)
        
        return input_ids_batch, attention_mask_batch
        
    def inference(self, model_input):
        """
        Given the data from .preprocess, perform inference using the model.
        
        :param reqeuest:
        :return: Logits or predictions given by the model
        """
        input_ids_batch, attention_mask_batch = model_input

        print(input_ids_batch.shape)

        with torch.no_grad():

            logits = self.model(
                input_ids_batch.to(self.device),
                attention_mask=attention_mask_batch.to(self.device)
            )

            print(logits)

        probs = torch.sigmoid(logits)
        
        return probs 
