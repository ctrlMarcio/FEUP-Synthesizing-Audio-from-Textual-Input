import torch
from transformers import AutoTokenizer, AutoModel


class Model:
    """
    This class represents a framework to the text transformer encoder
    """

    def __init__(self):
        self.model_name = "bert-base-uncased"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def encode(self, text):
        input_ids = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        input_tensor = torch.tensor([input_ids])
        with torch.no_grad():
            encoded_output = self.model(input_tensor)[0]
            flatten_output = torch.flatten(encoded_output, start_dim=1)
        return flatten_output

    def encode_batch(self, texts):
        """
        Encodes a batch of texts into a batch of vectors

        Args:
            texts (list): a list of texts to encode
        """
        # verifies if texts is a list
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")

        encoded_inputs = self.tokenizer(texts, add_special_tokens=True)
        input_tensor = torch.tensor(encoded_inputs["input_ids"])
        with torch.no_grad():
            encoded_output = self.model(input_tensor)[0]
            flatten_output = torch.flatten(encoded_output, start_dim=1)
        return flatten_output
