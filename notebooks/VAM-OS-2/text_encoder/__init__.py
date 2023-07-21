from text_encoder.encoder import Model


def load():
    """
    This function should load the trained model
    """
    return Model()

def main():
    model = load()
    print(model.encode_batch(["Hello, world!", "Hello, world!"]))

# THIS COMMENT BELOW IS WHAT GPT GAVE US INITIALLY
# # Load the pre-trained BERT tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# # Load the pre-trained BERT model
# model = AutoModel.from_pretrained("bert-base-uncased")

# # Encode a string using the BERT tokenizer
# input_string = "Hello, world!"
# input_ids = tokenizer.encode(input_string, add_special_tokens=True)

# # Convert the input_ids to a PyTorch tensor
# input_tensor = torch.tensor([input_ids])

# # Pass the input tensor through the BERT model to get the encoded output
# with torch.no_grad():
#     encoded_output = model(input_tensor)[0]

# # Print the encoded output
# print(encoded_output)
