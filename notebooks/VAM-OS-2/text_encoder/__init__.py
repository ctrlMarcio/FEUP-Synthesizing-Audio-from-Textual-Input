from text_encoder.encoder import Model

class _ModelSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance
    
    def __init__(self):
        self.model = Model()

_model_singleton = _ModelSingleton()

def encode(text_input):
    """
    This function receives text and returns the encoded text
    """
    return _model_singleton.model.encode(text_input)

def load():
    """
    This function should load the trained model
    """
    return Model()

def main():
    model = load()
    print(model.encode_batch(["Hello, world!", "Hello, world!"]))
