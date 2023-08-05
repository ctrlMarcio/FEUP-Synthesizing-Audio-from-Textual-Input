from clap import clap


class _ClapSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        self.clap_model = None  # Initialize clap_model to None

    def initialize_clap_model(self):
        self.clap_model = clap.Clap(2304)


_clap_singleton = _ClapSingleton()


def translate_encodings(text_encodings):
    """
    This function takes regular text_encodings and passes them through CLAP
    to generate CLAP encodings
    """
    return _clap_singleton.clap_model.encode_text(text_encodings)
