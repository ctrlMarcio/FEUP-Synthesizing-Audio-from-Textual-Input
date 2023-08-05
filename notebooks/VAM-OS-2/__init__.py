import res_net
import text_encoder
import trainer
import clap.training as training
from vamgen.diffusion.unet import test
from vamgen import hardcode

def _run_audio_encoder():
    res_net.main()


def _run_text_encoder():
    text_encoder.main()

def _train_clap():
    training.main()

def __main__(args=None):
    hardcode.fit()


if __name__ == "__main__":
    __main__()
