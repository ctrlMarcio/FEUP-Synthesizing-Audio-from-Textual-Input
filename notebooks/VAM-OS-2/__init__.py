import res_net
import text_encoder
import trainer
import clap
from vamgen.diffusion.unet import test
from vamgen.diffusion.train import train_diffusion_model

def _run_audio_encoder():
    res_net.main()


def _run_text_encoder():
    text_encoder.main()

def _train_clap():
    clap.main()

def __main__(args=None):
    train_diffusion_model()


if __name__ == "__main__":
    __main__()
