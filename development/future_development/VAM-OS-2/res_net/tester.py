# %%
import sys
sys.path.append('..')

# %%
# Import ResNet and ResBlock classes from the res_net package
from res_net.res_net import ResNet
from res_net.res_block import ResBlock

# Import the get_single_sample function from the data package
from data.tools import get_single_sample

def verify_output():
    # Instantiate the model
    model = ResNet(in_channels=1, resblock=ResBlock, repeat=[1, 1, 1, 1])

    # Get a random spectrogram from the data
    spectrogram = get_single_sample()

    # Pass the input through the model
    output = model(spectrogram)

    # Print the output shape
    print(output.shape)

# %%
# Call the verify_output function to see the output shape
verify_output()