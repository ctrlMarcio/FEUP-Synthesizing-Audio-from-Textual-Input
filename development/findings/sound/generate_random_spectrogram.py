def generate():
    import numpy as np

    # Generate 512 random frequency values
    freqs = np.random.uniform(low=0, high=1000, size=(512,))

    # Generate 512 random time values 
    times = np.linspace(0, 1, 512) 

    # Initialize empty spectrogram array
    spec = np.zeros((512, 512))

    # Loop through frequencies and times to generate random intensities 
    for f in range(len(freqs)):
        for t in range(len(times)):
            # Generate random intensity between 0 and 1
            intensity = np.random.uniform()
            
            # Set intensity at this frequency and time
            spec[f, t] = intensity
            
    # Write spectrogram to file
    np.save("random_spec.npy", spec)

if __name__ == "__main__":
    generate()