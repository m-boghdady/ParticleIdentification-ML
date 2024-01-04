
import numpy as np

# Load the data from the .npy file
data = np.load('combined_eta:pt.npy')

# Print the shape and contents of the data array
print(f"Data shape: {data.shape}")
print("Data contents:")
print(data)

