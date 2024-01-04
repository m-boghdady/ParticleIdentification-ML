import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data from the four .npy files for the x-axis
X = np.load('GammaA;5_eta.npy')

# Load data from the single .npy file for the y-axis
y = np.load('GammaA;5_pt.npy')

print("X shape:", X.shape)
print("y shape:", y.shape)


# Combine the x-axis data (X) and y-axis data (y) into a single array
combined_data = np.column_stack((X, y))

# Save the combined data into a single .npy file
np.save('combined_eta:pt.npy', combined_data)



