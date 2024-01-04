import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from the four .npy files for the x-axis
x1 = np.load('Data/GammaA;5_eta.npy')
x2 = np.load('Data/GammaA;5_MultAll.npy')
x3 = np.load('Data/GammaA;5_pt.npy')
x4 = np.load('Data/GammaA;5_ch.npy')

# Concatenate the x-axis data into a single 2D array
X = np.column_stack((x1, x2, x3, x4))

# Load data from the single .npy file for the y-axis
y = np.load('Data/GammaA;5_id.npy')

print("X shape:", X.shape)
print("y shape:", y.shape)


# Combine the x-axis data (X) and y-axis data (y) into a single array
combined_data = np.column_stack((X, y))

# Save the combined data into a single .npy file
np.save('combined_data_5D.npy', combined_data)

# Create a 3D scatter plot using the first three dimensions of X
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use the fourth dimension of X for color mapping and the y-axis values for size mapping
colors = plt.cm.viridis(X[:, 3])
sizes = (y - y.min()) / (y.max() - y.min()) * 100  # Normalize y values to the range [0, 100]

scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=sizes, cmap='viridis', alpha=0.7)
ax.set_xlabel('eta')
ax.set_ylabel('Mult')
ax.set_zlabel('pt')
plt.title('ch')

# Create a colorbar to show the mapping of the fourth dimension to colors
cbar = plt.colorbar(scatter)
cbar.set_label('Dimension 4')

plt.show()



