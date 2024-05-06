import matplotlib.pyplot as plt
import numpy as np

def show(X_grid, Y_grid, delta_grid_X, delta_grid_Y, g):
    # Calculate magnitude of rate of change
    magnitude = np.sqrt(delta_grid_X**2 + delta_grid_Y**2)

    # Apply logarithmic scale to magnitude
    magnitude = np.log10(magnitude + 1)  # Adding 1 to avoid log(0)

    # Create the streamplot with colormap representing magnitude of rate of change
    plt.figure(figsize=(14, 8))
    stream = plt.streamplot(X_grid, Y_grid, delta_grid_X, delta_grid_Y, color=magnitude, cmap='viridis', density=2, minlength=0.01, linewidth=2)

    # Add colorbar
    plt.colorbar(label='Response curve - log10')

    # Plot formatting
    plt.xlabel('Soil depth ($m$)')
    plt.ylabel('Biomass ($kg/m^2$)')
    plt.title(f'Streamplot of D and B - g = {g}')
    plt.grid(True)
    plt.show()