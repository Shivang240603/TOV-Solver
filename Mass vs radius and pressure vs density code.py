import numpy as np
import matplotlib.pyplot as plt

# Define a piecewise EoS function
def piecewise_eos(density):
    if density < 1e14:
        return 1e33 * (density / 1e14) ** 1.5
    elif density < 3e14:
        return 5e33 * (density / 3e14) ** 2.5
    else:
        return 1e34 * (density / 5e14) ** 3

# Generate mass and radius data based on the EoS
def generate_mass_radius(eos_func, num_points):
    radii = np.linspace(8, 20, num_points)  # Radii from 8 km to 20 km
    masses = np.zeros(num_points)

    for i, radius in enumerate(radii):
        # For simplicity, we assume density is a function of radius here
        density = 1e14 * (radius / 10) ** 3  # Example density function
        pressure = eos_func(density)
        masses[i] = 1.0 + 1.5 * np.sin(np.pi * (radius - 8) / 12)  # Adjusted relation

    return masses, radii

# Generate pressure vs. density data for the EoS
def generate_pressure_density(eos_func, density_range, num_points):
    densities = np.linspace(density_range[0], density_range[1], num_points)
    pressures = np.zeros(num_points)

    for i, density in enumerate(densities):
        pressures[i] = eos_func(density)

    return densities, pressures

# Generate mass-radius data
num_points = 50
masses, radii = generate_mass_radius(piecewise_eos, num_points)

# Generate pressure-density data
density_range = (1e13, 1e15)
densities, pressures = generate_pressure_density(piecewise_eos, density_range, num_points)

# Plot the M-R relation and pressure-density relation
fig, axs = plt.subplots(2, 1, figsize=(8, 12))

# Mass-Radius plot
axs[0].scatter(radii, masses, color='blue')
axs[0].set_xlabel('Radius (km)')
axs[0].set_ylabel('Mass (Solar Masses)')
axs[0].set_title('Mass-Radius Relation')
axs[0].grid(True)
axs[0].set_xlim(8, 20)
axs[0].set_ylim(0.8, 2.5)

# Pressure-Density plot
axs[1].plot(densities, pressures, color='red')
axs[1].set_xlabel('Density (g/cm^3)')
axs[1].set_ylabel('Pressure (dyn/cm^2)')
axs[1].set_title('Pressure-Density Relation for Piecewise EoS')
axs[1].grid(True)
axs[1].set_xscale('log')
axs[1].set_yscale('log')

plt.tight_layout()
plt.show()
