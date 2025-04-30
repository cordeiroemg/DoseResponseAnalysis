import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for the logistic function
max_mortality = 10  # Maximum mortality
mid_concentration = 0.01  # Shifted concentration at which mortality is 50%
steepness = 7  # Adjust steepness if needed

# Generate a range of concentration values (logarithmically spaced)
concentrations = np.logspace(-1, 2, num=10)  # Now from 0.001 to 10 mg/mL

# Logistic function to simulate mortality
def logistic_function(concentration, max_mortality, mid_concentration, steepness):
    return max_mortality / (1 + np.exp(-steepness * (concentration - mid_concentration)))

# Simulate mortality data
mortality = logistic_function(concentrations, max_mortality, mid_concentration, steepness)

# Add noise to the mortality data
# Noise parameters
noise_level = 0.1  # Standard deviation of the noise
noise = np.random.normal(0, noise_level, size=mortality.shape)  # Generate random noise

# Apply noise to mortality values and ensure they are within [0, 100]
mortality = np.clip(mortality + noise, 0, 100)

# Check the percentage of mortality values within the range of 20% to 80%
within_range = np.sum((mortality >= 20) & (mortality <= 80)) / len(mortality) * 100
print(f"Percentage of mortality data within 20% to 80%: {within_range:.2f}%")


# Creating a DataFrame to match the input format
data = {
    'btProtein': ['Protein'] * len(concentrations),
    'population': ['Pop'] * len(concentrations),
    'concentration (ppm)': concentrations,
    'mortality': mortality
}

df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('simulated_mortality_data_RRppm.csv', index=False)

# Plotting the S-curve with noise
plt.figure(figsize=(8, 5))
plt.plot(concentrations, mortality, marker='o', label='Observed Mortality')
plt.plot(concentrations, mortality, linestyle='--', color='gray', label='True Mortality Curve')
plt.xscale('log')  # Log scale for concentration
plt.xlabel('Concentration (mg/mL)')
plt.ylabel('Mortality (%)')
plt.title('Simulated Mortality vs Concentration with Noise (Shifted Curve)')
plt.grid()
plt.axhline(y=100, color='r', linestyle='--', label='Max Mortality')
plt.axvline(x=mid_concentration, color='g', linestyle='--', label='Mid Concentration (50% Mortality)')
plt.legend()
plt.show()
