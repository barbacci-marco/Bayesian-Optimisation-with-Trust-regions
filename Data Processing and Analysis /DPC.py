# Section 1: Import Required Libraries
# -------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Section 2: Load and Extract Calibration Data from Excel
# -------------------------------------------------------
data = pd.read_csv('Raw_data_BOP.csv')

# **Extract the last data point as validation data**
validation_data = data.iloc[-1]  # This extracts the last row as a Series

# **Remove the validation data from the main dataset**
data = data.iloc[:-1]  # Exclude the last row from the main DataFrame

# Extract data columns for analysis from the main dataset.
sample_number = data['Sample number'].values                      # Sample identifiers
Methanol_per_cal = data['Methanol%'].values                       # Calibration methanol percentage
Methanol_den_cal = data['Density (g·cm-3)'].values                # Density for calibration of methanol purity
Methanol_den_real = data['Sample density(g·cm-3)'].values         # Density values for purity interpolation
WCondensor_T_Out = data['Condensor T out'].values                 # Condenser outlet temperature
WCondensor_T_in = data['Condensor T in '].values                  # Condenser inlet temperature
Avg_Temp = data['average temperature'].values                     # Average temperature in Celsius
Dist_flowrate = data['Distilate flowrate'].values * (60 / 1000)   # Convert ml/min to L/h
R_flowrate = data['reflux flow rate '].values * (60 / 1000)       # Convert ml/min to L/h
vol_flow_rate_W = data['Flowrate of water'].values * 60           # Convert L/min to L/h
pump_power_per = data['Pump Power(%)'].values

# Convert average temperature to Kelvin
Avg_Temp_K = Avg_Temp + 273.15  # Kelvin conversion for average temperature
cp_W = 4.200  # kJ/kg·K, specific heat capacity of water

# Calculate Time to Reach 1.6 Liters of Distillate
# -------------------------------------------------------------
target_volume_l = 1.6  # Target volume in liters

# Calculate time to reach target volume based on distillate flow rate.
time_to_reach_volume = target_volume_l / Dist_flowrate  # Time in hours

# Section 3: Methanol Purity Interpolation Setup
# -----------------------------------------------
# Create an interpolator to convert sample density values into methanol percentages
purity_interpolator = interp1d(Methanol_den_cal, Methanol_per_cal, kind='linear', fill_value="extrapolate")

# Apply interpolator to sample densities to obtain methanol purity percentages
Purity = purity_interpolator(Methanol_den_real) / 100  # Convert to fraction (0 to 1)

# Section 5: Calculate Heat Transfer for Condenser and Reboiler
# -------------------------------------------------------------
# Condenser Duty calculation (in kW·h for the operation period)
# Assuming water density is 1 kg/L
Condenser_Duty = (vol_flow_rate_W * (WCondensor_T_Out - WCondensor_T_in) * cp_W * time_to_reach_volume)  # kW·h

# Reboiler Duty is fixed at 3 kW, with duty in kW·h
Reboiler_Duty = 3 * time_to_reach_volume  # kW·h for the operation period

# Pump power outage calculation
pump_power_outage = (1 * pump_power_per / 100) * time_to_reach_volume  # kW·h

# Section 6: Save Output Data for Analysis
# -----------------------------------------
output_data = pd.DataFrame({
    'Sample Number': sample_number,
    'Sample Density (g/cm³)': Methanol_den_real,
    'Methanol Purity (%)': Purity * 100,  # Convert back to percentage
    'Water Cooling Heat Transfer (kW·h)': Condenser_Duty,
    'Electric Reboiler Duty (kW·h)': Reboiler_Duty,
    'Reboiler Temperature (K)': Avg_Temp_K,
    'Distillate Flowrate (L/h)': Dist_flowrate,
    'Reflux Flowrate (L/h)': R_flowrate,
    'Time to Reach 1.6 L (hours)': time_to_reach_volume,
    'Pump Power Outage for 1.6L Distillate (kW·h)': pump_power_outage
})

# **Process the validation data separately**
validation_output_data = pd.DataFrame({
    'Sample Number': [validation_data['Sample number']],
    'Sample Density (g/cm³)': [validation_data['Sample density(g·cm-3)']],
    'Methanol Purity (%)': [purity_interpolator(validation_data['Sample density(g·cm-3)'])],
    'Water Cooling Heat Transfer (kW·h)': [
        (validation_data['Flowrate of water'] * 60 *
         (validation_data['Condensor T out'] - validation_data['Condensor T in ']) *
         cp_W * (target_volume_l / (validation_data['Distilate flowrate'] * (60 / 1000))))
    ],
    'Electric Reboiler Duty (kW·h)': [
        3 * (target_volume_l / (validation_data['Distilate flowrate'] * (60 / 1000)))
    ],
    'Reboiler Temperature (K)': [validation_data['average temperature'] + 273.15],
    'Distillate Flowrate (L/h)': [validation_data['Distilate flowrate'] * (60 / 1000)],
    'Reflux Flowrate (L/h)': [validation_data['reflux flow rate '] * (60 / 1000)],
    'Time to Reach 1.6 L (hours)': [
        target_volume_l / (validation_data['Distilate flowrate'] * (60 / 1000))
    ],
    'Pump Power Outage for 1.6L Distillate (kW·h)': [
        (1 * validation_data['Pump Power(%)'] / 100) *
        (target_volume_l / (validation_data['Distilate flowrate'] * (60 / 1000)))
    ]
})

# Export DataFrame to CSV for record-keeping and external analysis
output_data.to_csv('Lab_distillation_data.csv', index=False)

# **Optionally, save the validation data to a separate CSV**
validation_output_data.to_csv('Validation_data.csv', index=False)

# Section 7: Display Results
# --------------------------
print("\nProcessed Data (without validation data):")
print(output_data)
print("\nValidation Data:")
print(validation_output_data)

# Section 8: Data Science Plots using Matplotlib
# ----------------------------------------------

# Use the output_data for plotting (excluding the validation data)
# 1. Scatter Plot of Methanol Purity vs. Sample Density
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    output_data['Sample Density (g/cm³)'],
    output_data['Methanol Purity (%)'],
    c=output_data['Methanol Purity (%)'],
    cmap='viridis',
    edgecolor='black'
)
plt.colorbar(scatter, label='Methanol Purity (%)')
plt.title('Methanol Purity vs. Sample Density')
plt.xlabel('Sample Density (g/cm³)')
plt.ylabel('Methanol Purity (%)')
plt.grid(True)
plt.show()

# Continue with other plots using output_data

# 2. Scatter Plot of Methanol Purity vs. Total Energy Consumption
output_data['Total Energy Consumption (kW·h)'] = (
    output_data['Water Cooling Heat Transfer (kW·h)'] +
    output_data['Electric Reboiler Duty (kW·h)'] +
    output_data['Pump Power Outage for 1.6L Distillate (kW·h)']
)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    output_data['Total Energy Consumption (kW·h)'],
    output_data['Methanol Purity (%)'],
    c=output_data['Methanol Purity (%)'],
    cmap='viridis',
    edgecolor='black'
)
plt.colorbar(scatter, label='Methanol Purity (%)')
plt.title('Methanol Purity vs. Total Energy Consumption')
plt.xlabel('Total Energy Consumption (kW·h)')
plt.ylabel('Methanol Purity (%)')
plt.grid(True)
plt.show()

# 3. Plot of Reflux Ratio vs. Methanol Purity
output_data['Reflux Ratio'] = output_data['Reflux Flowrate (L/h)'] / output_data['Distillate Flowrate (L/h)']
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    output_data['Reflux Ratio'],
    output_data['Methanol Purity (%)'],
    c=output_data['Reflux Ratio'],
    cmap='cool',
    edgecolor='black'
)
plt.colorbar(scatter, label='Reflux Ratio')
plt.title('Reflux Ratio vs. Methanol Purity')
plt.xlabel('Reflux Ratio')
plt.ylabel('Methanol Purity (%)')
plt.grid(True)
plt.show()

# 4. Correlation Matrix Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = output_data.select_dtypes(include=[np.number]).corr()

# Heatmap using imshow
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')

# Add labels
labels = corr_matrix.columns
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# 5. 3D Scatter Plot of Methanol Purity vs. Distillate and Reflux Flow Rates
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(
    output_data['Distillate Flowrate (L/h)'],
    output_data['Reflux Flowrate (L/h)'],
    output_data['Methanol Purity (%)'],
    c=output_data['Methanol Purity (%)'],
    cmap='viridis',
    s=100,
    edgecolor='black'
)
ax.set_xlabel('Distillate Flowrate (L/h)')
ax.set_ylabel('Reflux Flowrate (L/h)')
ax.set_zlabel('Methanol Purity (%)')
ax.set_title('Methanol Purity vs. Distillate and Reflux Flow Rates')
fig.colorbar(p, ax=ax, shrink=0.5, aspect=5, label='Methanol Purity (%)')
plt.show()
