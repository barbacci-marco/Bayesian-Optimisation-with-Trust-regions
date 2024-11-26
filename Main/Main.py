from GPBO import bayesian_optimization_with_trust_region
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

np.random.seed(16)
# Step 1: Read Data from CSV Files
# -------------------------------
# Read the main data excluding validation data
data = pd.read_csv('/Users/marcobarbacci/Y3 Labs BOPT /Year 3 distilattion optimisation/Data/Lab_distillation_data.csv')

# Read the validation data
validation_data = pd.read_csv('/Users/marcobarbacci/Y3 Labs BOPT /Year 3 distilattion optimisation/Data/Validation_data.csv')

# Step 2: Extract Data Columns from Main Data
# --------------------------------------------
# Extract data for the main dataset
purities_percent = data['Methanol Purity (%)'].values
purities = purities_percent / 100  # Convert to fraction
Distillate_flowrate = data['Distillate Flowrate (L/h)'].values
Reflux_flowrate = data['Reflux Flowrate (L/h)'].values
Time = data['Time to Reach 1.6 L (hours)'].values
Pump_power_Outage = data['Pump Power Outage for 1.6L Distillate (kW·h)'].values
Condenser_Duty = data['Water Cooling Heat Transfer (kW·h)'].values
Reboiler_Duty = data['Electric Reboiler Duty (kW·h)'].values

# Step 3: Extract Data Columns from Validation Data
# -------------------------------------------------
validation_purity_percent = validation_data['Methanol Purity (%)'].values
validation_purity = validation_purity_percent / 100
validation_Distillate_flowrate = validation_data['Distillate Flowrate (L/h)'].values
validation_Reflux_flowrate = validation_data['Reflux Flowrate (L/h)'].values
validation_Time = validation_data['Time to Reach 1.6 L (hours)'].values
validation_Pump_power_Outage = validation_data['Pump Power Outage for 1.6L Distillate (kW·h)'].values
validation_Condenser_Duty = validation_data['Water Cooling Heat Transfer (kW·h)'].values
validation_Reboiler_Duty = validation_data['Electric Reboiler Duty (kW·h)'].values

# Step 4: Calculate Total Energy Consumption per Batch
# ----------------------------------------------------
# Total energy consumption per batch (kWh per batch) for main data
energy_consumptions = Reboiler_Duty + Condenser_Duty + Pump_power_Outage

# Total energy consumption for validation data
validation_energy_consumption = (
    validation_Reboiler_Duty +
    validation_Condenser_Duty +
    validation_Pump_power_Outage
)

# Step 5: Prepare Input Data for Polynomial Regression
# ----------------------------------------------------
inputs = np.column_stack((Distillate_flowrate, Reflux_flowrate))

# Ensure that the number of input pairs matches the number of purity measurements.
if inputs.shape[0] != purities.shape[0]:
    raise ValueError("Mismatch between number of input points and purity measurements.")

# Check for NaNs in inputs and outputs
valid_indices = (
    ~np.isnan(inputs).any(axis=1) &
    ~np.isnan(purities) &
    ~np.isnan(energy_consumptions)
)

# Filter out invalid data
inputs = inputs[valid_indices]
purities = purities[valid_indices]
purities_percent = purities_percent[valid_indices]
energy_consumptions = energy_consumptions[valid_indices]
Distillate_flowrate = Distillate_flowrate[valid_indices]
Reflux_flowrate = Reflux_flowrate[valid_indices]
Time = Time[valid_indices]

# Step 6: Create Polynomial Features and Fit Polynomial Regression Models
# -----------------------------------------------------------------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features up to degree 3
poly = PolynomialFeatures(degree=3, include_bias=False)
inputs_poly = poly.fit_transform(inputs)

# Fit the polynomial regression model for purity using main data
purity_model = LinearRegression()
purity_model.fit(inputs_poly, purities)

# Fit the polynomial regression model for energy consumption
energy_model = LinearRegression()
energy_model.fit(inputs_poly, energy_consumptions)

# Step 7: Define the Purity Function and Energy Function Using the Fitted Models
# ------------------------------------------------------------------------------
def purity_function(Distillate_flowrate, Reflux_flowrate):
    # Ensure inputs are numpy arrays
    Distillate_flowrate = np.asarray(Distillate_flowrate)
    Reflux_flowrate = np.asarray(Reflux_flowrate)

    # Flatten the input arrays to 1D
    D_flat = Distillate_flowrate.ravel()
    R_flat = Reflux_flowrate.ravel()

    # Stack the flattened inputs column-wise
    input_data = np.column_stack((D_flat, R_flat))

    # Transform the input data
    input_data_poly = poly.transform(input_data)

    # Predict the purity
    purity_flat = purity_model.predict(input_data_poly)

    # Reshape the output to match the input shape
    purity = purity_flat.reshape(Distillate_flowrate.shape)

    # Clip purity values to [0, 1]
    purity = np.clip(purity, 0, 1)

    return purity

def energy_function(Distillate_flowrate, Reflux_flowrate):
    # Ensure inputs are numpy arrays
    Distillate_flowrate = np.asarray(Distillate_flowrate)
    Reflux_flowrate = np.asarray(Reflux_flowrate)

    # Flatten the input arrays to 1D
    D_flat = Distillate_flowrate.ravel()
    R_flat = Reflux_flowrate.ravel()

    # Stack the flattened inputs column-wise
    input_data = np.column_stack((D_flat, R_flat))

    # Transform the input data
    input_data_poly = poly.transform(input_data)

    # Predict the energy consumption
    energy_flat = energy_model.predict(input_data_poly)

    # Reshape the output to match the input shape
    energy = energy_flat.reshape(Distillate_flowrate.shape)

    # Ensure energy consumption is non-negative
    energy = np.maximum(energy, 0)

    return energy

# Step 8: Define the Pricing Function Based on Purity
# ---------------------------------------------------
def price_of_methanol(purity):
    if purity < 0.85:
        return 0  # Price is £0 for purity below 85%
    else:
        # Exponential pricing function
        k = 25  # Adjusted k for better curve fitting
        numerator = 1 - np.exp(-k * (purity - 0.85))
        denominator = 1 - np.exp(-k * (0.9985 - 0.85))
        price = 23 * 26 * (numerator / denominator)
        return price  # Price in £ per liter

# Step 9: Compute Reflux Ratio from Existing Data
# -----------------------------------------------
reflux_ratio = Reflux_flowrate / Distillate_flowrate

# Reflux ratio for validation data
validation_reflux_ratio = validation_Reflux_flowrate / validation_Distillate_flowrate

# Step 10: Calculate Objective Values for Existing Data
# ----------------------------------------------------
objective_values = []
electricity_cost_per_kWh = 0.04  # £ per kWh

for i in range(len(purities)):
    purity = purities[i]
    energy_per_batch = energy_consumptions[i]  # kWh per batch
    distillate_flow_rate = Distillate_flowrate[i]  # L/h

    # Calculate the price of methanol based on purity
    price = price_of_methanol(purity)  # £ per liter

    # Revenue per batch
    revenue = price * 1.6  # £ per batch

    # Energy cost per batch
    energy_cost = energy_per_batch * electricity_cost_per_kWh  # £ per batch

    # Profit per batch
    profit = revenue - energy_cost

    # Append the negative profit to objective_values
    objective_values.append(-profit)  # Negative for consistency with plotting

objective_values = np.array(objective_values)

# *Calculate objective value for validation data*
validation_purity = validation_purity[0]
validation_energy_per_batch = validation_energy_consumption[0]  # kWh per batch
validation_distillate_flow_rate = validation_Distillate_flowrate[0]  # L/h

# Calculate the price of methanol based on purity
validation_price = price_of_methanol(validation_purity)  # £ per liter

# Revenue per batch
validation_revenue = validation_price * 1.6  # £ per batch

# Energy cost per batch
validation_energy_cost = validation_energy_per_batch * electricity_cost_per_kWh  # £ per batch

# Profit per batch
validation_profit = validation_revenue - validation_energy_cost

# Negative profit for consistency with objective function
validation_objective_value = -validation_profit
X = Reflux_flowrate, Distillate_flowrate

# Step 11: Define the Cost Function
# ---------------------------------
def cost_function(X):
    reflux_flow_rate = X[0]
    distillate_flow_rate = X[1]
    # Compute Reflux_flowrate based on reflux_ratio and distillate_flow_rate
    reflux_ratio= reflux_flow_rate / distillate_flow_rate

    # Predict purity and energy consumption
    purity = purity_function(distillate_flow_rate, reflux_flow_rate)
    energy_per_batch = energy_function(distillate_flow_rate, reflux_flow_rate)

    # Handle cases where the predictions are invalid
    if np.isnan(purity).any() or np.isnan(energy_per_batch).any():
        return 1e8  # Assign a large cost to invalid inputs

    # Energy cost per batch
    energy_cost = energy_per_batch * electricity_cost_per_kWh  # £ per batch

    # Revenue per batch
    price = price_of_methanol(purity)  # £ per liter
    revenue = price * 1.6  # £ per batch

    # Total profit per batch
    profit = revenue - energy_cost

    # If profit is an array (due to array inputs), take the mean
    if np.size(profit) > 1:
        profit = np.mean(profit)

    # Return negative profit to convert to a minimization problem
    objective_value = profit

    return objective_value

# Step 12: Define the Search Space for Optimization
# -------------------------------------------------
# Define bounds for reflux ratio and distillate flow rate based on your data
reflux_ratio_min = reflux_ratio.min()
reflux_ratio_max = reflux_ratio.max()
distillate_flow_rate_min = Distillate_flowrate.min()
distillate_flow_rate_max = Distillate_flowrate.max()

search_space = np.array([
    [reflux_ratio_min, reflux_ratio_max],          # reflux_ratio bounds
    [distillate_flow_rate_min, distillate_flow_rate_max]  # distillate_flow_rate bounds
])


# Step 13: Run Bayesian Optimization
# ----------------------------------
# Run Bayesian Optimization with Trust Regions
X_res, Y_res, iterations = bayesian_optimization_with_trust_region(
    n_iters=30,
    sample_loss= cost_function,
    bounds=search_space,
    n_pre_samples=5,
    alpha=1e-8,
    initial_trust_radius=0.5  # recommend adjusting 
)

# Step 15: Extract and Display the Optimal Parameters and Maximum Profit
# ----------------------------------------------------------------------
# Extract the optimal parameters
optimal_parameters = X_res[-1]
optimal_reflux_flow_rate = optimal_parameters[0]
optimal_distillate_flow_rate = optimal_parameters[1]

# Compute the optimal reflux ratio
optimal_reflux_ratio = optimal_reflux_flow_rate / optimal_distillate_flow_rate

maximum_profit = Y_res[-1]  # Maximum total profit (£ per batch)

print("\n")
print(f"Optimal reflux ratio: {optimal_reflux_ratio:.4f}")
print(f"Optimal distillate flow rate: {optimal_distillate_flow_rate:.4f} L/h")
print(f"Optimal reflux flow rate: {optimal_reflux_flow_rate:.4f} L/h")
print(f"Maximum profit per batch: £{maximum_profit:.2f}")

# Step 16: Calculate RMSE Between Predicted and Actual Values at Validation Data Point
# ------------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error

# Prepare the validation inputs
validation_distillate_flowrate = validation_Distillate_flowrate[0]
validation_reflux_flowrate = validation_Reflux_flowrate[0]

# Predict purity and energy consumption using the models
predicted_purity = purity_function(validation_distillate_flowrate, validation_reflux_flowrate)
predicted_energy = energy_function(validation_distillate_flowrate, validation_reflux_flowrate)

# Actual observed values
actual_purity = validation_purity
actual_energy = validation_energy_per_batch

# Calculate the prediction errors
purity_error = predicted_purity - actual_purity
energy_error = predicted_energy - actual_energy

# Calculate RMSE for purity and energy consumption
purity_rmse = np.sqrt(mean_squared_error([actual_purity], [predicted_purity]))
energy_rmse = np.sqrt(mean_squared_error([actual_energy], [predicted_energy]))

print(f"\nPurity RMSE at Validation Data Point: {purity_rmse:.4f}")
print(f"Energy Consumption RMSE at Validation Data Point: {energy_rmse:.4f} kWh")

# **Calculate Profit Prediction Error**
# Ensure electricity_cost_per_kWh is defined
if 'electricity_cost_per_kWh' not in locals():
    electricity_cost_per_kWh = 0.04  # £ per kWh

# Calculate predicted profit at validation data point
predicted_price = price_of_methanol(predicted_purity)
predicted_revenue = predicted_price * 1.6  # £ per batch
predicted_energy_cost = predicted_energy * electricity_cost_per_kWh  # £ per batch
predicted_profit = predicted_revenue - predicted_energy_cost

# Actual profit at validation data point
actual_profit = validation_profit

# Profit prediction error
profit_error = predicted_profit - actual_profit
profit_rmse = np.sqrt(mean_squared_error([actual_profit], [predicted_profit]))

print(f"Profit RMSE at Validation Data Point: £{profit_rmse:.2f}")

# Step 17: Interpret the RMSE Results
# -----------------------------------
print("\nInterpretation of RMSE Results:")
print(f"The model predicted a purity of {predicted_purity:.4f} compared to the actual purity of {actual_purity:.4f}.")
print(f"The purity prediction RMSE is {purity_rmse:.4f}, indicating a prediction error of {purity_rmse*100:.2f}% purity.")

print(f"\nThe model predicted an energy consumption of {predicted_energy:.4f} kWh compared to the actual consumption of {actual_energy:.4f} kWh.")
print(f"The energy consumption prediction RMSE is {energy_rmse:.4f} kWh.")

print(f"\nThe model predicted a profit of £{predicted_profit:.2f} compared to the actual profit of £{actual_profit:.2f}.")
print(f"The profit prediction RMSE is £{profit_rmse:.2f}, indicating the model's prediction deviates from the actual profit by £{abs(profit_error):.2f}.")

# Discuss the implications
if abs(profit_rmse) / abs(actual_profit) < 0.1:
    print(f"\nThe profit prediction error is less than 10% ({profit_rmse/abs(actual_profit)*100:.2f}%) of the actual profit, indicating good model accuracy at the validation point.")
else:
    print(f"\nThe profit prediction error is greater than 10% ({profit_rmse/abs(actual_profit)*100:.2f}%) of the actual profit, suggesting the model may not generalize well to this point.")

print("\n")

# Step 18: Visualization of Profit vs. Iterations
# -----------------------------------------------
plt.figure(figsize=(10, 6))
profit_values = np.array(Y_res)
iterations = np.arange(len(profit_values))  

plt.plot(iterations, profit_values, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Profit per Batch (£)')
plt.title('Convergence of Bayesian Optimization')
plt.grid(True)


lower_limit = np.percentile(profit_values, 1)
upper_limit = np.percentile(profit_values, 99)
margin = (upper_limit - lower_limit) * 0.05  # 5% margin
plt.ylim([lower_limit - margin, upper_limit + margin])

plt.show()

# Step 19: Visualization of Profit Function over Reflux Ratios and Distillate Flow Rates
# --------------------------------------------------------------------------------------
# Create a grid of reflux ratios and distillate flow rates
reflux_ratios_plot = np.linspace(reflux_ratio_min, reflux_ratio_max, 50)
distillate_flow_rates_plot = np.linspace(distillate_flow_rate_min, distillate_flow_rate_max, 50)
RR_grid, DFR_grid = np.meshgrid(reflux_ratios_plot, distillate_flow_rates_plot)

# Calculate profits over the grid
profit_grid = np.zeros_like(RR_grid)

for i in range(RR_grid.shape[0]):
    for j in range(RR_grid.shape[1]):
        reflux_ratio = RR_grid[i, j]
        distillate_flow_rate = DFR_grid[i, j]
        reflux_flow_rate = reflux_ratio * distillate_flow_rate  # Compute reflux flow rate
        params = [reflux_flow_rate, distillate_flow_rate]
        objective_val = cost_function(params)
        profit_grid[i, j] = -objective_val  # Convert back to profit

# Plot the profit contours
plt.figure(figsize=(12, 8))
contour = plt.contourf(RR_grid, DFR_grid, profit_grid, levels=20, cmap='viridis')
plt.colorbar(contour, label='Profit per Batch (£)')

# Plot the main data points
plt.scatter(
    reflux_ratio,
    Distillate_flowrate,
    color='white',
    edgecolor='black',
    label='Main Data'
)

# Plot the validation data point
plt.scatter(
    validation_reflux_ratio,
    validation_Distillate_flowrate,
    color='red',
    s=100,
    label='Validation Data',
    edgecolor='k'
)

# Mark the optimal point
plt.scatter(
    optimal_reflux_ratio,
    optimal_distillate_flow_rate,
    color='yellow',
    s=100,
    label='Optimal Point',
    edgecolor='k'
)

plt.xlabel('Reflux Ratio')
plt.ylabel('Distillate Flow Rate (L/h)')
plt.title('Profit Contour Plot over Reflux Ratio and Distillate Flow Rate')
plt.legend()
plt.grid(True)
plt.show()

# Step 20: Visualization of the Optimal Point with Respect to the Data
# --------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot main data
scatter = ax.scatter(
    reflux_ratio,
    Distillate_flowrate,
    -objective_values,  # Convert back to profit
    c=purities_percent,
    cmap='viridis',
    alpha=0.7,
    edgecolor='k',
    label='Main Data'
)

# Plot validation data
ax.scatter(
    validation_reflux_ratio,
    validation_Distillate_flowrate,
    -validation_objective_value,  # Convert back to profit
    color='red',
    s=100,
    label='Validation Data',
    edgecolor='k'
)

# Colorbar for purity
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Methanol Purity (%)', rotation=270, labelpad=15)

# Plot optimal point
ax.scatter(
    optimal_reflux_ratio,
    optimal_distillate_flow_rate,
    maximum_profit,
    color='yellow',
    s=100,
    label='Optimal Point',
    edgecolor='k'
)

ax.set_xlabel('Reflux Ratio')
ax.set_ylabel('Distillate Flow Rate (L/h)')
ax.set_zlabel('Profit per Batch (£)')
ax.set_title('Optimal Point in Relation to Main and Validation Data')

ax.legend()

plt.show()

# Step 21: Purity Surface Plot
# ----------------------------
# Create a grid of distillate and reflux flow rates
distillate_range = np.linspace(distillate_flow_rate_min, distillate_flow_rate_max, 50)
reflux_range = np.linspace(Reflux_flowrate.min(), Reflux_flowrate.max(), 50)
D_grid, R_grid = np.meshgrid(distillate_range, reflux_range)

# Predict purity over the grid
purity_grid = purity_function(D_grid, R_grid)

# Plot the purity surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    D_grid, R_grid, purity_grid,
    cmap='viridis',
    alpha=0.8,
    edgecolor='none'
)

# Overlay main data points
ax.scatter(
    Distillate_flowrate,
    Reflux_flowrate,
    purities,
    color='red',
    label='Main Data',
    edgecolor='k'
)

# Overlay validation data point
ax.scatter(
    validation_Distillate_flowrate,
    validation_Reflux_flowrate,
    validation_purity,
    color='blue',
    s=100,
    label='Validation Data',
    edgecolor='k'
)

ax.set_xlabel('Distillate Flow Rate (L/h)')
ax.set_ylabel('Reflux Flow Rate (L/h)')
ax.set_zlabel('Purity (Fraction)')
ax.set_title('Interpolated Purity Surface with Data Points')
ax.legend()

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Purity (Fraction)')
plt.show()

# Step 22: Energy Consumption Surface Plot Colored by Purity (Without Normalization)
# ----------------------------------------------------------------------------------

# Predict energy consumption over the grid
energy_grid = energy_function(D_grid, R_grid)

# Predict purity over the grid
purity_grid = purity_function(D_grid, R_grid)

# Import colormap
from matplotlib import cm

# Map purity values directly to colors
colors = cm.viridis(purity_grid)

# Plot the energy consumption surface colored by purity
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    D_grid, R_grid, energy_grid,
    facecolors=colors,
    alpha=0.8,
    edgecolor='none'
)

# Create a ScalarMappable and add a colorbar
from matplotlib.colors import Normalize

norm = Normalize(vmin=purity_grid.min(), vmax=purity_grid.max())

mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
mappable.set_array([])
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='Purity (Fraction)')

# Overlay main data points
ax.scatter(
    Distillate_flowrate,
    Reflux_flowrate,
    energy_consumptions,
    color='black',
    label='Main Data',
    edgecolor='k'
)

# Overlay validation data point
ax.scatter(
    validation_Distillate_flowrate,
    validation_Reflux_flowrate,
    validation_energy_consumption,
    color='red',
    s=100,
    label='Validation Data',
    edgecolor='k'
)

# Calculate optimal reflux flow rate and energy consumption
optimal_reflux_flow_rate = optimal_reflux_ratio * optimal_distillate_flow_rate
optimal_energy_consumption = energy_function(
    optimal_distillate_flow_rate, optimal_reflux_flow_rate
)

# Plot optimal point
ax.scatter(
    optimal_distillate_flow_rate,
    optimal_reflux_flow_rate,
    optimal_energy_consumption,
    color='yellow',
    s=100,
    label='Optimal Point',
    edgecolor='k'
)

ax.set_xlabel('Distillate Flow Rate (L/h)')
ax.set_ylabel('Reflux Flow Rate (L/h)')
ax.set_zlabel('Energy Consumption (kWh)')
ax.set_title('Energy Consumption Surface Colored by Purity (Without Normalization)')
ax.legend()

plt.show()
