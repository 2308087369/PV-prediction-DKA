import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = r'D:\Pythonworkshop\PV_DKA\all_model_predictions.csv'
data = pd.read_csv(file_path)
data[data < 1] = 0.01
# Assume the first column is the actual values, and the following columns are predictions
actual_values = data.iloc[:, 0]
predictions = data.iloc[:, 1:]

# Initialize a dictionary to store the results
results = {}

# Calculate the baseline model (shifted by one step)
baseline_model = actual_values.shift(1).fillna(method='bfill')  # fill the first NaN with the first actual value

# Calculate MAE, RMSE, SS-RMSE, and R² for each model
for model_name in predictions.columns:
    model_predictions = predictions[model_name]
    
    mae = mean_absolute_error(actual_values, model_predictions)
    rmse = mean_squared_error(actual_values, model_predictions, squared=False)
    baseline_rmse = mean_squared_error(actual_values, baseline_model, squared=False)
    ss_rmse = 1 - (rmse / baseline_rmse)
    r2 = r2_score(actual_values, model_predictions)
    
    results[model_name] = {
        'MAE': mae,
        'RMSE': rmse,
        'SS-RMSE': ss_rmse,
        'R²': r2
    }

# Convert the results to a DataFrame for easier viewing
results_df = pd.DataFrame(results).T
print(results_df)
