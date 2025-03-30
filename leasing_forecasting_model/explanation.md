## Model Training & Feature Selection

- Merged four datasets: Leases, Occupancy, Price & Availability, and Unemployment using common keys (`year`, `quarter`, `market`, `state`).
- Aggregated data by `year`, `quarter`, `market`, and `internal_class` to get one observation per market-quarter-class combination.
- Selected numeric predictors (e.g., RBA, available space, rent metrics, occupancy measures) along with the target variable (`leasing`).
- Removed rows with missing values.
- Split data into training and test sets.
- Applied bidirectional stepwise feature selection (with cross-validation) to maximize R^2 on the training data.
- Trained a final linear regression model using the selected predictors.
- Evaluated model performance on the test set using:
  - R^2 (variance explained)
  - MSE (prediction error)

## 2025 Forecasting

- Forecast built directly on the trained linear regression model.
- For each selected predictor:
  - Grouped historical data by year and computed yearly means.
  - Fitted a simple linear regression (predictor ~ year) to project 2025 values.
  - Used the most recent value if insufficient data was available.
- Compiled forecasted predictors into a future scenario DataFrame matching the model structure.
- Predicted 2025 leasing values using the trained model.
- This approach extends our trained model naturally without making a separate forecasting model.