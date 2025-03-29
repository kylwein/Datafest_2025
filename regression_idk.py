import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


'''
I wanted to regression analysis to see how leasing activity (square feet) depends on market conditions,
assuming that the most reasonable predictors would be occupancy and unemployment.
I thought it would be cool to make an interaction term between occupancy and local unemployment.

Regression Formula:
    leasing = β0 + β1 * available_space + β2 * overall_rent + β3 * avg_occupancy_proportion 
             + β4 * unemployment_rate + β5 * (avg_occupancy_proportion * unemployment_rate) + ε
             
Predictors:
    - available_space:  rentable space in the market
    - overall_rent: weighted average rent
    - avg_occupancy_proportion: Average occupancy (from occupancy data)
    - unemployment_rate: Local unemployment rate (from unemployment data)
    - occupancy_unemp: Interaction term = avg_occupancy_proportion * unemployment_rate.

Our Dependent Var: Total leased square feet, representing market-level leasing activity.
'''

price_df = pd.read_csv("Price and Availability Data.csv")
occupancy_df = pd.read_csv("Major Market Occupancy Data-revised.csv")
unemp_df = pd.read_csv("Unemployment.csv")
leases_df = pd.read_csv("Leases.csv")  # for state info

merged_df = pd.merge(price_df, occupancy_df, on=["market", "year", "quarter"], how="inner")
state_map = leases_df[["market", "state", "year", "quarter"]].drop_duplicates()
merged_df = pd.merge(merged_df, state_map, on=["market", "year", "quarter"], how="left")

# added state to merged df
print("Columns in merged_df:", merged_df.columns.tolist())

# merge unemployment_rate on year and state
unemp_subset = unemp_df[['year', 'state', 'unemployment_rate']]
merged_df = pd.merge(merged_df, unemp_subset, on=["year", "state"], how="left")

# Drop rows with no vals
reg_df = merged_df[['leasing', 'available_space', 'overall_rent',
                      'avg_occupancy_proportion', 'unemployment_rate']].dropna()

# occupancy and unemployment interaction
reg_df['occupancy_unemp'] = reg_df['avg_occupancy_proportion'] * reg_df['unemployment_rate']

X = reg_df[['available_space', 'overall_rent', 'avg_occupancy_proportion',
            'unemployment_rate', 'occupancy_unemp']]
y = reg_df['leasing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Leasing vs. Market Conditions")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.title("Impact of Occupancy and Unemployment on Leasing Activity")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# ----------------------------------------------------------------------------------
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

print('\nIs stepwise better?')


'''
Does stepwise give better results?
'''


candidate_predictors = [
    'available_space',
    'RBA',
    'availability_proportion',
    'overall_rent',
    'direct_available_space',
    'direct_availability_proportion',
    'sublet_available_space',
    'sublet_availability_proportion',
    'avg_occupancy_proportion',
    'unemployment_rate'
]

reg_df = merged_df[candidate_predictors + ['leasing']].dropna()

X = reg_df[candidate_predictors]
y = reg_df['leasing']

sfs = SFS(
    estimator=LinearRegression(),
    k_features='best',
    forward=True,        # forward first
    floating=True,       # Allow STEPWISE
    scoring='r2',
    cv=5,
    n_jobs=-1
)

sfs.fit(X, y)

best_features = list(sfs.k_feature_names_)


# MODEL WITH ONLY BEST FEATURES
X_selected = X[best_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=123)
final_model = LinearRegression()
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\nTrue Model")
print("Top Features:", best_features)
print("R^2:", test_r2)
print("MSE:", test_mse)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("STEPWISE Impact of Occupancy and Unemployment on Leasing Activity")
plt.xlabel("Actual Leasing")
plt.ylabel("Predicted Leasing")
plt.legend()
plt.show()

#---------------------------------------------------------
'''
The outlier clump is manhattan obviously.... 

proof:
'''


analysis_df = X_test.copy()
analysis_df['actual_leasing'] = y_test
analysis_df['predicted_leasing'] = y_pred
analysis_df['residual'] = analysis_df['actual_leasing'] - analysis_df['predicted_leasing']
analysis_df['abs_error'] = analysis_df['residual'].abs()
analysis_df['pct_error'] = analysis_df['abs_error'] / analysis_df['actual_leasing']

largest_residuals = analysis_df.sort_values(by='abs_error', ascending=False).head(10)


outlier_indices = largest_residuals.index
outliers_full_info = merged_df.loc[outlier_indices]

market_summary = outliers_full_info.groupby('market')['leasing'].describe()
print(market_summary)
