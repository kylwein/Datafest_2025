import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance


"""
Load the datasets for Leases, Occupancy, Price & Availability, and Unemployment.
"""

leases_df = pd.read_csv("../Leases.csv")
occupancy_df = pd.read_csv("../Major Market Occupancy Data-revised.csv")
price_df = pd.read_csv("../Price and Availability Data.csv")
unemployment_df = pd.read_csv("../Unemployment.csv")


"""
Merge the datasets on common keys:
- Merge leases with price data on year, quarter, and market.
- Merge occupancy data (keeping starting and average occupancy) on year, quarter, and market.
- Merge unemployment data on year, quarter, and state.
"""

df_merged = pd.merge(leases_df, price_df, on=["year", "quarter", "market"], how="left", suffixes=("", "_price"))
df_merged = pd.merge(df_merged,
                     occupancy_df[['year', 'quarter', 'market', 'starting_occupancy_proportion', 'avg_occupancy_proportion']],
                     on=["year", "quarter", "market"], how="left")
df_merged = pd.merge(df_merged, unemployment_df[['year', 'quarter', 'state', 'unemployment_rate']],
                     on=["year", "quarter", "state"], how="left")


"""
Because the leasing metric is reported at the market level for each internal class, 
we aggregate the merged data by year, quarter, market, and internal_class. This makes sure that 
each row represents a unique combination with a single leasing value.
"""

agg_cols = [
    'leasing', 'RBA', 'availability_proportion', 'available_space',
    'starting_occupancy_proportion', 'avg_occupancy_proportion',
    'direct_availability_proportion', 'direct_available_space',
    'direct_internal_class_rent', 'direct_overall_rent', 'internal_class_rent',
    'leasedSF', 'overall_rent', 'sublet_availability_proportion',
    'sublet_available_space', 'sublet_internal_class_rent', 'sublet_overall_rent',
    'unemployment_rate'
]
df_agg = df_merged.groupby(['year', 'quarter', 'market', 'internal_class'])[agg_cols].mean().reset_index()


"""
Select numeric predictors and the target (leasing). We drop rows with missing values so that
each row is complete. This allows that our target (leasing) 
and predictor variables are aligned.
"""

target = 'leasing'
numeric_predictors = [
    'RBA', 'availability_proportion', 'available_space', 'starting_occupancy_proportion',
    'avg_occupancy_proportion', 'direct_availability_proportion', 'direct_available_space',
    'direct_internal_class_rent', 'direct_overall_rent', 'internal_class_rent', 'leasedSF',
    'overall_rent', 'sublet_availability_proportion', 'sublet_available_space',
    'sublet_internal_class_rent', 'sublet_overall_rent', 'unemployment_rate'
]
predictors = [col for col in numeric_predictors if col in df_agg.columns]

df_model = df_agg[[target] + predictors].dropna().copy()
for col in predictors + [target]:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
df_model = df_model.dropna().reset_index(drop=True)


"""
Split the data into training and test sets (80/20 split) and run stepwise feature selection 
using Linear Regression. This identifies the best subset of predictors based on cross-validated R^2/
"""

X = df_model[predictors]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

lr = LinearRegression()
sfs = SFS(lr,
          k_features='best',
          forward=True,
          floating=True,
          scoring='r2',
          cv=5,
          n_jobs=-1)
sfs = sfs.fit(X_train, y_train)
best_features = list(sfs.k_feature_names_)
print("Selected Features:", best_features)
print("Cross-validated R^2 Score for best subset (from SFS):", sfs.k_score_)

X_train_selected = X_train[best_features]
X_test_selected = X_test[best_features]
lr.fit(X_train_selected, y_train)
y_pred = lr.predict(X_test_selected)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Final Model R^2 on test set:", r2)
print("Final Model MSE on test set:", mse)

# 5 fold again (better safe then sorry
kf = KFold(n_splits=5, shuffle=True, random_state=100)
cv_scores = cross_val_score(lr, X_train_selected, y_train, scoring='r2', cv=kf)
print("Additional 5-fold CV R^2 Scores:", cv_scores)
print("Mean CV R^2 on training set:", np.mean(cv_scores))


"""
Plot a scatter plot comparing actual leasing vs. predicted leasing (from our model) on the test set.
Values are converted to millions of square feet for better readability.

Example of one point may be: All Class A office leases in Manhattan during Q2 of 2020
"""

y_test_millions = y_test / 1e6
y_pred_millions = y_pred / 1e6

sns.set_theme(style="ticks", palette="deep")
sns.set_context("talk")
plt.figure(figsize=(10,8))
plt.scatter(y_test_millions, y_pred_millions, alpha=0.8, s=100,
            color='dodgerblue', edgecolor='white', linewidth=1, label='Test Observations')
min_val = min(y_test_millions.min(), y_pred_millions.min())
max_val = max(y_test_millions.max(), y_pred_millions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (Actual = Predicted)')
plt.xlabel("Actual Leasing (millions of sq ft)")
plt.ylabel("Predicted Leasing (millions of sq ft)")
plt.title("Actual vs. Predicted Leasing")
plt.legend()
plt.subplots_adjust(bottom=0.005)

plt.figtext(0.5, 0.08,
            "Each point represents one market–quarter–year-internal_class combination.\n"
            "Ex: Class A office leases in Manhattan during Q2 of 2020",
            wrap=True, horizontalalignment='center', fontsize=16)

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()


"""
Get permutation importance on the test set for the selected predictors. This tells us how much the model's R^2
decreases when each feature is randomly shuffled, showing feature's importance.
"""

result = permutation_importance(lr, X_test_selected, y_test, n_repeats=10, random_state=100, scoring='r2')
importance_df = pd.DataFrame({
    'Feature': best_features,
    'Importance Mean': result.importances_mean,
    'Importance Std': result.importances_std
}).sort_values(by='Importance Mean', ascending=False)

print("Permutation Importances:")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance Mean'], xerr=importance_df['Importance Std'], color='skyblue')
plt.xlabel("Mean Decrease in R²")
plt.title("Feature Importance (Permutation Importance)")
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()

"""
For each predictor in the selected feature set, we forecast its 2025 value using historical trends.
We group the aggregated data (df_agg) by year and compute the mean for each feature, then fit a simple linear 
regression (predictor value ~ year) to forecast 2025. If insufficient data is available, we use the last available value.
"""

future_predictors = {}
for feature in best_features:
    feat_hist = df_agg.groupby('year')[feature].mean().reset_index()
    feat_hist = feat_hist.dropna()
    if feat_hist.shape[0] < 2:
        future_value = feat_hist[feature].iloc[-1]
        print(f"Insufficient data for {feature}. Using last available value: {future_value}")
    else:
        X_feat = feat_hist[['year']]
        y_feat = feat_hist[feature]
        reg_feat = LinearRegression().fit(X_feat, y_feat)
        future_value = reg_feat.predict(np.array([[2025]]))[0]
    future_predictors[feature] = future_value
    print(f"Forecast for {feature} in 2025: {future_value}")

future_data = pd.DataFrame({feature: [future_predictors[feature]] for feature in best_features})
predicted_leasing_2025 = lr.predict(future_data)[0]
print("\nPredicted Leasing for 2025:", predicted_leasing_2025)


"""
Aggregate historical leasing by year (using df_agg) and then append the 2025 forecast.
Plot a line chart to compare historical mean leasing (2018-2024) with the 2025 forecast.
"""

historical = df_agg.groupby('year').agg(mean_leasing=('leasing', 'mean')).reset_index()
future_row = pd.DataFrame({'year': [2025], 'mean_leasing': [predicted_leasing_2025]})
historical_all = pd.concat([historical, future_row], ignore_index=True).sort_values('year')

historical_all['mean_leasing'] = historical_all['mean_leasing'] / 1e6

# can you tell im trying to make it look fancy
plt.figure(figsize=(10,6))
sns.lineplot(data=historical_all, x='year', y='mean_leasing', marker='o', markersize=8, linewidth=2, color='royalblue', label='Mean Leasing')
plt.axvline(2024, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='End of Historical Data')
plt.title("Historical Mean Leasing (2018-2024) & 2025 Forecast", fontsize=16, weight='bold', pad=15)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Mean Leasing (Million sq ft)", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
sns.despine()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.show()
