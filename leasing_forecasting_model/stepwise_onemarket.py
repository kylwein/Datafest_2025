import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

"""
Load the datasets for Leases, Occupancy, Price & Availability, and Unemployment.
"""

leases_df = pd.read_csv("../Leases.csv")
occupancy_df = pd.read_csv("../Major Market Occupancy Data-revised.csv")
price_df = pd.read_csv("../Price and Availability Data.csv")
unemployment_df = pd.read_csv("../Unemployment.csv")

"""
Merge the datasets on common keys and filter for Manhattan
"""
df_merged = pd.merge(leases_df, price_df, on=["year", "quarter", "market"], how="left", suffixes=("", "_price"))
df_merged = pd.merge(df_merged,
                     occupancy_df[
                         ['year', 'quarter', 'market', 'starting_occupancy_proportion', 'avg_occupancy_proportion']],
                     on=["year", "quarter", "market"], how="left")
df_merged = pd.merge(df_merged, unemployment_df[['year', 'quarter', 'state', 'unemployment_rate']],
                     on=["year", "quarter", "state"], how="left")

# Filter for Manhattan - try both "Manhattan" and "New York"
manhattan_df = df_merged[df_merged['market'].str.contains('Manhattan|New York', case=False, na=False)].copy()

# Check what we got
print(f"Manhattan/NY markets found: {manhattan_df['market'].unique()}")
print(f"Total observations for Manhattan/NY: {manhattan_df.shape[0]}")

"""
For robustness, let's handle the case where 'Manhattan' might be spelled differently
or be part of 'New York' in different datasets
"""
if manhattan_df.shape[0] == 0:
    # If no exact match, look for anything containing 'NY' or 'New York'
    manhattan_df = df_merged[df_merged['market'].str.contains('NY|New York', case=False, na=False) |
                             df_merged['state'].str.contains('NY|New York', case=False, na=False)].copy()
    print(f"Using broader NY criteria: {manhattan_df.shape[0]} observations")

"""
If we still don't have Manhattan data, we'll use the market with the most observations
"""
if manhattan_df.shape[0] < 10:
    market_counts = df_merged['market'].value_counts()
    most_common_market = market_counts.index[0]
    manhattan_df = df_merged[df_merged['market'] == most_common_market].copy()
    print(f"Not enough Manhattan data. Using {most_common_market} with {manhattan_df.shape[0]} observations instead.")

"""
Aggregate the data by year, quarter, and internal_class
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

# Handle the case where some columns might not exist in the data
agg_cols = [col for col in agg_cols if col in manhattan_df.columns]

manhattan_agg = manhattan_df.groupby(['year', 'quarter', 'internal_class'])[agg_cols].mean().reset_index()
print(f"Aggregated observations: {manhattan_agg.shape[0]}")

"""
Prepare data for modeling - convert to numeric and handle missing values
"""
target = 'leasing'
# Make sure target exists
if target not in manhattan_agg.columns:
    print(f"Target column '{target}' not found. Available columns: {manhattan_agg.columns.tolist()}")
    # Find a suitable alternative if needed
    if 'leasedSF' in manhattan_agg.columns:
        target = 'leasedSF'
        print(f"Using '{target}' as target instead.")
    else:
        raise ValueError("No suitable target column found")

# Define potential predictors
numeric_predictors = [
    'RBA', 'availability_proportion', 'available_space', 'starting_occupancy_proportion',
    'avg_occupancy_proportion', 'direct_availability_proportion', 'direct_available_space',
    'direct_internal_class_rent', 'direct_overall_rent', 'internal_class_rent', 'leasedSF',
    'overall_rent', 'sublet_availability_proportion', 'sublet_available_space',
    'sublet_internal_class_rent', 'sublet_overall_rent', 'unemployment_rate'
]

# Filter to only include columns that exist in the data
predictors = [col for col in numeric_predictors if col in manhattan_agg.columns and col != target]
print(f"Available predictors: {predictors}")

# Convert columns to numeric
for col in predictors + [target]:
    manhattan_agg[col] = pd.to_numeric(manhattan_agg[col], errors='coerce')

# Check how many missing values we have
missing_counts = manhattan_agg[predictors + [target]].isnull().sum()
print("Missing value counts:")
print(missing_counts)

# Instead of dropping rows with missing values, we'll use imputation
# Let's see how many complete cases we have first
complete_cases = manhattan_agg.dropna(subset=predictors + [target])
print(f"Complete cases: {complete_cases.shape[0]}")

# If we have very few complete cases, we'll use imputation
manhattan_model = manhattan_agg.copy()

# Only impute if we need to and have enough data
if complete_cases.shape[0] < 20 and manhattan_model.shape[0] >= 20:
    print("Using imputation for missing values")
    # Simple mean imputation for predictors
    imputer = SimpleImputer(strategy='mean')
    manhattan_model[predictors] = imputer.fit_transform(manhattan_model[predictors])

    # But we don't want to impute the target
    manhattan_model = manhattan_model.dropna(subset=[target]).reset_index(drop=True)
else:
    # If we have enough complete cases or too little data, use complete cases
    manhattan_model = complete_cases.reset_index(drop=True)

print(f"Final dataset size: {manhattan_model.shape[0]} observations with {len(predictors)} predictors")

"""
Split the data for modeling if we have enough observations
"""
if manhattan_model.shape[0] < 10:
    print("Not enough data for reliable modeling")
    exit()

X = manhattan_model[predictors]
y = manhattan_model[target]

# Use a larger test set proportion if we have less data
test_size = 0.33 if manhattan_model.shape[0] < 30 else 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)

print(f"Training set: {X_train.shape[0]} observations")
print(f"Test set: {X_test.shape[0]} observations")

"""
Feature selection - be more lenient if we have limited data
"""
lr = LinearRegression()

# Decide on feature selection approach based on data size
if X_train.shape[0] >= 20 and len(predictors) > 5:
    print("Running feature selection")
    # Reduce the number of CV folds if we have limited data
    cv_folds = min(5, max(3, X_train.shape[0] // 5))
    try:
        sfs = SFS(lr,
                  k_features=(1, len(predictors)),  # Consider all possible numbers of features
                  forward=True,
                  floating=False,  # Simpler selection to avoid overfitting
                  scoring='r2',
                  cv=cv_folds,
                  n_jobs=-1)
        sfs = sfs.fit(X_train, y_train)
        best_features = list(sfs.k_feature_names_)
        print(f"Selected {len(best_features)} features: {best_features}")
        print(f"Cross-validated R^2 Score: {sfs.k_score_}")
    except Exception as e:
        print(f"Feature selection failed: {e}")
        # Fall back to using all features
        best_features = predictors
else:
    print("Using all features (no feature selection)")
    best_features = predictors

# If feature selection returned no features, use all of them
if len(best_features) == 0:
    best_features = predictors
    print("Feature selection resulted in empty set. Using all features.")

X_train_selected = X_train[best_features]
X_test_selected = X_test[best_features]

"""
Fit the model and evaluate
"""
lr.fit(X_train_selected, y_train)
y_pred = lr.predict(X_test_selected)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Model R^2 on test set: {r2:.4f}")
print(f"Model MSE on test set: {mse:.4f}")

# Run cross-validation if we have enough data
if X_train.shape[0] >= 10:
    k_folds = min(5, max(2, X_train.shape[0] // 5))
    try:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=100)
        cv_scores = cross_val_score(lr, X_train_selected, y_train, scoring='r2', cv=kf)
        print(f"{k_folds}-fold CV R^2 Scores: {cv_scores}")
        print(f"Mean CV R^2: {np.mean(cv_scores):.4f}")
    except Exception as e:
        print(f"Cross-validation failed: {e}")

"""
Visualize actual vs predicted values
"""
y_test_millions = y_test / 1e6
y_pred_millions = y_pred / 1e6

sns.set_theme(style="ticks", palette="deep")
plt.figure(figsize=(10, 8))
plt.scatter(y_test_millions, y_pred_millions, alpha=0.8, s=100,
            color='dodgerblue', edgecolor='white', linewidth=1)
min_val = min(y_test_millions.min(), y_pred_millions.min())
max_val = max(y_test_millions.max(), y_pred_millions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel("Actual Leasing (millions of sq ft)")
plt.ylabel("Predicted Leasing (millions of sq ft)")
plt.title("Manhattan: Actual vs. Predicted Leasing")
plt.tight_layout()
plt.show()

"""
Feature Importance
"""
if X_test_selected.shape[0] >= 5:
    try:
        result = permutation_importance(lr, X_test_selected, y_test, n_repeats=10,
                                        random_state=100, scoring='r2')
        importance_df = pd.DataFrame({
            'Feature': best_features,
            'Importance Mean': result.importances_mean,
            'Importance Std': result.importances_std
        }).sort_values(by='Importance Mean', ascending=False)

        print("Feature Importance:")
        print(importance_df)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance Mean'],
                 xerr=importance_df['Importance Std'], color='skyblue')
        plt.xlabel("Mean Decrease in R²")
        plt.title("Manhattan: Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Permutation importance calculation failed: {e}")

"""
Forecast 2025 values
"""
future_predictors = {}
print("\nForecasting 2025 feature values:")
for feature in best_features:
    feat_hist = manhattan_agg.groupby('year')[feature].mean().reset_index()
    feat_hist = feat_hist.dropna()

    if feat_hist.shape[0] >= 2:
        X_feat = feat_hist[['year']]
        y_feat = feat_hist[feature]
        try:
            reg_feat = LinearRegression().fit(X_feat, y_feat)
            future_value = reg_feat.predict(np.array([[2025]]))[0]

            if (feature.endswith('proportion') and (future_value < 0 or future_value > 1)) or \
                    (not feature.endswith('proportion') and feature != 'unemployment_rate' and future_value < 0):
                future_value = feat_hist[feature].iloc[-1]
                print(f"  {feature}: Trend gave unreasonable value. Using last available: {future_value:.4f}")
            else:
                print(f"  {feature}: {future_value:.4f} (from trend)")
        except Exception as e:
            future_value = feat_hist[feature].iloc[-1]
            print(f"  {feature}: Trend failed. Using last available: {future_value:.4f}")
    else:
        future_value = manhattan_model[feature].mean()
        print(f"  {feature}: {future_value:.4f} (using mean)")

    future_predictors[feature] = future_value

future_data = pd.DataFrame({feature: [future_predictors[feature]] for feature in best_features})

predicted_leasing_2025 = lr.predict(future_data)[0]
print(f"\nPredicted Manhattan Leasing for 2025: {predicted_leasing_2025:.2f}")
print(f"Predicted Manhattan Leasing for 2025 (in millions sq ft): {predicted_leasing_2025 / 1e6:.2f}")

"""
Compare historical trends with forecast
"""
historical = manhattan_agg.groupby('year').agg(mean_leasing=(target, 'mean')).reset_index()
future_row = pd.DataFrame({'year': [2025], 'mean_leasing': [predicted_leasing_2025]})
historical_all = pd.concat([historical, future_row], ignore_index=True).sort_values('year')

historical_all['mean_leasing'] = historical_all['mean_leasing'] / 1e6

plt.figure(figsize=(12, 6))
sns.lineplot(data=historical_all, x='year', y='mean_leasing', marker='o', markersize=8,
             linewidth=2, color='royalblue')
plt.axvline(historical_all['year'].max() - 1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
plt.fill_betweenx([0, historical_all['mean_leasing'].max() * 1.2],
                  historical_all['year'].max() - 1, historical_all['year'].max() + 1,
                  color='lightgray', alpha=0.3)
plt.text(historical_all['year'].max() - 0.5, historical_all['mean_leasing'].max() * 1.1,
         'Forecast', fontsize=12, ha='center')

plt.title("Manhattan: Historical Mean Leasing & 2025 Forecast", fontsize=16)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Mean Leasing (Million sq ft)", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""
Compare the forecast to the last known value
"""
latest_year = historical['year'].max()
latest_leasing = historical[historical['year'] == latest_year]['mean_leasing'].values[0]
change_pct = (predicted_leasing_2025 - latest_leasing) / latest_leasing * 100

print(f"\nLatest known leasing ({latest_year}): {latest_leasing:.2f}")
print(f"Forecasted leasing (2025): {predicted_leasing_2025:.2f}")
print(f"Change: {change_pct:.1f}%")

"""
Bonus: Class breakdown if we have enough data
"""
classes = manhattan_model['internal_class'].unique()
print(f"\nClasses in Manhattan data: {classes}")

if len(classes) > 1 and manhattan_model.shape[0] >= 15:
    print("\nClass breakdown analysis:")

    # Current class distribution
    class_dist = manhattan_model.groupby('internal_class')[target].mean().reset_index()
    class_dist['mean_leasing_millions'] = class_dist[target] / 1e6

    print(class_dist)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=class_dist, x='internal_class', y='mean_leasing_millions', palette='viridis')
    plt.title("Manhattan: Mean Leasing by Class", fontsize=14)
    plt.xlabel("Property Class", fontsize=12)
    plt.ylabel("Mean Leasing (Million sq ft)", fontsize=12)
    plt.tight_layout()
    plt.show()

    for class_name in classes:
        class_data = manhattan_model[manhattan_model['internal_class'] == class_name]
        if class_data.shape[0] >= 10:
            print(f"\nClass {class_name} forecast:")
            X_class = class_data[best_features]
            y_class = class_data[target]

            class_lr = LinearRegression().fit(X_class, y_class)

            class_pred = class_lr.predict(future_data)[0]
            print(
                f"Predicted 2025 leasing for Class {class_name}: {class_pred:.2f} ({class_pred / 1e6:.2f} million sq ft)")
        else:
            print(f"\nNot enough data for Class {class_name} forecast ({class_data.shape[0]} observations)")