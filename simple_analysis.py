import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

leases_df = pd.read_csv("Leases.csv")
occupancy_df = pd.read_csv("Major Market Occupancy Data-revised.csv")
price_df = pd.read_csv("Price and Availability Data.csv")
unemployment_df = pd.read_csv("Unemployment.csv")

print("Leases DataFrame:")
print(leases_df)
print("\nColumns in Leases:")
print(leases_df.columns.tolist())

print("\nOccupancy DataFrame:")
print(occupancy_df)
print("\nColumns in Occupancy:")
print(occupancy_df.columns.tolist())

print("\nPrice & Availability DataFrame:")
print(price_df)
print("\nColumns in Price & Availability:")
print(price_df.columns.tolist())

print("\nUnemployment DataFrame:")
print(unemployment_df)
print("\nColumns in Unemployment:")
print(unemployment_df.columns.tolist())

print(leases_df.select_dtypes(include=[np.number]).describe())

market_agg = leases_df.groupby("market").agg(
    transactions=("leasedSF", "count"),
    avg_leasedSF=("leasedSF", "mean"),
    total_leasing=("leasing", "sum"),
    avg_overall_rent=("overall_rent", "mean")
).reset_index()

occupancy_summary = occupancy_df.groupby("market")["avg_occupancy_proportion"].mean().reset_index().rename(
    columns={"avg_occupancy_proportion": "avg_occupancy"}
)

market_summary = pd.merge(market_agg, occupancy_summary, on="market", how="left")

market_summary = market_summary.sort_values(by="total_leasing", ascending=False)

print("Market Summary:")
print(market_summary.to_string(index=False))

