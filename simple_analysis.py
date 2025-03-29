import pandas as pd
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

print("\nUnemployment DataFrame")
print(unemployment_df)
print("\nColumns in Unemployment:")
print(unemployment_df.columns.tolist())
