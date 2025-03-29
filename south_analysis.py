import pandas as pd
import matplotlib.pyplot as plt

# Load leasing data and create a copy of rows from the South region only
df = pd.read_csv("Leases.csv")
south_df = df.query("region == 'South'").copy()

# Create a year_quarter column for grouping and plotting trends by time
south_df["year_quarter"] = south_df["year"].astype(str) + south_df["quarter"].astype(str)

# Group the data by market to analyze leasing activity and average rent
south_markets = south_df.groupby("market").agg(
    south_transactions=("leasedSF", "count"),      
    south_avg_leasedSF=("leasedSF", "mean"),        
    south_total_leasingSF=("leasedSF", "sum"),       
    south_avg_overall_rent=("overall_rent", "mean") 
).reset_index()
