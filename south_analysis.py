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

# Sort by number of individual lease deals/transactions to track volume of deals/how active each market is
sorted_transactions = south_markets.sort_values(by='south_transactions', ascending=False)

plt.figure(figsize=(12, 5))
plt.bar(sorted_transactions['market'], sorted_transactions['south_transactions'], color='purple')
plt.title('Lease Volume by Market (South)')
plt.xlabel('Market')
plt.ylabel('Number of Lease Deals')
plt.tight_layout()
plt.show()

# Sort markets by total leased square feet (high to low) to track where demand is strongest 
sorted_leased = south_markets.sort_values(by='south_total_leasingSF', ascending=False)

plt.figure(figsize=(12, 5))
plt.bar(sorted_leased['market'], sorted_leased['south_total_leasingSF'], color='orange')
plt.title('Total Leasing Demand by Market (South)')
plt.xlabel('Market') 
plt.ylabel('Total Leased Square Feet')
plt.tight_layout()
plt.show()

# Sort by average rent to evaluate expensive vs affordable cities
sorted_rent = south_markets.sort_values(by='south_avg_overall_rent', ascending=False)

plt.figure(figsize=(12, 5))
plt.bar(sorted_rent['market'], sorted_rent['south_avg_overall_rent'], color='blue')
plt.title('Average Rent by Market (South)')
plt.xlabel('Market')
plt.ylabel('Avg Rent ($/sq ft)')
plt.tight_layout()
plt.show()