import pandas as pd
import matplotlib.pyplot as plt

# Load leasing data and create a copy of rows from the South region only
df = pd.read_csv("Leases.csv")
south_df = df.query("region == 'South'").copy()


# Leasing trends by market represented in bar graphs

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


# Leasing trends overtime represented in line graphs

# Create a year_quarter column for grouping and plotting trends by time
south_df["year_quarter"] = south_df["year"].astype(str) + "-" + south_df["quarter"].astype(str)

# Group data by year and quarter to analyze leasing activity and average rent overtime
time_summary = south_df.groupby("year_quarter").agg(
    lease_count=("leasedSF", "count"),         
    total_leasedSF=("leasedSF", "sum"),       
    avg_rent=("overall_rent", "mean")         
).reset_index()

plt.figure(figsize=(12, 5))
plt.plot(time_summary["year_quarter"], time_summary["lease_count"], color='purple')
plt.title("Lease Volume Over Time (South)")
plt.xlabel("Year and Quarter")
plt.ylabel("Number of Lease Deals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(time_summary["year_quarter"], time_summary["total_leasedSF"], color='orange')
plt.title("Total Leasing Demand Over Time (South)")
plt.xlabel("Year and Quarter")
plt.ylabel("Total Leased SF")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(time_summary["year_quarter"], time_summary["avg_rent"], color='blue')
plt.title("Average Rent Over Time (South)")
plt.xlabel("Year and Quarter")
plt.ylabel("Avg Rent ($/sq ft)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
