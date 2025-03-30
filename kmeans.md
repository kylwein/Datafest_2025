# Market Segmentation Analysis

## Overview
Merged leasing, price, occupancy, and unemployment data to segment markets using leasing volume, available space, leased SF, overall rent, and unemployment rate. We then applied **K-Means clustering** (k = 3) and **PCA** for visualization.

## Methodology
- **Data Aggregation:**  
  Group by year, quarter, market, and industry (internal_industry) to get average values.

- **K-Means Clustering:**  
  Standardize features and cluster the data into 3 segments based on similarity in leasing metrics.

- **PCA:**  
  Reduce the multi-dimensional data to two principal components (PC1 & PC2) that capture ~42.6% and ~21.0% of the variance, respectively. Each point represents one market–quarter–industry observation.
- Example would be: Healthcare industry in Atlanta during the second quarter of 2023

## Cluster Profiles
| **Cluster** | **Leasing (Mean)** | **Available Space (Mean)** | **Leased SF (Mean)** | **Overall Rent (Mean)** | **Unemployment Rate (Mean)** | **Top Industries**                                           |
|-------------|--------------------|----------------------------|----------------------|-------------------------|------------------------------|-------------------------------------------------------------|
| **0**       | 686k               | 10.3M                      | 27k                  | \$33.86                 | 4.70%                        | Tech (315); Healthcare (273); Legal (265)                   |
| **1**       | 1.68M              | 28.0M                      | 47.5k                | \$35.07                 | 4.43%                        | Tech (154); Legal (136); Construction (112)                 |
| **2**       | 3.88M              | 36.2M                      | 37.5k                | \$77.39                 | 5.39%                        | No dominant industry; evenly distributed (~28 each)       |

## Interpretation
- **Cluster 0:**  
  Represents stable, mid-tier markets with moderate leasing and space, where technology, healthcare, and legal services are most common.
  
- **Cluster 1:**  
  Consists of larger, more active markets with slightly higher leasing and available space, and the lowest unemployment—dominated by tech, legal, and construction sectors.
  
- **Cluster 2:**  
  Captures premium, high-demand markets with very high leasing and rent, and higher unemployment; industry representation is evenly spread with healthcare leading.

## Conclusion
Using K-Means and PCA, we identified three distinct market segments. This segmentation helps tailor strategies by revealing which industries dominate within each market type and highlights key differences in leasing dynamics.
