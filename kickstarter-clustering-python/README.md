# Kickstarter: Identifying factors contribute to a successful project through K-Means Clustering

## Overview
This project analyzes Kickstarter projects using K-Means clustering to identify factors contributing to success. The analysis provides insights into cluster-specific characteristics and offers recommendations for project creators and Kickstarter.

## Files
- `Kickstarter Clustering Task.ipynb`: The Jupyter Notebook containing the analysis.
- `Kickstarter Project Clustering`: A detailed project report summarizing the findings.
- `Kickstarter.xlsx`: The dataset used.
- `requirements.txt`: List of required Python libraries.

## Data Preprocessing
- Filtered the dataset to include only "successful" and "failed" projects.
- Applied log transformation to skewed variables (e.g., goals, pledged amounts).
- Engineered additional features
- Standardized numerical features for equal contribution to clustering.
- Removed 145 anomalies using Isolation Forest.
- Eliminated multicollinearity using VIF analysis.

## Model Development
Clustering Algorithm: K-Means

Optimal Number of Clusters: Determined to be 6 using Elbow Method and Silhouette Score 

## Results Summary

The analysis revealed six distinct clusters with varying success rates and characteristics:

- Cluster 1: 91.1% success rate: Projects have low goals, short funding duration, and focus on more popular categories with above-average success rates.
- Cluster 2: 0.0% success rate: Projects have high goals and focus on less popular categories with very low success rate.
- Cluster 3: 99.3% success rate: Projects have	low goals, short funding duration, and focus on niche categories with high success rates.
- Cluster 4: 94.6% success rate: Projects have high goals and pledged amounts, focus on high-success-rate categories, have strong visuals and detailed descriptions.
- Cluster 5: 0.0% success rate: Projects have high goals and focus on niche categories with low success rates.
- Cluster 6: 98.3% success rate: Projects have moderate goals, focus on moderately popular categories with high success rate.

## Key Insights:
- Realistic Goals: Projects with lower or moderate goals consistently perform better.
- Category Popularity: Moderately popular and high-success-rate categories yield better outcomes, while niche categories require extra effort to succeed.
- Campaign Quality: Strong visuals, detailed descriptions, and good engagement boost success rates.
- Niche Projects: While niche categories can excel (e.g., Cluster 3), they are also at higher risk of failure without proper campaign strategies (e.g., Cluster 5).

