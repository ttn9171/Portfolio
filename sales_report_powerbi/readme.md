# ***📊 Power BI Sales Report***
## **Overview**

This Power BI Sales Report provides an interactive and dynamic sales analysis across multiple countries, product categories, and time periods. The report helps stakeholders track sales performance, analyze targets vs. actual sales, and simulate future quota adjustments based on past performance.

You can view the Power BI report [here](https://app.powerbi.com/view?r=eyJrIjoiZjRkOTgyYjYtYjRmOS00YTRmLWE2ZjktYmE5NDg4MmEzYjY2IiwidCI6ImNkMzE5NjcxLTUyZTctNGE2OC1hZmE5LWZjZjhmODlmMDllYSIsImMiOjN9&pageName=0d4e1959a32a51d45c30).

<img width="811" alt="image" src="https://github.com/user-attachments/assets/d803954a-5440-4a19-a990-fc29134939f1" />
<img width="808" alt="image" src="https://github.com/user-attachments/assets/5f857839-ece5-4fb3-959d-4d4cf7aa9c83" />
<img width="807" alt="image" src="https://github.com/user-attachments/assets/12a7ec41-49f7-4b21-8a7e-77e763eaec2a" />
<img width="808" alt="image" src="https://github.com/user-attachments/assets/1f7e0ff3-b6a3-464b-93bb-0bf85b51e127" />
<img width="807" alt="image" src="https://github.com/user-attachments/assets/597c9ec4-3351-4a55-ad8e-798000658499" />


## **Pages**
* Main page: Overview page  
* Sales Performance Dashboard
  * Total sales breakdown by year quarter: Compare actual sales vs. targets for each quarter and year and track year-over-year (YoY) changes and gain/loss comparisons.
  * Total sales breakdown by country: Canada, France, Germany.
  * Total sales breakdown by product category
* Simulation: allows users to estimate 2025 sales with adjustable quota increase parameters.


## **Data Preprocessing & Transformation**
1. Data Cleaning & Preparation (Power Query): Handling missing and mistyped values, ensuring correct data formats (e.g., date, text, numeric, standardizing country and product categories to avoid duplicates and inconsistencies, removing unnecessary columns to enhance performance. 
2. Data Modeling & Relationships: Connected multiple tables via unique keys
* Fact Table: Contains sales transactions with revenue, targets, and time-based attributes.
* Dimension Tables: Include products, regions, and time for structured insights.
3. DAX Calculations & Metrics: used DAX to create KPIs: Total Sales, Sales YTD, Sales Same Period Last Year, Gain/Loss from Last Year, etc. 
4. Advanced Features: 
* Drill-through Navigation: Users can explore data by drilling down by country, year, or product category.
* Dynamic Titles & Labels: Titles change dynamically based on selected filters to improve readability.
* What-If Analysis with Parameters: Created a Sales Simulation Report to forecast 2025 sales targets based on user-defined quota increase percentages.


