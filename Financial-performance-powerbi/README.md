## Overview
This project presents a comprehensive Power BI dashboard designed to analyze the financial performance of a fictional company, Plant Co. The report delivers actionable insights through dynamic visualizations, tracking key business metrics such as Sales, Quantity, and Gross Profit on a Year-to-Date (YTD) and Prior Year-to-Date (PYTD) basis.

You can view the dashboard [here](https://app.powerbi.com/view?r=eyJrIjoiYTE4OWUyYTgtN2FkZi00NGZmLTk5NjYtNmQ4ZjIyMTE0MjQxIiwidCI6ImNkMzE5NjcxLTUyZTctNGE2OC1hZmE5LWZjZjhmODlmMDllYSIsImMiOjN9)

<img width="1246" alt="image" src="https://github.com/user-attachments/assets/e16cc9dd-6f72-4c2a-a8ad-ba8160434603" />


## Visualizations
* Tree Map: Display the bottom 10 countries by sales/quantity/gross profit.
* Waterfall Chart: Highlights monthly YTD vs PYTD performance.
* Stacked Column and Line Chart: Compared quarterly and monthly performance trend by product type. 
* Scatter Plot: Segments customer accounts by gross profit % and key sales metrics (quantity, sales, gross profit) for targeted strategy. 


## Data Preparation & Modeling
* Using Power Query to clean data (removed duplicates and null values)
* Connect Plant Sales table, i.e. Fact table, (Product IDs, quantities, prices, costs, invoice dates) with Accounts table (via Account IDs) and Product Hierarchy Table (via Product IDs)
* DAX calculations:
  * YTD and PYTD Metrics: Calculated sales, gross profit, and quantity using DAX measures.
  * Switch Measures: Enabled toggling between Sales, Quantity, and Gross Profit metrics for dynamic analysis.
  * Year-over-Year Analysis: Compared current and previous year data using TOTALYTD() and SAMEPERIODLASTYEAR() functions.

## Files
* Dashboard: PlantCo Financial Dashboard.pbix
* Dataset: Plant_DTS.xls

**Note: This dashboard is a fictional representation and is intended for educational purposes as part of a professional portfolio.**
