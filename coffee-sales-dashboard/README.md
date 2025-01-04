# Coffee Sales Dashboard Project

## Overview

This project involves creating an interactive Coffee Sales Dashboard in Excel to analyze sales data. The dashboard integrates data from three spreadsheets — Orders (Raw Data), Customers, and Products — to provide insights into total sales over time, sales by country, and the top 5 customers. 

<img width="1282" alt="image" src="https://github.com/user-attachments/assets/a0684a98-3faa-4b72-95b8-b0149500d265" />


## Features
### Interactive Dashboard
- Total Sales Over Time: A line chart showing sales trends for different coffee types across multiple years.
- Sales by Country: A bar chart summarizing total sales for each country.
- Top 5 Customers: A bar chart highlighting the top customers based on their total purchases.
### Dynamic Filtering
- Slicers: Filter by Coffee Type, Roast Type, Loyalty Card status, and Size (in kg).
- Timeline: Filter data by specific months or years, updating all visuals dynamically.

## Steps to Build the Dashboard
### Data Preparation:
- Combined raw data from Orders, Customers, and Products sheets into a single sheet called Orders with the help of XLOOKUP, INDEX-MATCH, and other Excel formulas to merge and clean the data.
- Created separate pivot tables to analyze: Total sales over time, Sales by country, and Top 5 customers.

### Interactive Filters:
- Added slicers for key fields (e.g., Coffee Type, Roast Type) and connected them to all three pivot tables.
- Inserted a timeline slicer to filter data by order date.

### Visualizations:
- Built dynamic charts (line chart and bar charts) using pivot table data.
- Ensured all charts update based on slicer and timeline filters.

## How to Use the Dashboard
- Open the file coffeeOrdersData.xlsx in Excel.
- Navigate to the Dashboard tab.
- Use the slicers and timeline to filter data: The charts will update dynamically to reflect your selections.
  - Select specific coffee types, roast types, sizes, or loyalty card statuses using the slicers.
  - Use the timeline to focus on a specific date range (e.g., a particular year or month).

## Key Insights from the Dashboard
- Clear fluctuations in total sales over time.
- The United States leads in total sales, significantly outperforming other countries.
- Identified the top 5 customers, with Allis Wilmore generating the highest revenue.

## Sources

Dataset: https://github.com/mochen862/excel-project-coffee-sales/blob/main/coffeeOrdersData.xlsx

## Files

coffeeOrdersData.xlsx: The Excel file containing the raw data, combined data, pivot tables, and the interactive dashboard.
