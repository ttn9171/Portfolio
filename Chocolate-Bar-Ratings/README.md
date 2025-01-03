# Predicting Chocolate Ratings: An R-Based Classification Approach

## Overview 
This project leverages machine learning techniques to classify chocolate bars as "Satisfactory" or "Unsatisfactory" based on expert ratings and product features. By framing the task as a binary classification problem, the analysis provides actionable insights for manufacturers to optimize product characteristics and improve consumer satisfaction.

The analysis focuses on tree-based models (Random Forest and Gradient Boosting) due to their robustness in handling non-linear relationships and complex feature interactions.

## Dataset 
- Source: Chocolate Bar Ratings dataset from Kaggle (https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings)
- Target Variable: Rating Category:
  - Ratings below 3 are classified as "Unsatisfactory."
  - Ratings of 3 or above are classified as "Satisfactory."
- Key Features:
  - Numerical Variables:
    - Cocoa.Percent: Cocoa content percentage.
    - REF and Review Date: Recency of the review and Date of the Review
  - Categorical Variables:
    - Company.Location: Country where the chocolate is produced.
    - Bean.Type: Type of cocoa beans used.
    - Broad.Bean.Origin and Specific.Bean.Origin: Cocoa bean origins.
    - Company.Maker.if.known: Chocolate manufacturer.

## Files

- chocolate-ratings.R: R script for preprocessing, feature engineering, and model training. 
- report_summary.pdf: Detailed project report summarizing the methodology, results, and insights.
- README.md: High level summary of the project.
- Chocolate bar ratings.csv: the dataset used for the classification task. 

## Data Preprocessing
1. Target Variable Transformation: Converted numerical Rating into a binary Rating Category for classification.
2. Reclassification of Low-Frequency Categories: Rare categories in Company.Location, Bean.Type, and Broad.Bean.Origin were grouped into an "Other" category to reduce sparsity.
3. Feature Engineering:
  - Company Popularity: Companies were categorized into tiers based on their frequency in the dataset.
  - Company Rating Mean: Target encoding of Company.Maker.if.known using company average ratings.
4. Feature Importance: Cocoa.Percent, Broad.Bean.Origin, and Company Rating Mean emerged as the most important predictors.

## Model Selection
Two machine learning models were evaluated: Random Forest (RF) and Gradient Boosting Machine (GBM)

Evaluation Metrics:
- Accuracy: Overall correctness of the model's predictions.
- Balanced Accuracy: Average of sensitivity and specificity to address class imbalance.
- F1 Score: Harmonic mean of precision and recall, emphasizing performance on the minority class.

## Results Summary

The Random Forest model was selected as the best-performing algorithm, achieving:
- Accuracy: 79.05%
- Balanced Accuracy: 62.75%
- Sensitivity: 30.34%
- Specificity: 95.17%
- F1 Score: 41.7%

Feature Importance:
- Company Rating Mean: Strongest predictor of satisfactory ratings.
- Broad Bean Origin: Differentiates product quality based on cocoa source.
- Cocoa.Percent: Captures consumer preference for specific cocoa content.
- Company Popularity: Reflects brand influence on consumer perception.

## Key Insights 
- While cocoa percentage is important, very high values negatively impact ratings, suggesting a preference for balanced flavors.
- Broad bean origins influence consumer perceptions due to regional flavor profiles.
- Manufacturer's reputation (Company Ratings and Company Popularity) significantly impacts product acceptance.
- Class Imbalance: the model struggles to identify "Unsatisfactory" chocolate bars, indicating the need for additional data or advanced techniques to improve sensitivity.
