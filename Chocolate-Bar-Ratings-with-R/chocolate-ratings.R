install.packages("dplyr")
library(caret)
library(dplyr)

install.packages("randomForest")
library(randomForest)
install.packages("gbm")
library(gbm) 

library(ggplot2)
library(stargazer)
library(viridis)

data = read.csv("./Chocolate Bar Ratings/Chocolate bar ratings.csv")
attach(data)

summary(data)

#Check for NA values
colSums(is.na(data))


###############################################################################
# ALL CODES FOR THE GRAPHS ARE INCLUDED IN BOTTOM SECTION


############################# PREPROCESSING #####################################
#Turn cocoa percentage to numerical - currently character 
data$Cocoa.Percent = as.numeric(gsub("%", '',data$Cocoa.Percent))


rf_model <- randomForest(
  Rating ~ Cocoa.Percent + Company...Maker.if.known. + Specific.Bean.Origin.or.Bar.Name + REF + Review.Date +
    Bean.Type + Broad.Bean.Origin + Company.Location,
  data = data,
  ntree = 500,       # Number of trees
  importance = TRUE, 
  na.action = na.omit
)

# Extract feature importance
importance_values <- importance(rf_model)
print(importance_values)
varImpPlot(rf_model, main = "Feature Importance from Random Forest")



#### Processing categorical values 
categorical_columns = names(data)[sapply(data, function(col) is.character(col))]

frequency_tables <- list()
categorical_columns <- names(data)[sapply(data, function(col) is.factor(col) || is.character(col))]


############################# FEATURE ENGINEERING #####################################
##### Classify Bean Origin Region 
region_mapping <- list(
  "South America" = c("Peru", "Venezuela", "Brazil", "Ecuador", "Bolivia", "Colombia", "Suriname"),
  "Africa" = c("Sao Tome", "Ghana", "Madagascar", "Togo", "Tanzania", "Congo", "Ivory Coast", "Nigeria", "Cameroon", "Liberia", "Gabon"),
  "Central America/Caribbean" = c("Dominican Republic", "Jamaica", "Grenada", "Guatemala", "Honduras", "Costa Rica", "Belize", "Trinidad", "Tobago", "Puerto Rico", "Martinique", "Haiti", "St. Lucia"),
  "Asia-Pacific" = c("Papua New Guinea", "Indonesia", "Philippines", "Vietnam", "Fiji", "Malaysia", "PNG", "Java", "Solomon Islands", "Samoa"),
  "North America" = c("Mexico", "Hawaii"),
  "Oceania" = c("Australia", "Vanuatu"),
  "Mixed/Multiple" = c("Ven., Trinidad, Mad.", "Ecuador, Mad., PNG", "Africa, Carribean, C. Am.", "South America, Africa", "Ghana, Domin. Rep", "Ven.,Ecu.,Peru,Nic.", "Mad., Java, PNG")
)

# Function to classify origin by region
classify_region <- function(origin) {
  for (region in names(region_mapping)) {
    if (any(grepl(paste(region_mapping[[region]], collapse = "|"), origin, ignore.case = TRUE))) {
      return(region)
    }
  }
  return("Unknown")  # Default for unmatched entries
}


data$Bean_Region <- sapply(data$Broad.Bean.Origin, classify_region)
data$Bean_Region <- as.factor(data$Bean_Region)


###### Classify Company Region 

company_region_map <- list(
  "North America" = c("U.S.A.", "Canada", "Mexico", "Puerto Rico"),
  "South America" = c("Ecuador", "Peru", "Brazil", "Argentina", "Colombia", "Venezuela", "Bolivia", "Chile", "Suriname"),
  "Europe" = c("France", "Switzerland", "Netherlands", "Spain", "Italy", "U.K.", "Wales", "Belgium", "Germany", 
               "Russia", "Portugal", "Denmark", "Sweden", "Poland", "Austria", "Finland", "Lithuania", "Ireland", "Hungary", "Czech Republic"),
  "Asia" = c("Japan", "India", "Vietnam", "Singapore", "Israel", "South Korea", "Philippines"),
  "Africa" = c("South Africa", "Ghana", "Madagascar", "Sao Tome"),
  "Oceania" = c("Fiji", "Australia", "New Zealand"),
  "Caribbean" = c("Martinique", "St. Lucia", "Grenada", "Domincan Republic", "Costa Rica", "Honduras", "Nicaragua")
)


classify_country_region <- function(country) {
  for (region in names(company_region_map)) {
    if (country %in% company_region_map[[region]]) {
      return(region)
    }
  }
  return("Unknown") 
}

data$Company_Region <- sapply(data$Company.Location, classify_country_region)
data$Company_Region <- as.factor(data$Company_Region)

######## Factor Company Location 
location_counts <- data %>%
  group_by(Company.Location) %>%
  summarise(Unique_Companies = n_distinct(Company...Maker.if.known.)) %>%
  arrange(desc(Unique_Companies))

print(location_counts)

threshold <- 5

rare_locations <- location_counts %>%
  filter(Unique_Companies < threshold) %>%
  pull(Company.Location)

# Replace rare locations with "Other" in the dataset
data$Company.Location <- ifelse(
  data$Company.Location %in% rare_locations,
  "Other",
  data$Company.Location
)

data$Company.Location <- as.factor(data$Company.Location)
print(table(data$Company.Location))

#######Factor only high frequency Bean.Type
bean_counts <- table(data$Bean.Type)
rare_beans <- names(bean_counts[bean_counts < 11])
data$Bean.Type  <- ifelse(data$Bean.Type  %in% rare_beans, "Other", data$Bean.Type )
data$Bean.Type <- as.factor(data$Bean.Type)

#######Factor only high frequency Broad.Bean.Origin 
broad_origin_counts <- table(data$Broad.Bean.Origin)
rare_origin <- names(broad_origin_counts[broad_origin_counts < 20])
data$Broad.Bean.Origin <- ifelse(data$Broad.Bean.Origin %in% rare_origin, "Other", data$Broad.Bean.Origin)
data$Broad.Bean.Origin <- as.factor(data$Broad.Bean.Origin)


attach(data)

######Create Company.Popularity Feature 
# Calculate frequencies for each Company Maker
company_counts <- table(data$Company...Maker.if.known.)

# Define thresholds for grouping
small_threshold <- 5
medium_threshold <- 14
big_threshold <- 25


# Group Company Maker into "Small", "Medium", "High"
data$Company.Popularity <- ifelse(
  company_counts[data$Company...Maker.if.known.] < small_threshold, "Low",
  ifelse(
    company_counts[data$Company...Maker.if.known.] <= medium_threshold, "Low to Medium",
    ifelse(
      company_counts[data$Company...Maker.if.known.] <= big_threshold, "Medium",
      'High')
  )
)

data$Company.Popularity <- as.factor(data$Company.Popularity)

table(data$Company.Popularity)

######Create Company.Rating Mean Feature 
#### SPLIT DATA TO CREATE FEATURE TO AVOID DATA LEAKAGE 

set.seed(123)  # For reproducibility

# Create 5-fold cross-validation
folds <- createFolds(data$Rating, k = 5, returnTrain = TRUE)

# Initialize the column for Company.Rating.Mean
data$Company.Rating.Mean <- NA

for (fold in folds) {
  train_data <- data[fold, ]  # Training data
  test_data <- data[-fold, ]  # Test data
  
  company_mean <- train_data %>%
    group_by(Company...Maker.if.known.) %>%  # Reference train_data directly
    summarize(Mean_Rating = mean(Rating, na.rm = TRUE))
  
  test_data <- test_data %>%
    left_join(company_mean, by = "Company...Maker.if.known.")
  
  data[-fold, "Company.Rating.Mean"] <- test_data$Mean_Rating
}

#### FILL NA VALUES WITH THE AVERAGE RATING - all missing values fall in small companies (less frequent companies)
# Calculate the mean rating for small companies
small_company_mean <- data %>%
  filter(Company.Popularity == "Low" & !is.na(Rating)) %>%
  summarize(Small_Company_Mean = mean(Rating, na.rm = TRUE)) %>%
  pull(Small_Company_Mean)
# Fill missing values for small companies
data$Company.Rating.Mean <- ifelse(
  is.na(data$Company.Rating.Mean) & data$Company.Popularity == "Low",
  small_company_mean,
  data$Company.Rating.Mean
)

sum(is.na(data$Company.Rating.Mean))
summary(data$Company.Rating.Mean)


####Adding Cocoa Percent Ratio 

ggplot(data, aes(x = Cocoa.Percent, y = Rating, color = Company.Popularity)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Cocoa.Percent vs. Rating by Company Popularity",
       x = "Cocoa Percent",
       y = "Rating") +
  theme_minimal() +
  scale_fill_viridis(discrete = TRUE)


ggplot(data, aes(x = Cocoa.Percent / Rating, y = Rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Cocoa.Percent/Rating Ratio vs. Rating",
       x = "Cocoa Percent / Rating",
       y = "Rating") +
  theme_minimal()


#data <- data[, -ncol(data)]

set.seed(123)  

folds <- createFolds(data$Rating, k = 5, returnTrain = TRUE)

data$Cocoa_Rating <- NA  

for (fold in folds) {
  train_data <- data[fold, ]  # Training data
  test_data <- data[-fold, ]  # Test data
  
  train_data$Cocoa_Rating <- train_data$Cocoa.Percent / train_data$Rating
  
  test_data$Cocoa_Rating <- test_data$Cocoa.Percent / mean(train_data$Rating, na.rm = TRUE)
  
  data[-fold, "Cocoa_Rating"] <- test_data$Cocoa_Rating
}


sum(is.na(data$Cocoa_Rating))
sum(is.infinite(data$Cocoa_Rating))

# Visualize the feature distribution
hist(data$Cocoa_Rating, breaks = 30, main = "Distribution of Cocoa/Rating", xlab = "Cocoa/Rating")

#### 
attach(data)

#### 

######################################################################################
################## Numerical cols EDA######################
numerical_col = data[, c(3,4,5)]

#### regression for numerical values 
mreg_formula = as.formula(paste("Rating ~", paste(names(numerical_col), collapse="+")))
mreg_numerical = lm(mreg_formula, data=data)

summary(mreg_numerical)

# Linear regression for REF
lm_ref <- lm(Rating ~ REF, data = data)
summary(lm_ref)

# Linear regression for Review.Date
lm_review_date <- lm(Rating ~ Review.Date, data = data)
summary(lm_review_date)

# Linear regression for Cocoa.Percent
lm_cocoa <- lm(Rating ~ Cocoa.Percent, data = data)
summary(lm_cocoa)


######################################################################################
###### CREATE BINARY RATING

data$Rating_Binary <- ifelse(data$Rating >= 3.0, 1, 0)
data$Rating_Binary <- as.numeric(data$Rating_Binary)

# Check the distribution of the target variable
table(data$Rating_Binary)


######################################################################################
###################################GRADIENT BOOSTING#############################


set.seed(123)  # For reproducibility
train_index <- createDataPartition(data$Rating_Binary, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

gbm_model <- gbm(
  Rating_Binary ~ Company.Popularity + Company.Rating.Mean + Cocoa.Percent +
    Broad.Bean.Origin + Bean.Type + Company.Location,
  data = train_data,
  distribution = "bernoulli",  # Binary classification
  n.trees = 10000,             # Number of trees
  interaction.depth = 4,
  shrinkage = 0.01,
  bag.fraction = 0.5 # Depth of each tree
)

summary(gbm_model)
predicted_prob <- predict(gbm_model, newdata = test_data, n.trees = 10000, type = "response")
# Convert probabilities to binary classes
predicted_class <- ifelse(predicted_prob > 0.5, 1, 0)
predicted_class <- as.factor(predicted_class)
test_data$Rating_Binary = as.factor(test_data$Rating_Binary)
confusionMatrix(predicted_class, test_data$Rating_Binary)


################################### Tuning based on accuracy score 

tune_grid <- expand.grid(
  n.trees = c(2000, 5000, 10000),
  interaction.depth = c(3, 4, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = c(10, 20)
)

train_data$Rating_Binary <- factor(train_data$Rating_Binary, levels = c(0, 1), labels = c("Unsatisfactory", "Satisfactory"))
test_data$Rating_Binary <- factor(test_data$Rating_Binary, levels = c(0, 1), labels = c("Unsatisfactory", "Satisfactory"))


# Train GBM with caret
set.seed(123)
gbm_tuned <- train(
  Rating_Binary ~ Company.Popularity + Company.Rating.Mean + Cocoa.Percent +
    Broad.Bean.Origin + Bean.Type + Company.Location,
  data = train_data,
  method = "gbm",
  distribution = "bernoulli",
  tuneGrid = tune_grid,
  trControl = trainControl(method = "cv", number = 5),  # 5-fold CV
  verbose = TRUE
)

print(gbm_tuned$bestTune)
#### > print(gbm_tuned$bestTune)
### n.trees interaction.depth shrinkage n.minobsinnode
### 7    2000                 4      0.01             10

## Fit the best parameters 

data$Rating_Binary <- ifelse(data$Rating >= 3, 1, 0)
data$Rating_Binary <- as.numeric(data$Rating_Binary)

set.seed(123)  # For reproducibility
train_index <- createDataPartition(data$Rating_Binary, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

gbm_model_tuned <- gbm(
  Rating_Binary ~ Company.Popularity + Company.Rating.Mean + Cocoa.Percent +
    Broad.Bean.Origin + Bean.Type + Company.Location,
  data = train_data,
  distribution = "bernoulli",  # Binary classification
  n.trees = 2000,             # Number of trees
  interaction.depth = 4,
  shrinkage = 0.01,
  bag.fraction = 0.5 # Depth of each tree
)

summary(gbm_model_tuned)
predicted_prob <- predict(gbm_model_tuned, newdata = test_data, n.trees = 2000, type = "response")
# Convert probabilities to binary classes
predicted_class <- ifelse(predicted_prob > 0.5, 1, 0)
predicted_class <- as.factor(predicted_class)
test_data$Rating_Binary = as.factor(test_data$Rating_Binary)
confusionMatrix(predicted_class, test_data$Rating_Binary)



########### TUNING with F1 score focus 
data$Rating_Binary <- as.factor(data$Rating_Binary)
custom_summary <- function(data, lev = NULL, model = NULL) {
  precision <- posPredValue(data$pred, data$obs, positive = lev[1])
  recall <- sensitivity(data$pred, data$obs, positive = lev[1])
  f1 <- ifelse((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
  
  c(F1 = f1, Accuracy = sum(data$pred == data$obs) / nrow(data))
}

# Define trainControl
control <- trainControl(
  method = "cv",                # Cross-validation
  number = 5,                   # Number of folds
  verboseIter = TRUE,
  classProbs = TRUE,            # Enable class probabilities
  summaryFunction = custom_summary  # Use custom F1 score metric
)



set.seed(123)  # For reproducibility
train_index <- createDataPartition(data$Rating_Binary, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

train_data$Rating_Binary <- factor(train_data$Rating_Binary, levels = c(0, 1), labels = c("Unsatisfactory", "Satisfactory"))
test_data$Rating_Binary <- factor(test_data$Rating_Binary, levels = c(0, 1), labels = c("Unsatisfactory", "Satisfactory"))


# Define tuning grid
tune_grid <- expand.grid(
  n.trees = c(1000, 2000),
  interaction.depth = c(3, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = c(10, 20)
)

# Train GBM model
gbm_tuned <- train(
  Rating_Binary ~ Company.Popularity + Company.Rating.Mean + Cocoa.Percent +
    Broad.Bean.Origin + Bean.Type + Company.Location,
  data = train_data,
  method = "gbm",
  metric = "F1",                # Optimize for F1 score
  tuneGrid = tune_grid,
  trControl = control
)

print(gbm_tuned$bestTune)

############### Fit best parameters 

data$Rating_Binary <- ifelse(data$Rating >= 3, 1, 0)
data$Rating_Binary <- as.numeric(data$Rating_Binary)

set.seed(123)  # For reproducibility
train_index <- createDataPartition(data$Rating_Binary, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

gbm_model_tuned <- gbm(
  Rating_Binary ~ Company.Popularity + Company.Rating.Mean + Cocoa.Percent +
    Broad.Bean.Origin + Bean.Type + Company.Location,
  data = train_data,
  distribution = "bernoulli",  # Binary classification
  n.trees = 2000,             # Number of trees
  interaction.depth = 5,
  shrinkage = 0.1,
  bag.fraction = 0.5 
)

summary(gbm_model_tuned)
predicted_prob <- predict(gbm_model_tuned, newdata = test_data, n.trees = 2000, type = "response")
# Convert probabilities to binary classes
predicted_class <- ifelse(predicted_prob > 0.5, 1, 0)
predicted_class <- as.factor(predicted_class)
test_data$Rating_Binary = as.factor(test_data$Rating_Binary)
confusionMatrix(predicted_class, test_data$Rating_Binary)



######################################################################################
###################################RANDOM FOREST#############################

data$Rating_Binary <- as.factor(data$Rating_Binary)

set.seed(123)  
train_index <- createDataPartition(data$Rating_Binary, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


rf_model <- randomForest(
  Rating_Binary ~ Company.Popularity + Company.Rating.Mean + Cocoa.Percent +
    Broad.Bean.Origin + Bean.Type + Company.Location,
  data = train_data,
  ntree = 500,       # Number of trees
  mtry = 2,          # Number of variables tried at each split
  nodesize = 10, # Minimum number of samples in each terminal node
  importance = TRUE, # Enable variable importance
  na.action = na.omit
)


print(rf_model)
rf_predictions <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_predictions, test_data$Rating_Binary)

importance(rf_model)  
varImpPlot(rf_model) 



###### Tuning

train_data$Rating_Binary <- factor(train_data$Rating_Binary, 
                                   levels = c(0, 1), 
                                   labels = c("Unsatisfactory", "Satisfactory"))

control <- trainControl(
  method = "cv",          # Cross-validation
  number = 5,             # 5-fold cross-validation
  verboseIter = TRUE,     # Print progress
  classProbs = TRUE,      # Calculate class probabilities
  summaryFunction = twoClassSummary  # Use metrics like AUC, sensitivity, specificity
)

tuning_grid <- expand.grid(
  mtry = c(2, 3, 4),
  nodesize = c(5, 10, 15),
  maxnodes = c(10, 20, 30),
  ntree = c(500,1000,1500,2000)
)

train_rf <- function(train_data, test_data, mtry, ntree, nodesize, maxnodes) {
  rf_model <- randomForest(
    Rating_Binary ~ Company.Popularity + Company.Rating.Mean + Cocoa.Percent +
      Broad.Bean.Origin + Bean.Type + Company.Location,
    data = train_data,
    mtry = mtry,
    ntree = ntree,
    nodesize = nodesize,
    maxnodes = maxnodes,
    importance = TRUE,
    na.action = na.omit
  )
  
  # Predict on test data
  predictions <- predict(rf_model, newdata = test_data)
  
  # Calculate evaluation metrics
  confusion <- confusionMatrix(predictions, test_data$Rating_Binary)
  f1 <- (2 * confusion$byClass["Sensitivity"] * confusion$byClass["Precision"]) /
    (confusion$byClass["Sensitivity"] + confusion$byClass["Precision"])
  
  # Return metrics
  return(list(model = rf_model, f1 = f1, accuracy = confusion$overall["Accuracy"]))
}

set.seed(123)  # For reproducibility
train_index <- createDataPartition(data$Rating_Binary, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

results <- list()  # To store results

# Loop over the grid
for (i in 1:nrow(tuning_grid)) {
  params <- tuning_grid[i, ]
  
  # Train and evaluate the model
  result <- train_rf(
    train_data = train_data,
    test_data = test_data,
    mtry = params$mtry,
    ntree = params$ntree,
    nodesize = params$nodesize,
    maxnodes = params$maxnodes
  )
  
  # Store the result
  results[[i]] <- list(
    params = params,
    f1 = result$f1,
    accuracy = result$accuracy,
    model = result$model
  )
}

# Combine results into a data frame
results_df <- do.call(rbind, lapply(results, function(x) {
  cbind(x$params, F1 = x$f1, Accuracy = x$accuracy)
}))

# Find the best combination
best_result <- results_df[which.max(results_df$F1), ]
print(best_result)

### best params 
###mtry nodesize maxnodes ntree        F1  Accuracy
###24    4       10       30   500 0.2881356 0.7653631


###########Tuned model

tuned_rf <- randomForest(
  Rating_Binary ~ Company.Popularity + Company.Rating.Mean + Cocoa.Percent +
    Broad.Bean.Origin + Bean.Type  + Company.Location,
    # + Cocoa_Rating Company_Region  +   
    #Bean_Region ,
  data = train_data,
  mtry = 4,
  ntree = 500,
  nodesize = 10,
  maxnodes = 30,
  importance = TRUE,
  na.action = na.omit
)

# Evaluate the final model on the test set
final_predictions <- predict(tuned_rf, newdata = test_data)
confusionMatrix(final_predictions, test_data$Rating_Binary)

importance(tuned_rf)  # Numeric values for %IncMSE and IncNodePurity
varImpPlot(tuned_rf) 


############################# FINALIZING MODEL SELECTION #################################
##### rf_model with no tuning - default metrics produce best resul with highest accuracy and sensitivity 
rf_final = rf_model 

print(rf_final)
rf_predictions <- predict(rf_final, newdata = test_data)
confusionMatrix(rf_predictions, test_data$Rating_Binary)

importance_final = importance(rf_final)  
varImpPlot(rf_final, main = "Final Model Feature Importance") 




######################################################################################
###################################CREATING GRAPHS, PLOTS#############################
# Use stargazer to create the table of Feature Importance 
stargazer(
  importance_final, 
  summary = FALSE, 
  rownames = TRUE, 
  type = "html", # Change to "html" or "latex" for other formats
  title = "Feature Importance Table",
  digits = 5
)


# Use stargazer to create the table of Final model Metrics 
conf_matrix <- confusionMatrix(rf_predictions, test_data$Rating_Binary)

results <- data.frame(
  Metric = c(
    "Accuracy", 
    "95% CI", 
    "No Information Rate", 
    "P-Value [Acc > NIR]", 
    "Kappa", 
    "Mcnemar's Test P-Value", 
    "Sensitivity", 
    "Specificity", 
    "Positive Predictive Value", 
    "Negative Predictive Value", 
    "Prevalence", 
    "Detection Rate", 
    "Detection Prevalence", 
    "Balanced Accuracy"
  ),
  Value = c(
    conf_matrix$overall["Accuracy"],
    paste0("(", round(conf_matrix$overall["AccuracyLower"], 4), 
           ", ", round(conf_matrix$overall["AccuracyUpper"], 4), ")"),
    conf_matrix$overall["AccuracyNull"],
    conf_matrix$overall["AccuracyPValue"],
    conf_matrix$overall["Kappa"],
    conf_matrix$overall["McnemarPValue"],
    conf_matrix$byClass["Sensitivity"],
    conf_matrix$byClass["Specificity"],
    conf_matrix$byClass["Pos Pred Value"],
    conf_matrix$byClass["Neg Pred Value"],
    conf_matrix$byClass["Prevalence"],
    conf_matrix$byClass["Detection Rate"],
    conf_matrix$byClass["Detection Prevalence"],
    conf_matrix$byClass["Balanced Accuracy"]
  )
)

# Use stargazer to create the table
stargazer(
  results, 
  summary = FALSE, 
  rownames = FALSE, 
  type = "html", # Change to "html" or "latex" for other formats
  title = "Confusion Matrix Metrics",
  digits = 4
)


##################################################################################

############ Create the Chart for Rating Variable ###############

data$Rating_Bin <- cut(
  data$Rating,
  breaks = c(0.99, 1.99, 2.99, 3.99, Inf),  
  labels = c("1 to 2", "2 to 3", "3 to 4", "4+"),
  include.lowest = FALSE
)

# Create the histogram
rating_histogram <- ggplot(data, aes(x = Rating)) +
  geom_histogram(binwidth = 0.5, fill = "darkgreen", color = "black", alpha = 0.7) +  # Adjust bin width as needed
  labs(
    title = "Distribution of Ratings",
    x = "Rating",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  )

# Print the histogram
print(rating_histogram)


# Calculate percentages for each bin
rating_distribution <- data %>%
  count(Rating_Bin) %>%  # Count occurrences of each bin
  mutate(Percentage = (n / sum(n)) * 100)  # Calculate percentage

rating_distribution <- rating_distribution %>%
  mutate(Legend_Label = paste0(Rating_Bin, " (", round(Percentage, 1), "%)"))

# Create the pie chart
pie_chart <- ggplot(rating_distribution, aes(x = "", y = Percentage, fill = Legend_Label)) +
  geom_bar(stat = "identity", width = 1) +  # Bar chart for pie slices
  coord_polar(theta = "y") +  # Convert bar chart to pie chart
  labs(
    title = "Percentage of Distribution of Ratings",
    fill = "Rating Range"
  ) +
  theme_void() +  # Minimalist theme for pie chart
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),  # Center the title
    legend.title = element_text(size = 14),  # Make legend title bigger
    legend.text = element_text(size = 12)    # Make legend text bigger
  ) +
  #theme_minimal()
  scale_fill_brewer(palette = "Set5")  # Optional: Use a color palette
#scale_color_discrete(name = "Rating Range")
# Print the pie chart
print(pie_chart)

##################################################################################

######################Create the chart for Numerical Variables#################### 
install.packages("patchwork")
library(patchwork)  
library(viridis)    

qq_plots <- list()
box_plots <- list()
scatter_plots <- list()

for (col in colnames(numerical_col)) {
  formula_numerical <- as.formula(paste("Rating ~", col))
  reg_numerical <- lm(formula_numerical, data = data)
  
  # 1. QQ Plot
  residuals_data <- data.frame(Residuals = residuals(reg_numerical))
  qq_plot <- ggplot(residuals_data, aes(sample = Residuals)) +
    stat_qq(size = 2, color = "steelblue") +
    stat_qq_line(color = "red", linetype = "dashed") +
    ggtitle(paste("QQ Plot of", col)) +
    theme_minimal(base_size = 12) +
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") +
    scale_color_viridis(discrete = FALSE) +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 2. Box Plot
  box_plot <- ggplot(data, aes_string(y = col)) +
    geom_boxplot(fill = viridis(1), alpha = 0.6) +
    ggtitle(paste("Box Plot of", col)) +
    theme_minimal(base_size = 12) +
    labs(x = "", y = col) +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 3. Scatter Plot
  scatter_plot <- ggplot(data, aes_string(x = col, y = "Rating")) +
    geom_point(alpha = 0.6, color = viridis(1)) +
    geom_smooth(method = "lm", color = "red", linetype = "dashed") +
    ggtitle(paste("Scatter Plot:", col, "vs Rating")) +
    theme_minimal(base_size = 12) +
    labs(x = col, y = "Rating") +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Store plots in their respective lists
  qq_plots[[col]] <- qq_plot
  box_plots[[col]] <- box_plot
  scatter_plots[[col]] <- scatter_plot
}

combined_plot <- (wrap_plots(qq_plots) / wrap_plots(box_plots) / wrap_plots(scatter_plots)) + 
  plot_annotation(title = "QQ, Box, and Scatter Plots for Numerical Variables")
print(combined_plot)



##################################################################################

######################Create the charts for categorical Variables#################### 


categorical_columns = names(data)[sapply(data, function(col) is.character(col))]

frequency_tables <- list()
categorical_columns <- names(data)[sapply(data, function(col) is.factor(col) || is.character(col))]


for (col in categorical_columns) {
  
  # Create a frequency table for the current categorical variable
  plot_data <- data %>%
    group_by(!!sym(col)) %>%
    summarize(Frequency = n()) %>%
    arrange(desc(Frequency)) # Sort by descending frequency
  
  # Limit to top 30 for specific variables
  if (col %in% c("Company...Maker.if.known.", "Specific.Bean.Origin.or.Bar.Name")) {
    plot_data <- plot_data %>%
      slice(1:30) # Keep only the top 30 categories
    title_suffix <- " (Top 30 Observations)" # Title suffix for specific variables
  } else {
    title_suffix <- "" # No suffix for other variables
  }
  
  # Dynamically create a bar chart for the categorical variable
  plot <- ggplot(plot_data, aes_string(x = col, y = "Frequency", fill = col)) +
    geom_bar(stat = "identity", width = 0.7) +
    geom_text(aes(label = Frequency), vjust = -0.5, size = 3.5) +  # Add labels above bars
    labs(
      title = paste("Frequency Distribution of", col, title_suffix), # Dynamic title
      x = col,
      y = "Frequency"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),  # Rotate x-axis labels
      legend.position = "none"  # Remove legend for bar charts
    ) +
    scale_fill_viridis(discrete = TRUE)
  
  # Print the bar chart
  print(plot)
}

for (col in categorical_columns) {
  
  # Filter data to include only the top 30 observations for specific variables
  if (col %in% c("Company...Maker.if.known.", "Specific.Bean.Origin.or.Bar.Name")) {
    top_categories <- data %>%
      group_by(!!sym(col)) %>%
      summarize(Frequency = n()) %>%
      arrange(desc(Frequency)) %>%
      slice(1:30) %>%
      pull(!!sym(col))
    
    filtered_data <- data %>%
      filter(!!sym(col) %in% top_categories)
    
    title_suffix <- " (Top 30 Observations)" # Add suffix to title
  } else {
    filtered_data <- data
    title_suffix <- "" # No suffix for other variables
  }
  
  # Dynamically create a plot for each categorical variable
  plot <- ggplot(filtered_data, aes_string(x = col, y = "Rating", fill = col)) +
    geom_boxplot(alpha = 0.8, outlier.color = "red", outlier.size = 1.5) +
    labs(
      title = paste("Box Plot of Ratings by", col, title_suffix), # Dynamic title
      x = col,
      y = "Rating"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),  # Rotate x-axis labels
      legend.position = "none"  # Remove legend for box plots
    ) +
    scale_fill_viridis(discrete = TRUE) # Optional: Use the Set3 color palette
  
  # Print the plot
  print(plot)
}
