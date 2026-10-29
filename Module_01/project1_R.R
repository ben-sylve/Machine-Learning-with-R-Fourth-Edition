# -----------------------------------------------------------------------

# You will use the Taiwan Credit Card Default dataset to build and evaluate a k-Nearest Neighbors classification 
# model that predicts whether a client will default on their credit card payment next month. This project will 
# challenge you to preprocess real-world data, choose an optimal value for k, and interpret classification results.

# -----------------------------------------------------------------------


# Dataset
# Dataset name: Default of Credit Card Clients Dataset
# Source: UCI Machine Learning Repository
# Link: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clientsLinks to an external site.

# Download Instructions:
# Go to the link above
# Download the dataset (default of credit card clients.xls)
# Save it as credit_default.csv (you may convert it in Excel or R)
# Features (Selected for this project):
# LIMIT_BAL: Credit limit
# AGE: Age of the client
# EDUCATION: Education level
# PAY_1, PAY_2: Repayment status in the last two months
# BILL_AMT1: Bill amount from the previous month
# default_payment_next_month: Target variable (0 = No Default, 1 = Default)


# -----------------------------------------------------------------------

# Step 1: Load and Prepare the Data
# Read in the CSV using read.csv() or readxl::read_excel() and rename it credit_data
#
# Sample from Chapter 3 (Chapter_03.R):
# wbcd <- read.csv("wisc_bc_data.csv")
# Sample from Chapter 3 (Chapter_03.R):
# wbcd <- wbcd[-1]

# 1.1 read in csv

cc_data <- read.csv("credit_default.csv")
# get rid of the first row (column names) and rename the columns
colnames(cc_data) <- cc_data[1, ]
cc_data <- cc_data[-1, ]

# 1.2 Select only the columns: LIMIT_BAL, AGE, EDUCATION, PAY_1, PAY_2, BILL_AMT1, and default_payment_next_month
cc_data <- cc_data[, c("LIMIT_BAL", "AGE", "EDUCATION", "PAY_0", "PAY_2", "BILL_AMT1", "default payment next month")]



# 1.3 Check for missing values and summarize the dataset

# Sample from Chapter 3 (Chapter_03.R):
# str(wbcd)
# table(wbcd$diagnosis)
# wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"),
#                          labels = c("Benign", "Malignant"))
# round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)
# summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

# Summarize missing data

colSums(is.na(cc_data))
summary(cc_data)

# Make sure the features are numeric
cc_data$LIMIT_BAL <- as.numeric(cc_data$LIMIT_BAL)
cc_data$AGE <- as.numeric(cc_data$AGE)
cc_data$EDUCATION <- as.numeric(cc_data$EDUCATION)
cc_data$PAY_0 <- as.numeric(cc_data$PAY_0)
cc_data$PAY_2 <- as.numeric(cc_data$PAY_2)
cc_data$BILL_AMT1 <- as.numeric(cc_data$BILL_AMT1)
# Make the default_payment_next_month column a factor 
cc_data$`default payment next month` <- factor(cc_data$`default payment next month`, levels = c(0, 1),
                                                labels = c("No", "Yes"))

summary(cc_data)

# 1.4 Normalize all numeric features.

# Sample from Chapter 3 (Chapter_03.R):

# normalize <- function(x) {
#   return ((x - min(x)) / (max(x) - min(x)))
# }
# normalize(c(1, 2, 3, 4, 5))
# normalize(c(10, 20, 30, 40, 50))

# wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
# summary(wbcd_n$area_mean)

# use the scale() function to z-score standardize a data frame
# wbcd_z <- as.data.frame(scale(wbcd[-1]))

# Normalize using scale function
cc_data_n <- as.data.frame(scale(cc_data[, c("LIMIT_BAL", "AGE", "EDUCATION", "PAY_0", "PAY_2", "BILL_AMT1")]))


# -----------------------------------------------------------------------
# Step 2: Split the Data
# Set a seed for reproducibility to 25
# Split the dataset into 80% training and 20% testing sets
#
# Sample from Chapter 3 (Chapter_03.R):
# wbcd_train <- wbcd_n[1:469, ]
# wbcd_test <- wbcd_n[470:569, ]
#
# Ensure that the class labels are stored separately from the features
#
# Sample from Chapter 3 (Chapter_03.R):
# wbcd_train_labels <- wbcd[1:469, 1]
# wbcd_test_labels <- wbcd[470:569, 1]

# Set seed for reproducibility
set.seed(25)

# Shuffle the rows so the 80/20 split is random
shuffled_rows <- sample(nrow(cc_data_n))
cc_data_n <- cc_data_n[shuffled_rows, ]
cc_data <- cc_data[shuffled_rows, ]

# Get number of rows for 80% and 20% split
train_size <- floor(0.8 * nrow(cc_data_n))

# Create training and test sets
cc_data_train <- cc_data_n[1:train_size, ]
cc_data_test <- cc_data_n[(train_size + 1):nrow(cc_data_n), ]

# Create labels for training and test data
cc_data_train_labels <- cc_data[1:train_size, "default payment next month"]
cc_data_test_labels <- cc_data[(train_size + 1):nrow(cc_data), "default payment next month"]

# -----------------------------------------------------------------------
# Step 3: Apply the k-NN Algorithm

# Use the class::knn() function
# Predict default_payment_next_month for the test set
# Train and evaluate models using k = 3, 7, and 15

# Sample from Chapter 3 (Chapter_03.R):
# library(class)
# wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test,
#                       cl = wbcd_train_labels, k = 21)
#
# wbcd_train <- wbcd_n[1:469, ]
# wbcd_test <- wbcd_n[470:569, ]

# k_values <- c(1, 5, 11, 15, 21, 27)
#
# for (k_val in k_values) {
#   wbcd_test_pred <- knn(train = wbcd_train,
#                         test = wbcd_test,
#                         cl = wbcd_train_labels,
#                         k = k_val)
#   CrossTable(x = wbcd_test_labels,
#              y = wbcd_test_pred,
#              prop.chisq = FALSE)
# }
#

# -----------------------------------------------------------------------
# Step 4: Evaluate and Compare Performance
# For each k-value:
#
# Produce a confusion matrix
#
# Sample from Chapter 3 (Chapter_03.R):
# library(gmodels)
# CrossTable(x = wbcd_test_labels, y = wbcd_test_pred,
#            prop.chisq = FALSE)
#
# Report accuracy, precision, recall, and F1-score
#
# If possible, create a table comparing all three values of k
#

library(class)
library(gmodels)

# Train and evaluate models using k = 3, 7, and 15
k_values <- c(3, 7, 15)

# Table comparing all k values
results <- data.frame(k = integer(), accuracy = numeric(), precision = numeric(),
                       recall = numeric(), f1_score = numeric())

for (k_val in k_values) {
  cat("===== k =", k_val, "=====\n")

  cc_data_test_pred <- knn(train = cc_data_train,
                           test = cc_data_test,
                           cl = cc_data_train_labels,
                           k = k_val)
  
  # Evaluate model performance
  # Include accuracy, precision, recall, and F1-score
  CrossTable(x = cc_data_test_labels,
             y = cc_data_test_pred,
             prop.chisq = FALSE)

  # Confusion matrix
  confusion_matrix <- table(Actual = cc_data_test_labels, Predicted = cc_data_test_pred)
  print(confusion_matrix)

  # Accuracy, precision, recall, and F1-score (treating "Yes" as the positive class)
  TN <- confusion_matrix["No", "No"]
  FP <- confusion_matrix["No", "Yes"]
  FN <- confusion_matrix["Yes", "No"]
  TP <- confusion_matrix["Yes", "Yes"]

  accuracy <- (TP + TN) / sum(confusion_matrix)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)

  results <- rbind(results, data.frame(k = k_val, accuracy = accuracy, precision = precision,
                                        recall = recall, f1_score = f1_score))
}

# Print the comparison table after the loop
print(results)



# -----------------------------------------------------------------------
# Step 5: Visualize the Results
# Plot accuracy vs. k values using ggplot2
# Include a bar plot or table of precision and recall for additional insight
# (No sample available from Chapter_03.R - the chapter does not cover visualization with ggplot2)

library(ggplot2)

# Bar chart of accuracy, precision, and recall for each k
results_long <- data.frame(
  k = rep(results$k, 3),
  metric = rep(c("accuracy", "precision", "recall"), each = nrow(results)),
  value = c(results$accuracy, results$precision, results$recall)
)

ggplot(results_long, aes(x = factor(k), y = value, fill = metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)), position = position_dodge(width = 0.9), vjust = -0.3) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Accuracy, Precision, and Recall by k", x = "k", y = "Value", fill = "Metric") +
  theme_minimal()

