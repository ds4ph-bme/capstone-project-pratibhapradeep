## User Behavior Classification Using CNN, Random Forest, and Logistic Regression

.mp4


## Project Overview

This project is focused on predicting user behavior classes based on their mobile usage patterns. We analyze the dataset using three distinct models: a Convolutional Neural Network (CNN), a Random Forest Classifier, and Logistic Regression. The project involves extensive data preprocessing, feature selection, and model evaluation using cross-validation.

### Background and Motivation

In today's data-driven world, understanding user behavior is crucial for developing personalized mobile applications and enhancing user experience. The rise of mobile device usage has prompted researchers and businesses alike to analyze how individuals interact with their mobile devices. The dataset used in this project provides a rich set of features, including app usage time, data usage, battery consumption, and more, which offer insights into different user behavior classes. This analysis aims to predict user behavior classes based on these mobile usage patterns using machine learning models. By accurately classifying user behavior, we can potentially offer tailored mobile experiences, optimize app performance, and make data-driven decisions in user engagement strategies.

### Key Project Components:
1. Data Loading and Preprocessing: Cleaning and transforming the raw data for modeling.
2. Modeling:
   CNN: Utilized to capture complex patterns and relationships between features.
   Random Forest Classifier: Leveraged for its ability to handle structured data and provide insights through feature importance.
   Logistic Regression: Used as a baseline to compare with more complex models.
3. Cross-validation: Used for all models to mitigate overfitting and assess performance more reliably.
4. Evaluation: Models are evaluated using accuracy, confusion matrices, precision, recall, and feature importance visualization.

### Dataset
The dataset contains mobile usage data of various users, with features such as:

App Usage Time (min/day)
Data Usage (MB/day)
Battery Drain (mAh/day)
Screen On Time (hours/day)
Number of Apps Installed
Age

The target variable is the User Behavior Class, which categorizes users into different behavioral classes based on their mobile usage patterns.

## Data Preprocessing
Steps:
Drop Unnecessary Columns: User ID, Device Model, Operating System, and Gender were dropped as they did not contribute to behavior prediction.
Feature Selection: After analyzing correlations, features like Battery Drain, Number of Apps Installed, and Screen On Time were removed to improve model performance.
Scaling: Data was normalized using StandardScaler to ensure consistent scaling across features.

A correlation heatmap was created to visualize feature relationships with the target variable.

## Models
### Convolutional Neural Network (CNN)
CNN was chosen for its strength in identifying patterns in the data. The model was implemented using torch and trained over 200 epochs with the following parameters:

Loss function: CrossEntropyLoss
Optimizer: Adam with a learning rate of 0.001
Final Accuracy: 99.29%
Loss Plot: The model's loss decreased steadily during training over 200 epochs.

### Random Forest Classifier
Random Forest was chosen due to its ability to handle structured data and generate feature importance insights. The model used 10 estimators with a maximum depth of 10.

Test Accuracy: 100% (indicating overfitting)
Cross-Validation Accuracy: 99.82%
Feature Importance: The most important features were App Usage Time (min/day) and Data Usage (MB/day). Age had the least impact on behavior classification.

### Logistic Regression
Logistic Regression was selected as a baseline model for comparison. Although simpler, it provided valuable insights into model performance in the presence of class imbalance.

Accuracy: 76.43%
Cross-Validation Accuracy: 73.39%


## Results & Evaluation
### CNN
Final Accuracy: 98.57%
Confusion Matrix: The CNN struggled with certain classes due to imbalances in the data.

Precision/Recall:
Class 0 - Precision: 0.48, Recall: 1.00
Class 2 - Precision: 1.00, Recall: 0.88
Class 4 - Precision: 0.46, Recall: 1.00

### Random Forest
Final Accuracy: 100% (overfitting)
Cross-Validation Accuracy: 99.82%

Feature Importance:
1. App Usage Time
2. Data Usage
3. Age (least important)

### Logistic Regression
Final Accuracy: 76.43%
Confusion Matrix: Logistic Regression struggled the most with class imbalance, especially with class 1 and 2.
Precision/Recall:
Class 1 - Precision: 0.59, Recall: 1.00
Class 2 - Precision: 0.59, Recall: 0.34
Class 5 - Precision: 0.87, Recall: 1.00

## Conclusion

This project implemented and compared three different models for user behavior classification:

The CNN provided the best accuracy, capturing complex patterns in the data.
The Random Forest was prone to overfitting but provided valuable feature importance insights.
The Logistic Regression served as a baseline and exposed challenges in handling class imbalance.

### Impact of Findings:
Feature Insights: App Usage Time and Data Usage were identified as the most influential features in determining user behavior. This can guide further studies or applications in mobile usage analysis.
Model Comparison: The CNN's superior performance shows the advantage of deep learning in classifying structured data, while Random Forest offered insights into feature importance.

### Limitations

Class Imbalance: The dataset has imbalanced classes, which affected the models’ ability to accurately classify certain groups. This could be addressed with techniques like SMOTE or class weighting.

Overfitting: The Random Forest achieved 100% accuracy, indicating overfitting. Limiting tree depth or using more cross-validation folds may improve generalization.

Feature Correlation: Several features, such as App Usage Time and Data Usage, are highly correlated. More advanced feature selection techniques could reduce redundancy.

Model Complexity: The CNN performed well but was computationally intensive. Simpler models may offer similar accuracy with less overhead.

Interpretability: While the Random Forest provides some feature importance, the CNN model lacks interpretability. Adding explainability techniques could improve understanding of model decisions.

Generalization: The models were trained on a single dataset, so testing on more diverse data could ensure better real-world performance.

### Future Improvements

Future analyses could explore more advanced techniques to address class imbalance, such as using Synthetic Minority Over-sampling Technique (SMOTE) or other resampling methods. Additionally, other deep learning architectures beyond CNN, such as LSTMs, could be tested, particularly if time-series or sequential data is included in future datasets. Another avenue for future exploration could involve feature engineering to derive more meaningful insights from the raw data or incorporating external datasets to improve prediction accuracy.

## Running the Code
Prerequisites
Python 3.x
Required Libraries: torch, pandas, numpy, seaborn, matplotlib, sklearn

