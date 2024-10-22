## User Behavior Classification Using CNN, Random Forest, and Logistic Regression

## Video:



https://github.com/user-attachments/assets/9aad9fb7-3386-49c3-bb54-5a2cab66f8ff



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
Final Accuracy: 99.29%
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

Overfitting Risk: The Random Forest model achieved perfect accuracy on the test data, which suggests that it may have overfitted to the data. Overfitting occurs when a model captures not just the underlying patterns but also the noise in the training data, leading to overly optimistic results on test data that is similar to the training data. This model might perform less well when evaluated on a completely new dataset.

Limited Dataset: The dataset used in this project may not be diverse enough to represent all potential variations in user behavior. A more extensive and varied dataset would provide a better assessment of how well the models can generalize to unseen data.

Class Imbalance: While the dataset appears balanced in terms of the number of instances per class, real-world mobile usage datasets often contain imbalanced classes, where certain types of user behaviors dominate. If such imbalance existed, it could affect model performance, leading to biases towards more frequent classes.

Simplistic Feature Set: The features used for prediction, such as app usage time, data usage, and age, are relatively simple. There might be other latent factors influencing user behavior that are not captured by these features, such as interaction patterns, app categories, or usage context (e.g., location, time of day).

### Future Improvements

Add Regularization and Tuning: To reduce overfitting, we could apply regularization techniques such as pruning in decision trees, adjusting hyperparameters like max_depth, or using cross-validation to fine-tune the model’s complexity. For the Random Forest model, reducing the number of trees or limiting the depth of each tree might help prevent overfitting.

Introduce Ensemble Learning: While we already used Random Forest, further ensemble techniques like Gradient Boosting or XGBoost could offer better performance by reducing variance and bias, potentially improving generalization to unseen data. These models can also offer more flexibility with regularization.

Explore Neural Networks Further: While the CNN performed well, deep learning models can also be prone to overfitting, especially on small datasets. Adding dropout layers or applying early stopping could help prevent overfitting. Additionally, experimenting with more complex architectures (e.g., LSTM for sequential patterns in time-based data) could provide better results if mobile data with a temporal component is available.

Feature Engineering: We could explore additional features that provide richer information about user behavior. For example, interactions between users and specific apps or clustering of similar behaviors could be valuable. More advanced techniques, such as dimensionality reduction using PCA, could also help simplify the feature space.

Gather More Data: Expanding the dataset with more diverse examples of mobile behavior would allow us to better understand how these models generalize to new data. We could also simulate realistic data with more variability, which would help the models learn patterns that are more representative of real-world mobile usage.

Evaluate Other Metrics Beyond Accuracy: Given that overfitting could be an issue, accuracy alone might not be the best indicator of model performance. In future analyses, it would be beneficial to focus more on metrics such as precision, recall, and F1-score for different classes. In particular, the classification report from models like Logistic Regression highlights the importance of evaluating performance across all classes, not just based on overall accuracy.
## Running the Code
Prerequisites
Python 3.x
Required Libraries: torch, pandas, numpy, seaborn, matplotlib, sklearn

