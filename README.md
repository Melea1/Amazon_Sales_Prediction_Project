# Amazon_Sales_Prediction_Project
Predicting sales volume based on Amazon data

Introduction

Amazon is one of the largest e-commerce platforms in the world, with millions of products and sellers. Understanding and predicting sales volume on Amazon is extremely valuable for both businesses and data scientists. Accurate sales prediction can help sellers plan inventory, adjust pricing strategies, optimize marketing budgets, and anticipate market demand trends. From a data science perspective, this problem is also fascinating because it combines numerical and categorical features, non-linear patterns, and noise, making it an excellent challenge for testing and improving machine learning models.

The goal of this project was therefore to predict the sales volume of Amazon products using product characteristics and contextual information, while learning how to manage overfitting and improve model generalization.

⸻

Step 1: Data Preparation

The first phase consisted of importing the dataset and performing an initial inspection to understand the structure and variable types. We checked for missing values, duplicates, and inconsistent records.
We handled missing values by applying appropriate imputation techniques depending on the type of variable (mean/median for numerical, mode for categorical). Duplicates were removed to avoid biasing the models. All numerical columns were converted into consistent data types, and irrelevant columns were dropped to reduce noise.

We then separated the dataset into training and testing sets 80%/20% to make sure we could evaluate model performance on unseen data.

⸻

Step 2: Exploratory Data Analysis (EDA)

In the EDA phase, we analyzed the main statistical properties of the dataset. We visualized feature distributions, correlations, and potential outliers. The correlation heatmap helped identify relationships between key numerical variables such as price, ratings, and previous sales volume.

We observed that some features had strong linear or non-linear relationships with sales, while others added little predictive value. This phase also helped detect multicollinearity and imbalanced feature distributions, guiding later feature engineering and model selection.

⸻

Step 3: Data Cleansing and Feature Engineering

After cleaning the data, we performed feature engineering to enrich the dataset.
This included creating derived variables that might better explain the target (for example: log transformation of sales volume, price ratios, review scores, or binary indicators for promotions).

We also encoded categorical features using One-Hot Encoding and Label Encoding depending on the model family. Continuous variables were scaled or normalized when necessary, mainly for linear models.

By the end of this step, we had a structured, numerical dataset ready for machine learning.

⸻

Step 4: Model Selection

We experimented with several models to understand their behavior and compare performance:
	•	Linear Regression — served as the baseline model. It achieved an R² of about 0.25, which indicated underfitting: the model was too simple to capture complex patterns in the data.
	•	Decision Tree — achieved a training R² of about 0.991. This extremely high score showed severe overfitting, as the model memorized the training data instead of learning general patterns.
	•	Random Forest — reached around 0.95 on the training data, performing better than the single tree but still showing overfitting signs.
	•	AdaBoost — achieved about 0.88, a more balanced result.
	•	Gradient Boosting — reached around 0.994 on training, again showing high variance.

At this stage, we clearly observed the difference between underfitting (too simple models) and overfitting (too complex models).

⸻

Step 5: Model Evaluation and Diagnostics

The main evaluation metrics used were R², RMSE (Root Mean Square Error), and MAE (Mean Absolute Error).
We noticed that tree-based models produced very low errors on the training data but were likely to perform worse on the test set, which confirmed overfitting.

To make the evaluation fair and reliable, we implemented cross-validation (k-fold) to obtain an average performance estimate and standard deviation, allowing a more robust model comparison.

⸻

Step 6: Fine-Tuning and Regularization

To reduce overfitting and improve generalization, we performed hyperparameter tuning using GridSearchCV and RandomizedSearchCV.

We adjusted:
	•	max_depth to limit tree complexity,
	•	min_samples_leaf and min_samples_split to prevent memorization of small details,
	•	n_estimators and max_features for Random Forest,
	•	learning_rate and subsample for Gradient Boosting and AdaBoost.

We also applied regularization for linear models (Ridge, Lasso) to control coefficient magnitudes.

After tuning, the gap between training and testing performance began to shrink, indicating better generalization and reduced overfitting.

⸻

Step 7: Key Insights and Interpretation

Through this project, we learned that:
	1.	Linear Regression is useful as a baseline but too simple for Amazon sales data.
	2.	Tree-based models like Decision Tree and Gradient Boosting can capture complex relationships but need regularization and cross-validation.
	3.	Random Forest and AdaBoost provided the best balance between bias and variance.
	4.	The most crucial step in achieving robust performance was not model choice but data preparation, feature quality, and tuning.

⸻

Step 8: Conclusion

This Amazon sales prediction project showed the full machine learning workflow — from raw data to model evaluation.
We explored how different algorithms behave, how to detect and reduce overfitting, and how to balance complexity with generalization.

The next steps could include:
	•	Testing on completely unseen Amazon data or other e-commerce sources.
	•	Deploying the final model through a Streamlit app for interactive predictions.
	•	Adding time-series features or text-based attributes (e.g., product descriptions) for better accuracy.

Overall, this project demonstrates a structured, real-world application of data science: predicting business outcomes from product data while ensuring statistical validity and model reliability.
