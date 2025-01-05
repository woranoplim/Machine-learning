# **Meteor Hazard Prediction Project**

This project focuses on predicting the severity of meteors using machine learning techniques. Developed as part of a third-year Computer Science student project, the solution utilizes the Random Forest algorithm combined with feature selection methods to enhance model performance.

### **Project Description**

The goal of this project is to classify meteors based on their potential hazard levels. The dataset includes various meteor characteristics, and the classification outcomes are:

ชนแน่นอน (Will hit for sure)

มีโอกาสชน (Possible impact)

ไม่ชน (No impact)

###Key Features in the Dataset:

id: Unique identifier for each meteor.

name: Name of the meteor.

est_diameter_min and est_diameter_max: Estimated minimum and maximum diameters.

relative_velocity: Relative velocity of the meteor.

miss_distance: Distance of the meteor's closest approach.

orbiting_body: The celestial body the meteor orbits.

sentry_object: Boolean indicating if it's a sentry object.

absolute_magnitude: Brightness magnitude of the meteor.

hazardous: Boolean indicating if the meteor is hazardous.

### **Methods and Techniques**

1. Machine Learning Model

Algorithm: Random Forest

Selected for its robustness and ability to handle high-dimensional data.

2. Feature Selection Techniques

RFE (Recursive Feature Elimination): Used to identify the most important features by recursively removing the least significant ones.

ANOVA (Analysis of Variance): Applied to evaluate the statistical significance of individual features with respect to the target variable.

3. Data Preprocessing

Handling missing values.

Scaling numerical features.

Encoding categorical variables (e.g., orbiting_body).

4. Model Evaluation

Metrics:

Accuracy

Precision

Recall

F1-score

Tools and Technologies

Programming Language: Python

Libraries:

Pandas, NumPy for data manipulation.

Scikit-learn for machine learning.

Matplotlib, Seaborn for data visualization.

Results

Achieved a high accuracy score with balanced precision and recall.

Feature selection significantly reduced the dataset's dimensionality, improving computation time without compromising model performance.
