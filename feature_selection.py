import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE

# Load the dataset
data = pd.read_csv('neo.csv')


X = data[['absolute_magnitude','est_diameter_min','est_diameter_max','relative_velocity','miss_distance']]
y = data['hazardous']

# Feature names can be extracted from the dataframe itself
feature_names = X.columns

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=50, random_state=42,max_depth=20, min_samples_split =3)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# SelectKBest with ANOVA
selector = SelectKBest(f_classif, k=3)
X_train_anova = selector.fit_transform(X_train, y_train)
X_test_anova = selector.transform(X_test)

# Extract selected feature names
selected_features = selector.get_support(indices=True)
selected_feature_names = feature_names[selected_features]
print("Selected features using ANOVA:", selected_feature_names)

# Train a RandomForestClassifier using the selected features
clf_anova = RandomForestClassifier(random_state=42)
clf_anova.fit(X_train_anova, y_train)
y_pred_anova = clf_anova.predict(X_test_anova)
print(classification_report(y_test, y_pred_anova))
print("//////////////////////////////////////////////////////")

# Recursive Feature Elimination (RFE) with RandomForestClassifier
clf_rfe = RandomForestClassifier(random_state=42)
rfe = RFE(clf_rfe, n_features_to_select=3)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Extract selected feature names (RFE uses the same selector object as ANOVA, reselecting is incorrect)
selected_features_rfe = rfe.get_support(indices=True)
selected_feature_names_rfe = feature_names[selected_features_rfe]
print("Selected features using RFE:", selected_feature_names_rfe)

# Train a RandomForestClassifier with RFE selected features
clf_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = clf_rfe.predict(X_test_rfe)
print(classification_report(y_test, y_pred_rfe))
