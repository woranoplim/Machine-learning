import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import AgglomerativeClustering, KMeans

data = pd.read_csv('neo.csv')

X = data[['absolute_magnitude','est_diameter_min','est_diameter_max','relative_velocity','miss_distance']]
y = data['hazardous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

fig = plt.figure(1,figsize=(8,6))
plt.clf()
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = LogisticRegression()
scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold CV
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())



