import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler


data = pd.read_csv("Breast_Cancer_dataset.csv")

data.columns = data.columns.str.strip(" ")
columns_to_standardize = ['Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']

robust_scaler = RobustScaler()
data[columns_to_standardize] = robust_scaler.fit_transform(data[columns_to_standardize])

scaler = StandardScaler()
data[['Age']] = scaler.fit_transform(data[['Age']])

data_scaled = data

data_scaled = data_scaled.drop(columns=['differentiate'], axis=1)

print(data_scaled.head(10))


scale_mapper = {
    "IIA": 0,
    "IIB": 1,
    "IIIA": 2,
    "IIIB": 3,
    "IIIC": 4
}

data_scaled['6th Stage'] = data_scaled['6th Stage'].map(scale_mapper)
scale_mapper = {
    "N1": 0,
    "N2": 1,
    "N3": 2
}

data_scaled['N Stage'] = data_scaled['N Stage'].map(scale_mapper)
scale_mapper = {
    "T1": 0,
    "T2": 1,
    "T3": 2,
    "T4": 3
}

data_scaled['T Stage'] = data_scaled['T Stage'].map(scale_mapper)

mapper = {'Alive': 0, 'Dead': 1}
data_scaled['Status'] = data_scaled['Status'].map(mapper)

print(data_scaled.head(10))

data_scaled['A Stage'].astype('category')
data_scaled['Estrogen Status'].astype("category")
data_scaled['Progesterone Status'].astype("category")
data_scaled['Status'].astype('category')

#print(data_scaled.columns)

sample = pd.get_dummies(data_scaled)
#sample.head()

#print(sample.columns)

features = sample.columns.drop("Status")


X = sample.drop('Status',axis=1).values
y = sample['Status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "C4.5 Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        "Accuracy": accuracy,
        "Classification Report": classification_report(y_test, y_pred)
    }


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results).T
print(results_df)
