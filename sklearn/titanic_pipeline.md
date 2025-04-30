### Tatanic Survior

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data_dir = r"D:\Non_Documents\2025\datasets\titanic"
data_path = os.path.join(data_dir, "Titanic-Dataset.csv")
df = pd.read_csv(data_path)

x_data = df.drop("Survived", axis=1)
y_data = df["Survived"]

x_data.head(10)
```

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

class FamilySize(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        return X

num_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
cat_features = ["Pclass", "Sex", "Embarked"]

column_selector = ColumnTransformer(
    [("selected", "passthrough", num_features + cat_features)], remainder="drop"
)

feature_engineering = Pipeline(steps=[
    ("family_size", FamilySize()),
    # ("selector", column_selector)
])


num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scalar", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

pipeline = Pipeline(steps=[
    ("feature_eng", feature_engineering),
    ("preprocess", preprocessor),
    ("pca", PCA(n_components=10)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
pipeline.fit(x_train, y_train)
```

```python
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

y_pred = pipeline.predict(x_test)
y_proba = pipeline.predict_proba(x_test)[:, 1]

print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"roc_auc : {roc_auc_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
```
