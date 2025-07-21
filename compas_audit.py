# --- Import necessary libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aif360.datasets import CompasDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Load and prepare the dataset ---
dataset = CompasDataset()

# Define privileged and unprivileged groups (race)
privileged_groups = [{'race': 1}]   # Caucasian
unprivileged_groups = [{'race': 0}] # African-American

# Convert to DataFrame for plotting
df, _ = dataset.convert_to_dataframe()

# Attempt to add risk score or risk category column to dataframe
# Check if dataset has 'scores' or 'raw_scores' attribute
if hasattr(dataset, 'scores'):
    df['risk_score'] = dataset.scores
elif hasattr(dataset, 'raw_scores'):
    df['risk_score'] = dataset.raw_scores
else:
    # If no risk score attribute, use labels as proxy
    df['risk_score'] = dataset.labels.ravel()

print("DataFrame columns:", df.columns)
print("DataFrame sample data:")
print(df.head())

# --- Advanced Visualization 1: Risk score distribution ---
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='risk_score', hue='race')
plt.title("Distribution of COMPAS Risk Scores by Race")
plt.xlabel("Risk Category")
plt.ylabel("Count")
plt.legend(title="Race", labels=["African-American", "Caucasian"])
plt.tight_layout()
plt.show()

# --- Split into training and test sets ---
dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

# Extract features and labels
X_train = dataset_train.features
y_train = dataset_train.labels.ravel()
X_test = dataset_test.features
y_test = dataset_test.labels.ravel()

# --- Standardize features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train a simple logistic regression model ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# --- Add predictions to dataset for bias analysis ---
dataset_test_pred = dataset_test.copy()
dataset_test_pred.labels = y_pred

# --- Bias Metrics ---
metric_test = BinaryLabelDatasetMetric(dataset_test, unprivileged_groups, privileged_groups)
classified_metric = ClassificationMetric(dataset_test, dataset_test_pred, unprivileged_groups, privileged_groups)

# --- Statistical Bias Summary ---
spd = metric_test.statistical_parity_difference()
di = metric_test.disparate_impact()

print("Statistical Parity Difference:", spd)
print("Disparate Impact Ratio:", di)
print(df.columns.tolist())
print(dataset.feature_names)
print(dataset.label_names)
print(dataset.protected_attribute_names)
print(dataset.metadata)
print(dataset.features[:5])
print(dataset.labels[:5])

# --- Advanced Visualization 2: False Positive Rates ---
fpr_unpriv = classified_metric.false_positive_rate(False)
fpr_priv = classified_metric.false_positive_rate(True)

plt.figure(figsize=(8, 5))
plt.bar(['African-American', 'Caucasian'], [fpr_unpriv, fpr_priv], color=['red', 'blue'])
plt.title('False Positive Rate by Race')
plt.ylabel('FPR')
plt.tight_layout()
plt.show()

# --- Advanced Visualization 3: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Recidivism", "Recidivism"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression on COMPAS")
plt.tight_layout()
plt.show()
