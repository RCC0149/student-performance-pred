import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import pickle

# Load dataset
df = pd.read_csv('StudentsPerformance.csv')

# Features and target
X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Split data (no stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Calculate class weights
unique, counts = np.unique(y_train, return_counts=True)
class_weights = dict(zip(unique, 1 / counts))  # Inverse frequency
max_weight = max(class_weights.values())
class_weights = {k: (v / max_weight) * 0.9 for k, v in class_weights.items()}  # Amplification

# Define XGBoost classifier
clf = xgb.XGBClassifier(
    colsample_bytree=0.7,
    gamma=0,
    learning_rate=0.3,
    max_depth=3,
    n_estimators=100,
    reg_alpha=0.8,
    reg_lambda=1.0,
    objective='multi:softmax',
    num_class=len(np.unique(y_encoded)),
    random_state=42,
    eval_metric='mlogloss'
)

# Train model
clf.fit(X_train, y_train, sample_weight=[class_weights[label] for label in y_train])

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Evaluate
y_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {test_accuracy:.2f}')
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))