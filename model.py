import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("../DATA/credit_card_fraud_dataset.csv")

print(df.head())

# Remove date column
df = df.drop("TransactionDate", axis=1)

# Convert text columns to numbers
for col in ["TransactionType", "Location"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("IsFraud", axis=1)
y = df["IsFraud"]
# Apply SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", round(precision*100,2), "%")
print("Recall:", round(recall*100,2), "%")
print("F1 Score:", round(f1*100,2), "%")
# Save model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
# Save accuracy to file
with open("accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))
    

print("Accuracy file created successfully!")
import numpy as np

np.save("confusion_matrix.npy", cm)

print("Confusion Matrix saved successfully!")
# Save Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("Feature Importance saved successfully!")
with open("metrics.txt", "w") as f:
    f.write(f"{accuracy*100:.2f}\n")
    f.write(f"{precision*100:.2f}\n")
    f.write(f"{recall*100:.2f}\n")
    f.write(f"{f1*100:.2f}\n")

print("Metrics file created successfully!")