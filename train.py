import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Check dataset structure
print(df.iloc[1140:1145])  # Adjust the range as needed


# Define features and target
X = df.drop(columns=['label'])  # All columns except crop name
y = df['label']  # Crop name as the target variable


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(y_train.value_counts())  


# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training complete. Saved as model.pkl")
