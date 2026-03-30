import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the stations data
stations_path = "model_files/stations.csv"
stations_df = pd.read_csv(stations_path)

# Create dummy features and labels
stations_df['label'] = (stations_df['sta_lat'] > stations_df['sta_lat'].mean()).astype(int)
X = stations_df[['sta_lat', 'sta_lon']]
y = stations_df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")

# Save the model
output_path = "model_files/japanmodel.pkl"
joblib.dump(model, output_path)
print(f"Model saved to {output_path}")
