import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv('housing.csv')

# Clean column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# Convert location
df['location'] = df['location'].astype('category').cat.codes

# Features
X = df[['area', 'bedrooms', 'location']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# Save
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(accuracy, open('accuracy.pkl', 'wb'))

print("✅ Model trained successfully!")
print("Accuracy:", accuracy)