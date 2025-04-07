import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example test input (change values to test different cases)
test_input = [[20, 50, 60, 30, 70, 6.5, 150]]  # Modify these values

# Make prediction
predicted_crop = model.predict(test_input)
print("Recommended Crop:", predicted_crop[0])
