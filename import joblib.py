import joblib
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template

# Custom unpickler for handling legacy class names
class CustomUnpickler(joblib.Parallel):
    def find_class(self, module, name):
        if name == 'logistic':
            return LogisticRegression
        return super().find_class(module, name)

# Load model
def load_custom_model(file_path):
    with open(file_path, 'rb') as file:
        return joblib.load(file)

model_path = 'C:/Users/58in/Downloads/Fake_news_detection1/Fake_news_detection/pipeline.sav'

try:
    pipeline = load_custom_model(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Model file not found at: {model_path}")
except Exception as e:
    print("Error loading model:", e)


