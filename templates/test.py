import joblib
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

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

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def get_delay():
    result = request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']
    print(f"Query Text: {query_text}")  # Print query text for debugging
    
    # Replace this with your actual function to prepare the query
    query = get_all_query(query_title, query_author, query_text)  # Ensure this function is defined
    user_input = {'query': query}
    
    # Make prediction
    pred = pipeline.predict(query)
    print(f"Prediction: {pred}")  # Print prediction for debugging

    # Mapping prediction to labels
    dic = {1: 'real', 0: 'fake'}
    return f'<html><body><h1>{dic[pred[0]]}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'

if __name__ == '__main__':
    app.run(port=8080, debug=True)
