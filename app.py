from flask import Flask, render_template, request, jsonify
import pickle

# Load the model
try:
    model = pickle.load(open('admission_model.pkl', 'rb'))
except FileNotFoundError:
    print("Model file 'admission_model.pkl' not found.")
    exit()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Debugging: Print received form data
        print("Received data:", data)  

        features = []

        for key in ['gre', 'toefl', 'university_rating', 'sop', 'lor', 'cgpa', 'research']:
            try:
                if key not in data:
                    return jsonify({'error': f'Missing input for {key}.'}), 400

                value = float(data[key])  # Convert input to float

                # Validate the range
                if key == 'gre':
                    assert 260 <= value <= 340  # GRE valid range
                elif key == 'toefl':
                    assert 0 <= value <= 120  # TOEFL valid range
                elif key == 'cgpa':
                    assert 0 <= value <= 10  # CGPA valid range
                elif key in ['university_rating', 'sop', 'lor']:
                    assert 1 <= value <= 5  # Ratings range from 1 to 5
                
                features.append(value)
            except (ValueError, AssertionError):
                return jsonify({'error': f'Invalid input for {key}. Please enter a valid number in the correct range.'}), 400
        
        # Convert research to integer (0 or 1)
        features[-1] = int(features[-1])

        # Make prediction
        prediction = model.predict([features])
        percentage = prediction[0] * 100
        return jsonify({'prediction': percentage})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Use debug=False for production
