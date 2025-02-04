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
    data = request.form.to_dict()
    features = [float(data['gre']), float(data['toefl']), float(data['university_rating']),
                float(data['sop']), float(data['lor']), float(data['cgpa']), float(data['research'])]
    prediction = model.predict([features])
    # Convert the prediction to a percentage
    percentage = prediction[0] * 100
    return jsonify({'Your chances of selection is: ': percentage})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
