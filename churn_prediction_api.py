import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS
from category_encoders import BinaryEncoder

model = load('churn_prediction_model.joblib')

X = pd.read_csv('customer_churn_dataset.csv');

categorical_features = ['Gender', 'Subscription Type', 'Contract Length']
encoder = BinaryEncoder()
encoder.fit_transform(X[categorical_features])

api = Flask(__name__)
CORS(api)

@api.route('/api/customer_churn_prediction', methods=['POST'])
def customer_churn_prediction():
    try:
        data = request.json['inputs']
        input_df = pd.DataFrame(data)
        input_encoded = encoder.transform(input_df[categorical_features])
        input_df = input_df.drop(categorical_features, axis=1)
        input_encoded = input_encoded.reset_index(drop=True)
        
        final_input = pd.concat([input_df, input_encoded], axis=1)
        
        prediction = model.predict_proba(final_input)
        class_labels = model.classes_
        
        response = []
        for prob in prediction:
            prob_dict = {}
            for k, v in zip(class_labels, prob):
                prob_dict[str(k)] = round(float(v) * 100, 2)
            response.append(prob_dict)
        
        return jsonify({
            'status': 'success',
            'message': 'Prediction generated successfully.',
            'data': {
                'predictions': response
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while processing the request.',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    api.run(port=8000)
