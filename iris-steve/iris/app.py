from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import os
import pickle
import numpy as np

def create_app():
    '''
    Create app factory
    '''

    load_dotenv()

    app = Flask(__name__)

    # Path to save/load the pickled model
    MODEL_PATH = 'iris_model.pkl'

    @app.route('/')
    def index():
        return "Hello, from Iris!"


    @app.route('/iris')
    def iris():
        """Return the class predictions for the Iris dataset based on Logistic Regression"""
        X, y = load_iris(return_X_y=True)
        clf = LogisticRegression(
            random_state=0,
            solver='lbfgs',
            multi_class='multinomial'
        ).fit(X, y)
        
        return str(clf.predict(X[:2, :]))
    

    @app.route('/iris/score')
    def iris_score():
        """Return the mean accuracy of the logistic regression model on the Iris dataset."""
        X, y = load_iris(return_X_y=True)
        clf = LogisticRegression(
            random_state=0,
            solver='lbfgs',
            multi_class='multinomial'
        ).fit(X, y)
        score = clf.score(X, y)
        return f"Logistic Regression model accuracy on Iris dataset: {score:.4f}"
    

    @app.route('/iris/predict', methods=['POST'])
    def iris_predict():
        """Predict Iris class labels or probabilities from input data."""
        try:
            # Get JSON data from request
            data = request.get_json()
            if not data or 'features' not in data:
                return jsonify({"error": "Missing 'features' in JSON payload"}), 400
            
            # Convert feature to numpy array
            features = np.array(data['features'])
            if features.shape[1] != 4: # expect 4 features (sepal length, sepal width, petal length, petal width)
                return jsonify({"error": "Each input must have exactly 4 features"}), 400
            
            # Train model
            X, y = load_iris(return_X_y=True)
            clf = LogisticRegression(
                random_state=0,
                solver='lbfgs',
                multi_class='multinomial'
            ).fit(X, y)

            # Get predictions or probabilities
            return_probs = request.args.get('probabilities', 'false').lower() == 'true'
            if return_probs:
                predictions = clf.predict_proba(features).tolist()
                return jsonify({"probabilities": predictions})
            else:
                predictions = clf.predict(features).tolist()
                return jsonify({"labels": predictions})
        
        except ValueError as e:
            return jsonify({"error": f"Invalid input data: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": f"Server error: {str(e)}"}), 500
    

    @app.route('/iris/save_model', methods=['POST'])
    def save_model():
        """Train and save the logistic regression model to a file."""
        try:
            X, y = load_iris(return_X_y=True)
            clf = LogisticRegression(
                random_state=0,
                solver='lbfgs',
                multi_class='multinomial'
            ).fit(X, y)

            # Pickle the model
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(clf, f)

            # Verify pickling
            pickled_model = pickle.dumps(clf)
            unpickled_model = pickle.loads(pickled_model)
            score = unpickled_model.score(X, y)
            return jsonify({"message": "Model saved successfully", "score": score})

        except Exception as e:
            return jsonify({"error": f"Failed to save model: {str(e)}"}), 500
    

    @app.route('/iris/load_predict', methods=['POST'])
    def load_predict():
        """Load the pickled model and predict from input data."""
        try:
            # Load the pickled model
            with open(MODEL_PATH, 'rb') as f:
                clf = pickle.load(f)

            # Get JSON data
            data = request.get_json()
            if not data or 'features' not in data:
                return jsonify({"error": "Missing 'features' in JSON payload"}), 400

            features = np.array(data['features'])
            if features.shape[1] != 4:
                return jsonify({"error": "Each input must have exactly 4 features"}), 400

            # Predict or get probabilities
            return_probs = request.args.get('probabilities', 'false').lower() == 'true'
            if return_probs:
                predictions = clf.predict_proba(features).tolist()
                return jsonify({"probabilities": predictions})
            else:
                predictions = clf.predict(features).tolist()
                return jsonify({"labels": predictions})

        except FileNotFoundError:
            return jsonify({"error": "Model file not found. Save the model first using /iris/save_model"}), 404
        except Exception as e:
            return jsonify({"error": f"Server error: {str(e)}"}), 500


    return app
