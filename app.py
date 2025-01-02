from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline
import numpy as np
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        print("Request data:", request.form)
        user_ids = request.form.getlist('user_id')  # Expecting multiple user IDs
        movie_ids = request.form.getlist('movie_id')  # Expecting multiple movie IDs
        # Convert to numpy arrays
        user_ids_array = np.array([int(uid) for uid in user_ids])
        movie_ids_array = np.array([int(mid) for mid in movie_ids]) 
        print("Before Prediction")
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(user_ids_array, movie_ids_array)
        print("after Prediction")
        return render_template('result.html', prediction=results)
        
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
