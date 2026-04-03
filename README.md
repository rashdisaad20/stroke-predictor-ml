#  Heart Stroke Risk Prediction System
**An End-to-End Machine Learning Solution for Clinical Risk Assessment**

##  Executive Summary
This project provides a data-driven approach to identifying high-risk stroke patients. By leveraging clinical parameters such as blood pressure, glucose levels, and smoking history, the system outputs a probability score using a trained Logistic Regression model.

##  Technical Stack
* **Core Engine:** Python 3.10+
* **Data Science:** Scikit-Learn (Model & Scaler), Pandas (EDA), NumPy
* **Web Framework:** Streamlit (UI/UX)
* **Deployment:** Streamlit Cloud / GitHub Actions

##  Data Engineering & Features
The model was trained on the [Kaggle Stroke Dataset]. Key technical implementations include:
* **Feature Engineering:** Created `pulse_pressure` (SysBP - DiaBP) and `age_glucose_impact` to capture non-linear risks.
* **Preprocessing:** Robust handling of class imbalance and `StandardScaler` normalization.
* **Pipeline:** Integrated a serialized `joblib` pipeline for seamless deployment.

##  Local Setup
```bash
# Clone the repository
git clone [https://github.com/YourUsername/Heart-disease-predictor.git](https://github.com/YourUsername/Heart-disease-predictor.git)

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app/app.py











![alt text]({37630E2B-5050-48C1-8287-A30238E8BB4F}.png)
![alt text](image.png)
![alt text](image-2.png)


