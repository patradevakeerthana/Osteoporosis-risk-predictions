## **Osteoporosis Risk Prediction Web App**



This is a remote monitoring application for osteoporosis risk analysis. The project consists of a Python Flask backend API that uses a pre-trained machine learning model and a single-file HTML frontend for a user-friendly interface.



Features

Public-Facing Risk Checker: A web form where users can input personal and lifestyle information to receive a hypothetical osteoporosis risk prediction.



Clinician Dashboard: A secure, password-protected section for researchers and clinicians to view model performance metrics and a real-time audit log of user predictions.



Dynamic Visualizations: The dashboard includes dynamically generated charts for model evaluation (ROC curve) and feature importance, driven by the backend API.



Robust Backend: The Flask API correctly handles incoming data, preprocesses it to match the trained model's requirements, and returns accurate predictions.



CORS-Enabled: The backend is configured to allow cross-origin requests, so the frontend can be run locally or hosted separately.



Getting Started

To get this application running, you need to set up both the backend API and the frontend web page.



1\. Backend Setup

The backend is built with Python and Flask.



Prerequisites

Python 3.7 or newer



pip (Python package installer)



Instructions

Navigate to the project directory in your terminal or command prompt.



cd C:\\Users\\Lidiy\\Downloads\\Risk\_prediction



Place the necessary files in this directory:



api.py (The backend code)



osteoporosis\_screening\_model.pkl (Your trained machine learning model)



eval\_curves.json (The model evaluation data for the ROC curve)



Install the required libraries. You can do this by running pip install for each library.



pip install Flask Flask-Cors scikit-learn xgboost pandas joblib



Run the backend server.



python api.py



You will see output indicating that the server is running on http://127.0.0.1:5000. Keep this terminal window open while you are using the web app.



2\. Frontend Setup

The frontend is a single two\_page\_app.html file that runs directly in your web browser.



Save the file. Place the two\_page\_app.html file in the same directory as your backend files.



Open in a browser. Simply double-click the two\_page\_app.html file, and it will open in your default web browser.



3\. Using the Application

For the Public: Use the form on the main page to enter your data and receive a risk prediction. The results are powered by the backend API.



For Clinicians: Click the "Clinician/Researcher" tab. Log in with the following demo credentials to access the dashboard and view model performance.



Username: admin



Password: password



Project Structure

/Risk\_prediction

├── api.py                  # The Python backend API

├── osteoporosis\_screening\_model.pkl  # The pre-trained ML model

├── eval\_curves.json        # Evaluation data for the ROC curve

└── two\_page\_app.html       # The frontend web page



