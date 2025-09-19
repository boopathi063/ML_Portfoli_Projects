# Bank Marketing Prediction
**AI-Powered Customer Subscription Analysis**

[![Python](https://img.shields.io/badge/python-3.8-blue?logo=python)](https://www.python.org/)  
[![Flask](https://img.shields.io/badge/flask-2.3.2-orange?logo=flask)](https://flask.palletsprojects.com/)  
[![Docker](https://img.shields.io/badge/docker-20.10-blue?logo=docker)](https://www.docker.com/)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![AWS CI/CD](https://github.com/boopathi063/aws-ci-cd-bank-prediction/actions/workflows/aws.yml/badge.svg)](https://github.com/boopathi063/aws-ci-cd-bank-prediction/actions/workflows/aws.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/boopathi064/bankprediction-app)](https://hub.docker.com/r/boopathi064/bankprediction-app)  
[![GitHub Stars](https://img.shields.io/github/stars/boopathi063/aws-ci-cd-bank-prediction?style=social)](https://github.com/boopathi063/aws-ci-cd-bank-prediction/stargazers)  

---

## 🚀 Project Overview
This project is an **AI-powered web application** that predicts whether a bank customer will subscribe to a term deposit based on their demographics, past interactions, and economic indicators. The model leverages advanced **machine learning algorithms** like **XGBoost** for high accuracy, and the application is built using **Flask** for web deployment and **Docker** for containerization and cloud readiness.

---

## 📊 Features
- Predict customer subscription probability for a term deposit.
- Provides risk levels: **High** or **Low**.
- Generates actionable recommendations for marketing campaigns.
- Interactive **web interface** for real-time predictions.
- RESTful **API endpoint** for integration with other applications.
- Fully **Dockerized** for cloud deployment (AWS, GCP, Azure).
- Supports **feature engineering** and data validation for robust predictions.

---

## 🛠️ Tech Stack
- **Python 3.11**
- **Flask** – Web application framework
- **XGBoost & LightGBM** – Machine learning models
- **Pandas & NumPy** – Data manipulation
- **Seaborn & Matplotlib** – Visualization
- **Docker** – Containerization for deployment
- **AWS / Cloud Hosting** – Scalable deployment
- **Optuna** – Hyperparameter optimization

---

## 🤖 Models Evaluated  
The project benchmarks several machine learning models before selecting the best:  

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- XGBoost  
- LightGBM  

**Final Choice:**  
- **XGBoost** was selected as the production model, achieving:  

| Metric   | Score |
|----------|-------|
| Accuracy | 0.87  |
| Recall   | 0.93  |
| F1 Score | 0.62  |
| ROC-AUC  | 0.96  |

---


## 📂 Project Structure
├── app.py # Flask application
├── Dockerfile # Docker image setup
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── models/ # Saved trained models
│ ├── best_model.pkl
│ └── xgb_pipeline.pkl
├── data/ # Dataset files
│ ├── bank.csv
│ ├── processed_train.csv
│ └── processed_test.csv
├── src/
│ ├── init.py
│ └── utils.py # Prediction and feature engineering utilities
├── templates/ # HTML templates
│ ├── index.html
│
├── notebook/ # Jupyter notebooks for EDA & modeling
└── .github/workflows/ # CI/CD workflows 

## Run Locally with Docker

Pull the Docker image:

--bash
docker pull boopathi064/bankprediction-app:latest

## Project Screenshots

### AWS EC2 Instance
![EC2 Instance](results/aws_ec2_instance_summary.jpg)

### Prediction
![Prediction](results/prediction_positive.jpg)

### CI/CD & Project Flow
![CI/CD Flow](results/ci_cd_pipeline_flow.jpg)

