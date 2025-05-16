# Sales Prediction Web Application

![Flask](https://img.shields.io/badge/Flask-2.0.1-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM-orange)

A machine learning-powered web application for predicting sales outcomes based on business parameters.

## Features

- 🌍 Region-based data filtering (South, West, East, Central)
- 🏙️ Dynamic city-state dropdown menus
- 📈 Sales prediction using trained LightGBM model
- 📊 Input validation and error handling
- 📚 Detailed documentation page
- 🔄 Async API endpoints for location data

## Requirements

- Python 3.8+
- Flask
- pandas
- scikit-learn
- LightGBM
- python-dotenv

## Installation

1. Clone repository:
```bash
  git clone https://github.com/yourusername/sales-prediction-app.git
  cd sales-prediction-app
```

## Create virtual environment:
```bash
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  venv\Scripts\activate    # Windows
```

## Install dependencies:

```bash
  pip install -r requirements.txt
```
---

## File Structure

    sales-prediction/    
    ├── app.py
    ├── best_model.pkl
    ├── features_columns.pkl
    ├── requirements.txt
    ├── .env
    ├── static/
    │   └── styles.css
    └── templates/
        ├── index.html
        └── documentation.html


## Configuration
- Create **`.env`** file:

```bash
  FLASK_HOST=0.0.0.0
  FLASK_PORT=5000
  FLASK_DEBUG=True
```

## Usage
  Start the application:

```bash
  python app.py
```
## Access in browser:
```bash
  http://localhost:5000
```









