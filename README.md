# 🌧️ Weather Prediction Using LSTM

A web-based rainfall prediction application built with **Streamlit** and **Long Short-Term Memory (LSTM)**. This project predicts whether it will rain **1 hour ahead** based on historical weather sensor data.

## ✨ Features

- Upload weather sensor data in CSV format
- Automatic data preprocessing
- Rainfall prediction using an LSTM model
- Prediction probability output
- Export prediction results to CSV
- Interactive Streamlit interface

## 🛠️ Technologies

- Python
- Streamlit
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Joblib

## 📁 Project Structure

```text
weather-prediction-lstm/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
├── data/
├── model/
├── notebook/
└── results/
```

## 🚀 Installation

Clone this repository:

```bash
git clone https://github.com/USERNAME/weather-prediction-lstm.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📊 Dataset

The application uses historical weather sensor data including:

- Rainfall
- Temperature
- Humidity
- Wind Speed

## 🧠 Machine Learning Model

Model:

- Long Short-Term Memory (LSTM)

Prediction Horizon:

- 1 Hour Ahead

Input Sequence:

- 12 Time Steps (5-minute intervals)

## 📈 Output

The application provides:

- Rain / No Rain prediction
- Prediction probability
- Prediction timestamp
- CSV export

## 📸 Application Preview

Screenshots will be added after deployment.

## 👤 Author

**Aditya Jordan Alfaqih**

Fresh Graduate – Computer Systems

GitHub: https://github.com/ditjorr