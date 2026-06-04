from flask import Flask
import serial
import joblib
import numpy as np
import threading

app = Flask(__name__)

print("AI server started")

# load trained model
model = joblib.load("model/gas_water_model.pkl")

# serial connection (ESP8266)
ser = serial.Serial("COM3", 115200, timeout=1)

latest_prediction = "normal"
latest_confidence = 0


def read_serial():

    global latest_prediction, latest_confidence

    while True:

        try:

            line = ser.readline().decode(errors="ignore").strip()

            if "," in line:

                # split sensor data
                gas, water = map(int, line.split(","))

                # ML input
                features = np.array([[gas, water]])

                # prediction
                pred = model.predict(features)[0]

                # probability
                prob = model.predict_proba(features)[0]

                conf = int(max(prob) * 100)

                latest_prediction = pred
                latest_confidence = conf

                print(
                    "Gas:", gas,
                    "Water:", water,
                    "Prediction:", pred,
                    "Confidence:", conf, "%"
                )

        except Exception as e:
            print("Error:", e)


# run serial reader in background
thread = threading.Thread(target=read_serial)
thread.daemon = True
thread.start()


@app.route("/predict")
def predict():
    return f"{latest_prediction},{latest_confidence}"


# start server
app.run(host="0.0.0.0", port=5000)
