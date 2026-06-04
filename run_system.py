import joblib
import serial
import pandas as pd

model = joblib.load("model/gas_water_model.pkl")

ser = serial.Serial("COM3",115200)

print("AI Safety System Running")

while True:

    line = ser.readline().decode(errors="ignore").strip()

    if "," in line:

        gas,water = map(int,line.split(","))

        data = pd.DataFrame([[gas,water]], columns=["gas","water"])

        prediction = model.predict(data)

        print("Gas:",gas,"Water:",water,"Prediction:",prediction)
