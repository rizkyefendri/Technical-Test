import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, time
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open('D:/Technical Test/models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title("Waiting Time Estimation")

# User input for arrival times
st.write("Select Arrival Date and Time:")
arrival_date = st.date_input("Arrival Date")
arrival_time = st.time_input("Arrival Time", time(0, 0))

# User input for additional features
queue_length = st.number_input("Queue Length", min_value=0, value=1)

# Combine the selected date and time into a datetime format
selected_datetime = datetime.combine(arrival_date, arrival_time)

# Convert arrival_time to datetime and extract features
arrival_hour = selected_datetime.hour
arrival_minute = selected_datetime.minute

# Generate ticket number (for simplicity, using a random number here, you may adjust as needed)
ticket_number = st.number_input("Ticket Number", min_value=1, value=1)

# Predict button
if st.button("Predict Waiting Time"):
    # Create a DataFrame for the selected arrival time and other features
    df = pd.DataFrame({
        'arrival_minute': [arrival_minute],
        'arrival_hour' : [arrival_hour],
        'queue_length': [queue_length],
        'ticket_number': [ticket_number]
    })
    
    
    # Define features for prediction
    features = df[['arrival_minute','arrival_hour' , 'queue_length', 'ticket_number']]
    
    # Make predictions
    prediction = model.predict(features)[0]

    # Display the predicted wait time
    st.write(f"Predicted Waiting Time: {prediction:.2f} minutes")
    
