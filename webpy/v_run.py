import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
petrol_data = pd.read_csv('webpy/petrolcars.csv')
ev_data = pd.read_csv('webpy/Cheapestelectriccars-EVDatabase.csv')

# Fill missing values with mean for each dataset
petrol_data.fillna(petrol_data.select_dtypes(include='number').mean(), inplace=True)
ev_data.fillna(ev_data.select_dtypes(include='number').mean(), inplace=True)

# Separate features and target variable for each dataset
X_petrol = petrol_data[['Acceleration', 'TopSpeed', 'Range', 'Efficiency', 'NumberofSeats']]
y_petrol = petrol_data['Price']

X_ev = ev_data[['Acceleration', 'TopSpeed', 'Range', 'Efficiency', 'NumberofSeats']]
y_ev = ev_data['Price']

# Standardize the features for each dataset
scaler_petrol = StandardScaler()
X_petrol_scaled = scaler_petrol.fit_transform(X_petrol)

scaler_ev = StandardScaler()
X_ev_scaled = scaler_ev.fit_transform(X_ev)

# Train the RandomForestRegressor model for each dataset
model_petrol = RandomForestRegressor()
model_petrol.fit(X_petrol_scaled, y_petrol)

model_ev = RandomForestRegressor()
model_ev.fit(X_ev_scaled, y_ev)

# Streamlit app
st.title('Electric Vehicle Recommender')

# User input in the sidebar
st.sidebar.header('User Input')
user_acceleration = st.sidebar.number_input('Enter Acceleration:', min_value=0.0, step=0.1, value=8.0)
user_topspeed = st.sidebar.number_input('Enter Top Speed:', min_value=0, value=180)
user_range = st.sidebar.number_input('Enter Range:', min_value=0, value=300)
user_efficiency = st.sidebar.number_input('Enter Efficiency:', min_value=0.0, step=0.1, value=20.0)
user_seats = st.sidebar.number_input('Enter Number of Seats:', min_value=0, value=5)

user_petrol_input = [user_acceleration, user_topspeed, user_range, user_efficiency, user_seats]

# Scale the user input for each dataset
user_input_petrol_scaled = scaler_petrol.transform([user_petrol_input])
user_input_ev_scaled = scaler_ev.transform([user_petrol_input])

# Predict prices for both datasets
predicted_petrol_price = model_petrol.predict(user_input_petrol_scaled)
predicted_ev_price = model_ev.predict(user_input_ev_scaled)

# Find the top 10 similar EVs based on the EV dataset
similar_ev_indices = (model_ev.predict(X_ev_scaled) - predicted_ev_price).argsort()[:10]
similar_ev_models = ev_data.loc[similar_ev_indices, 'CarModel']

# Display the recommended EVs
st.header("Top 10 Recommended EV Cars as alternatives to petrol/Diesel cars:")
for ev_model in similar_ev_models:
    st.write(ev_model)

# Display feature comparison plots with dynamically changing colors
features_to_compare = ['Acceleration', 'TopSpeed', 'Range', 'Efficiency', 'NumberofSeats', 'Price']
colors = sns.color_palette('husl', n_colors=len(similar_ev_models))

for i, feature in enumerate(features_to_compare):
    plt.figure(figsize=(12, 6))
    plt.bar(similar_ev_models, ev_data.loc[similar_ev_indices, feature], color=colors[i])
    plt.title(f'Comparison of {feature} for Top 10 Recommended EV Cars')
    plt.xlabel('Car Model')
    plt.ylabel(feature)
    st.pyplot(plt)
