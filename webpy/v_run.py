import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
petrol_data = pd.read_csv('petrolcars.csv')
ev_data = pd.read_csv('Cheapestelectriccars-EVDatabase.csv')
combined_data = pd.concat([petrol_data, ev_data], ignore_index=True)
combined_data.fillna(combined_data.select_dtypes(include='number').mean(), inplace=True)
X = combined_data[['Acceleration', 'TopSpeed', 'Range', 'Efficiency', 'NumberofSeats']]
y = combined_data['Price']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
st.title('Electric Vehicle Recommender')
st.sidebar.header('User Input')
user_acceleration = st.sidebar.number_input('Enter Acceleration:', min_value=0.0, step=0.1, value=0.0)
user_topspeed = st.sidebar.number_input('Enter Top Speed:', min_value=0, value=0)
user_range = st.sidebar.number_input('Enter Range:', min_value=0, value=0)
user_efficiency = st.sidebar.number_input('Enter Efficiency:', min_value=0.0, step=0.1, value=0.0)
user_seats = st.sidebar.number_input('Enter Number of Seats:', min_value=0, value=0)
user_petrol_input = [user_acceleration, user_topspeed, user_range, user_efficiency, user_seats]
user_input_scaled = scaler.transform([user_petrol_input])
predicted_ev_prices = model.predict(user_input_scaled)
similar_ev_indices = (model.predict(X_scaled) - predicted_ev_prices).argsort()[:10]
similar_ev_models = combined_data.loc[similar_ev_indices, 'CarModel']
st.header("Top 10 Recommended EV Cars as an alternative to petrol/diesel cars:")
for ev_model in similar_ev_models:
    st.write(ev_model)
features_to_compare = ['Acceleration', 'TopSpeed', 'Range', 'Efficiency', 'NumberofSeats', 'Price']
for feature in features_to_compare:
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette('husl', n_colors=len(similar_ev_models))
    sns.barplot(x=similar_ev_models, y=combined_data.loc[similar_ev_indices, feature], palette=colors)
    plt.title(f'Comparison of {feature} for Top 10 Recommended EV Cars')
    plt.xlabel('Car Model')
    plt.ylabel(feature)
    st.pyplot(plt)
