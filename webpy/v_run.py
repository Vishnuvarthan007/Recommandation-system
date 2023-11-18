import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load datasets
petrol_data = pd.read_csv('/content/petrolcars.csv')
ev_data = pd.read_csv('/content/Cheapestelectriccars-EVDatabase.csv')

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

# Split the data into training and testing sets for each dataset
X_train_petrol, X_test_petrol, y_train_petrol, y_test_petrol = train_test_split(
    X_petrol_scaled, y_petrol, test_size=0.2, random_state=42
)

X_train_ev, X_test_ev, y_train_ev, y_test_ev = train_test_split(
    X_ev_scaled, y_ev, test_size=0.2, random_state=42
)

# Train the RandomForestRegressor model for each dataset
model_petrol = RandomForestRegressor()
model_petrol.fit(X_train_petrol, y_train_petrol)

model_ev = RandomForestRegressor()
model_ev.fit(X_train_ev, y_train_ev)

# User input
user_acceleration = float(input("Enter Acceleration: "))
user_topspeed = float(input("Enter Top Speed: "))
user_range = float(input("Enter Range: "))
user_efficiency = float(input("Enter Efficiency: "))
user_seats = float(input("Enter Number of Seats:"))

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
print("Top 10 Recommended EV Cars as alternatives to petrol/Diesel cars:")
for ev_model in similar_ev_models:
    print(ev_model)

# Plotting comparison charts (similar to your original code)
features_to_compare = ['Acceleration', 'TopSpeed', 'Range', 'Efficiency', 'NumberofSeats', 'Price']
for feature in features_to_compare:
    plt.figure(figsize=(12, 6))
    plt.bar(similar_ev_models, ev_data.loc[similar_ev_indices, feature])
    plt.title(f'Comparison of {feature} for Top 10 Recommended EV Cars')
    plt.xlabel('Car Model')
    plt.ylabel(feature)
    plt.show()
