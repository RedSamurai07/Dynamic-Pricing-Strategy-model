# Dynamic Pricing Strategy model

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
This project focuses on the development and implementation of a Dynamic Pricing Model for a ride-sharing service. Traditional ride-sharing costs are often static or based solely on distance and time. This project leverages data science to transition into a "surge pricing" model where prices fluctuate in real-time based on the immediate balance between supply (available drivers) and demand (active riders). By analyzing a dataset of 1,000 rides, the project identifies how to adjust prices to maximize revenue and maintain service availability during peak periods.

### Executive Summary
The analysis of the provided dataset reveals a significant imbalance in the ride-sharing ecosystem, with an average of 60 riders competing for only 27 drivers at any given time. Currently, prices are calculated based on historical costs (averaging $372.50) which do not account for this scarcity.

The executive findings suggest that by applying Demand and Supply Multipliers—calculated using the distribution of riders and drivers (25th and 75th percentiles)—the company can implement a dynamic pricing engine. This model identifies "High Demand" and "Low Supply" states to adjust the final cost. Early results from this strategy indicate that approximately 82.7% of rides under this dynamic model result in a profitable "Win" for the platform, ensuring that the service remains viable even when driver availability is low.

### Goal
The primary objectives of this project are:

1. Balance Market Supply and Demand: Use price as a lever to moderate demand during peak times and incentivize more drivers to enter the market when supply is low.

2. Maximize Revenue Generation: Identify "High Demand" scenarios where customers are willing to pay a premium (Surge Pricing) to increase the total historical cost per ride.

3. Data-Driven Price Optimization: Move away from fixed-rate pricing to a model that uses statistical percentiles (from the describe() table) to set thresholds for "High" and "Low" activity.

4. Improve Operational Efficiency: Categorize ride outcomes into "Profitable" and "Loss" segments to ensure the pricing algorithm consistently covers the cost of service while remaining competitive.

5. Maintain Customer Satisfaction: Ensure that even with price fluctuations, the average rating (currently 4.25) remains high by ensuring ride availability during critical times.
   
### Data structure and initial checks
[Dataset](kindly look into the repository which is attached)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| Number_of_Riders | The count of passengers requesting a ride at a specific time. | int64 |
| Number_of_Drivers | The count of available drivers in the vicinity at the same time. | int64 |
| Location_Category | The type of area where the ride is requested (e.g., Urban, Suburban, Rural).| object |
| Customer_Loyalty_Status | The membership or loyalty tier of the customer (e.g., Silver, Regular). | object |
| Number_of_Past_Rides | The total number of rides previously completed by the customer. | int64 |  
| Average_Ratings | The historical performance rating of the driver or service. | float64 |
| Time_of_Booking | The time of day when the request was made (e.g., Night, Evening, Afternoon). | object |
| Vehicle_Type  | The category of vehicle requested, often mapped to binary values like Premium (1) or Economy (0) for modeling.| object |
| Expected_Ride_Duration | The estimated time in minutes for the trip to be completed. | int64  |
| Historical_Cost_of_Ride | The original baseline cost of the ride, typically calculated based solely on duration before dynamic adjustments. | float64 |

### Tools
- Python: Google Colab - Data Preparation, Exploratory Data Analysis, Descriptive Statistics, Data manipulation, Visualization, Feature Engineering, Model developoment
  
### Analysis
Python
Importing all the libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```
```python
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

```
Loading the dataset
``` python
df = pd.read_csv("dynamic_pricing.csv")
df.head()
```
<img width="1545" height="188" alt="image" src="https://github.com/user-attachments/assets/d43058c2-100d-430f-ba5d-478afc1cf881" />

information about the dataset
``` python
df.info()
```
<img width="498" height="423" alt="image" src="https://github.com/user-attachments/assets/69baeded-08e0-40a2-915c-69c8e9088006" />

**Exploratpory data analysis**
Descriptive statistics
``` python
df.describe()
```
<img width="1015" height="267" alt="image" src="https://github.com/user-attachments/assets/db89bf7a-a992-48c4-bff9-087e7fd68383" />

**Insights**
- On average, there are about 60 riders for only 27 drivers. This indicates that demand typically doubles the available supply.
- At the maximum recorded levels, there are 100 riders competing for 89 drivers.
- In the worst-case scenario (minimum values), only 5 drivers are available to serve 20 riders.
- This imbalance is the primary driver for implementing a dynamic pricing strategy, which uses "multipliers" to increase prices during these high-demand/low-supply periods to incentivize more drivers to join the platform.
- The average ride lasts approximately 100 minutes, with the longest trips reaching 180 minutes (3 hours).
- The average cost for these rides is $372.50. The high standard deviation ($187.16) shows significant price variability, likely due to the wide range in ride durations (from 10 to 180 minutes).
- The "historical cost" in this dataset was originally calculated using only expected ride duration as the determining factor.
- On average, riders have a history of 50 past rides, suggesting a loyal customer base.
- The mean rating is 4.25 out of 5.0. Even at the 25th percentile, ratings remain relatively high at 3.87, indicating generally consistent service quality.
- The project calculates a Demand Multiplier and a Supply Multiplier based on the percentiles shown in your table (specifically the 25th and 75th percentiles).
- By adjusting the historical cost using these multipliers, the project aims to balance the market. In simulations using this data, approximately 82.7% of rides under the new dynamic pricing model were found to be profitable.

Expected Ride duration versus Historical cost of ride
``` python
fig = px.scatter(df, x='Expected_Ride_Duration', 
                 y='Historical_Cost_of_Ride',
                 title='Expected Ride Duration vs. Historical Cost of Ride', 
                 trendline='ols')
fig.show()
```
<img width="740" height="412" alt="image" src="https://github.com/user-attachments/assets/1ebd88f9-b2de-4b75-90a8-3454b8224415" />

Distribution of historial cost of ride versus vehicle type
``` python
fig = px.box(df, x='Vehicle_Type', 
             y='Historical_Cost_of_Ride',
             title='Historical Cost of Ride Distribution by Vehicle Type')
fig.show()
```
<img width="760" height="425" alt="image" src="https://github.com/user-attachments/assets/e59f6205-7bc8-4915-9867-18c8fe28e923" />

Correlation Analysis
``` python
numerical_df = df.select_dtypes(include=[np.number])
corr_matrix = numerical_df.corr()

fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, 
                                x=corr_matrix.columns, 
                                y=corr_matrix.columns,
                                colorscale='plasma'))
fig.update_layout(title='Correlation Matrix')
fig.show()
```
<img width="816" height="453" alt="image" src="https://github.com/user-attachments/assets/0e5b1762-8142-4f8f-9ed6-1af331d20ab8" />

**Implementing a Dynamic Pricing Strategy**

The data provided by the company states that the company uses a pricing model that only takes the expected ride duration as a factor to determine the price for a ride. Now, we will implement a dynamic pricing strategy aiming to adjust the ride costs dynamically based on the demand and supply levels observed in the data. It will capture high-demand periods and low-supply scenarios to increase prices, while low-demand periods and high-supply situations will lead to price reductions.

``` python
high_demand_percentile = 75
low_demand_percentile = 25

df['demand_multiplier'] = np.where(df['Number_of_Riders'] > np.percentile(df['Number_of_Riders'], high_demand_percentile),
                                     df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], high_demand_percentile),
                                     df['Number_of_Riders'] / np.percentile(df['Number_of_Riders'], low_demand_percentile))

# Calculate supply_multiplier based on percentile for high and low supply
high_supply_percentile = 75
low_supply_percentile = 25

df['supply_multiplier'] = np.where(df['Number_of_Drivers'] > np.percentile(df['Number_of_Drivers'], low_supply_percentile),
                                     np.percentile(df['Number_of_Drivers'], high_supply_percentile) / df['Number_of_Drivers'],
                                     np.percentile(df['Number_of_Drivers'], low_supply_percentile) / df['Number_of_Drivers'])

# Define price adjustment factors for high and low demand/supply
demand_threshold_high = 1.2  # Higher demand threshold
demand_threshold_low = 0.8  # Lower demand threshold
supply_threshold_high = 0.8  # Higher supply threshold
supply_threshold_low = 1.2  # Lower supply threshold

# Calculate adjusted_ride_cost for dynamic pricing
df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (
    np.maximum(df['demand_multiplier'], demand_threshold_low) *
    np.maximum(df['supply_multiplier'], supply_threshold_high)
)
```
In the above code, we first calculated the demand multiplier by comparing the number of riders to percentiles representing high and low demand levels. If the number of riders exceeds the percentile for high demand, the demand multiplier is set as the number of riders divided by the high-demand percentile. Otherwise, if the number of riders falls below the percentile for low demand, the demand multiplier is set as the number of riders divided by the low-demand percentile.

Next, we calculated the supply multiplier by comparing the number of drivers to percentiles representing high and low supply levels. If the number of drivers exceeds the low-supply percentile, the supply multiplier is set as the high-supply percentile divided by the number of drivers. On the other hand, if the number of drivers is below the low-supply percentile, the supply multiplier is set as the low-supply percentile divided by the number of drivers.

Finally, we calculated the adjusted ride cost for dynamic pricing. It multiplies the historical cost of the ride by the maximum of the demand multiplier and a lower threshold (demand_threshold_low), and also by the maximum of the supply multiplier and an upper threshold (supply_threshold_high). This multiplication ensures that the adjusted ride cost captures the combined effect of demand and supply multipliers, with the thresholds serving as caps or floors to control the price adjustments.

Now let’s calculate the profit percentage we got after implementing this dynamic pricing strategy

``` python
# Calculate the profit percentage for each ride
df['profit_percentage'] = ((df['adjusted_ride_cost'] - df['Historical_Cost_of_Ride']) / df['Historical_Cost_of_Ride']) * 100
# Identify profitable rides where profit percentage is positive
profitable_rides = df[df['profit_percentage'] > 0]

# Identify loss rides where profit percentage is negative
loss_rides = df[df['profit_percentage'] < 0]

# Calculate the count of profitable and loss rides
profitable_count = len(profitable_rides)
loss_count = len(loss_rides)

# Create a donut chart to show the distribution of profitable and loss rides
labels = ['Profitable Rides', 'Loss Rides']
values = [profitable_count, loss_count]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
fig.update_layout(title='Profitability of Rides (Dynamic Pricing vs. Historical Pricing)')
fig.show()
```
<img width="800" height="409" alt="image" src="https://github.com/user-attachments/assets/4d2592ac-d828-4162-bc49-f939e6f07a91" />

Now let’s have a look at the relationship between the expected ride duration and the cost of the ride based on the dynamic pricing strategy.

``` python
fig = px.scatter(df, 
                 x='Expected_Ride_Duration', 
                 y='adjusted_ride_cost',
                 title='Expected Ride Duration vs. Cost of Ride', 
                 trendline='ols')
fig.show()
```
<img width="751" height="418" alt="image" src="https://github.com/user-attachments/assets/f960d330-5e78-4a2b-baf9-2a3eab2b15e2" />

**Training a Predictive Model**

Now, as we have implemented a dynamic pricing strategy, let’s train a Machine Learning model. Before training the model, let’s preprocess the data

```python
def data_preprocessing_pipeline(data):
    #Identify numeric and categorical features
    numeric_features = data.select_dtypes(include=['float', 'int']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    #Handle missing values in numeric features
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

    #Detect and handle outliers in numeric features using IQR
    for feature in numeric_features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        data[feature] = np.where((data[feature] < lower_bound) | (data[feature] > upper_bound),
                                 data[feature].mean(), data[feature])

    #Handle missing values in categorical features
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

    return data
```
let’s convert it into a numerical feature before moving forward.
``` python
df["Vehicle_Type"] = df["Vehicle_Type"].map({"Premium": 1, 
                                           "Economy": 0})
```
Now let’s split the data and train a Machine Learning model to predict the cost of a ride
``` python
#splitting data
x = np.array(df[["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type", "Expected_Ride_Duration"]])
y = np.array(df[["adjusted_ride_cost"]])

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Reshape y to 1D array
y_train = y_train.ravel()
y_test = y_test.ravel()

# Training a random forest regression model
model = RandomForestRegressor()
model.fit(x_train, y_train)
```
Now let’s test this Machine Learning model using some input values.
``` python
def get_vehicle_type_numeric(vehicle_type):
    vehicle_type_mapping = {
        "Premium": 1,
        "Economy": 0
    }
    vehicle_type_numeric = vehicle_type_mapping.get(vehicle_type)
    return vehicle_type_numeric
  
# Predicting using user input values
def predict_price(number_of_riders, number_of_drivers, vehicle_type, Expected_Ride_Duration):
    vehicle_type_numeric = get_vehicle_type_numeric(vehicle_type)
    if vehicle_type_numeric is None:
        raise ValueError("Invalid vehicle type")
    
    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, Expected_Ride_Duration]])
    predicted_price = model.predict(input_data)
    return predicted_price

# Example prediction using user input values
user_number_of_riders = 50
user_number_of_drivers = 25
user_vehicle_type = "Economy"
Expected_Ride_Duration = 30
predicted_price = predict_price(user_number_of_riders, user_number_of_drivers, user_vehicle_type, Expected_Ride_Duration)
print("Predicted price:", predicted_price)
```
<img width="257" height="32" alt="image" src="https://github.com/user-attachments/assets/3a1f2856-52fe-4b20-9615-cdade9df8cc4" />

Here’s a comparison of the actual and predicted results

```python
# Predict on the test set
y_pred = model.predict(x_test)

# Create a scatter plot with actual vs predicted values
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_test.flatten(),
    y=y_pred,
    mode='markers',
    name='Actual vs Predicted'
))

# Add a line representing the ideal case
fig.add_trace(go.Scatter(
    x=[min(y_test.flatten()), max(y_test.flatten())],
    y=[min(y_test.flatten()), max(y_test.flatten())],
    mode='lines',
    name='Ideal',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='Actual vs Predicted Values',
    xaxis_title='Actual Values',
    yaxis_title='Predicted Values',
    showlegend=True,
)

fig.show()
```
<img width="802" height="423" alt="image" src="https://github.com/user-attachments/assets/e5487d92-3a76-4ff6-b802-e33946e346c8" />

This is how you can use Machine Learning to implement a data-driven dynamic pricing strategy. In a dynamic pricing strategy, the aim is to maximize revenue and profitability by pricing items at the right level that balances supply and demand dynamics. It allows businesses to adjust prices dynamically based on factors like time of day, day of the week, customer segments, inventory levels, seasonal fluctuations, competitor pricing, and market conditions.

### Insights

- Significant Supply-Demand Imbalance: On average, there are approximately 60 riders for only 27 drivers. This indicates that demand typically doubles the available supply, which is the primary driver for a dynamic pricing strategy.

- Market Volatility: In extreme cases, the supply-demand gap is even more severe, with as few as 5 drivers available to serve 20 riders.

- Pricing Efficiency: The dynamic pricing model, which uses demand and supply multipliers to adjust historical costs, resulted in approximately 82.7% of rides being profitable in simulations.

- Strong Customer Retention and Quality: The dataset shows a loyal customer base with an average of 50 past rides per user and a high mean rating of 4.25 out of 5.0.

- Predictive Performance: The model demonstrates a strong linear relationship between expected ride duration and the adjusted cost, as shown in the visualization "Expected Ride Duration vs. Cost of Ride".

### Recommendations

- Tiered Incentive Systems: Beyond just increasing prices, use the supply and demand multipliers to trigger real-time bonuses for drivers in low-supply areas to rebalance the market.

- Loyalty-Based Price Shielding: Protect high-value customers (averaging 50+ rides) from extreme surge pricing. Implementing a "Surge Cap" for loyal members can prevent churn while maintaining profitability from newer users.

- Predictive Fleet Rebalancing: Use "Time of Booking" and "Location Category" (Urban vs. Rural) to create heat maps. Move drivers toward high-demand zones before the surge peaks to stabilize prices for riders.

- Rating-Integrated Pricing: Incorporate the Average_Ratings feature into the pricing logic. Drivers with top-tier ratings (>4.8) could receive a larger share of the adjusted ride cost, incentivizing high service quality.

- Duration-Based Forecasting: Given the long-haul nature of many rides (up to 3 hours), the model should account for the "supply opportunity cost". Drivers on 180-minute trips are unavailable for multiple shorter bookings; a specific multiplier for long-duration trips during low-supply periods could compensate for this.

- Automated Data Robustness: Continue utilizing the automated pre-processing pipeline that handles missing values and outliers via IQR (Interquartile Range) to ensure the model adapts as the dataset grows.


