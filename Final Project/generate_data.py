# File: data/generate_synthetic_dataset.py

import random
import pandas as pd
import numpy as np

def generate_data(n: int) -> pd.DataFrame:
    locations = ['Downtown', 'Suburbs', 'Mall', 'Airport', 'University']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weathers = ['Sunny', 'Rainy', 'Cloudy', 'Stormy', 'Snowy']
    events = ['None', 'Concert', 'SportsEvent', 'Parade', 'Festival']
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    traffic_levels = ['Low', 'Medium', 'High']

    data = {
        'Location': [random.choice(locations) for _ in range(n)],
        'DayOfWeek': [random.choice(days) for _ in range(n)],
        'PastProfit': [random.randint(1500, 12000) for _ in range(n)],
        'Weather': [random.choice(weathers) for _ in range(n)],
        'SpecialEvent': [random.choice(events) for _ in range(n)],
        'CompetitorNearby': [random.choice(['Yes', 'No']) for _ in range(n)],
        'Holiday': [random.choice(['Yes', 'No']) for _ in range(n)],
        'Season': [random.choice(seasons) for _ in range(n)],
        'PopulationDensity': [random.randint(100, 20000) for _ in range(n)],
        'AverageIncome': [random.randint(25000, 150000) for _ in range(n)],
        'MarketingSpend': [random.randint(500, 10000) for _ in range(n)],
        'CustomerFootfall': [],
        'OnlineOrders': [],
        'StoreAge': [random.randint(1, 30) for _ in range(n)],
        'EmployeeCount': [],
        'NearbyParkingAvailability': [random.choice(['Yes', 'No']) for _ in range(n)],
        'RoadTraffic': [random.choice(traffic_levels) for _ in range(n)],
        'CompetitorDiscounts': [random.randint(0, 50) for _ in range(n)],
    }

    # Derive dependent columns
    for i in range(n):
        base_footfall = 50 if data['Location'][i] == 'Suburbs' else 200
        event_boost = 300 if data['SpecialEvent'][i] != 'None' else 0
        holiday_boost = 150 if data['Holiday'][i] == 'Yes' else 0
        season_factor = 1.2 if data['Season'][i] == 'Summer' else 1
        traffic_penalty = -50 if data['RoadTraffic'][i] == 'High' else 0

        footfall = base_footfall + event_boost + holiday_boost + random.randint(-30, 30) + traffic_penalty
        footfall = max(footfall * season_factor, 0)
        data['CustomerFootfall'].append(int(footfall))

        online_orders = random.randint(5, 50)
        if data['Location'][i] == 'Downtown':
            online_orders += 20
        elif data['Location'][i] == 'Mall':
            online_orders += 10
        data['OnlineOrders'].append(online_orders)

        employee_count = int(footfall / 20) + random.randint(1, 5)
        data['EmployeeCount'].append(employee_count)

    return pd.DataFrame(data)

# Generate dataset
df = generate_data(450)

# Write dataset to CSV
df.to_csv('generated_profit_data.csv', index=False)