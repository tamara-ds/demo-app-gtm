import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random

def generate_dummy_data():

    np.random.seed(42)

    # Number of records (simulate time series by repeating employees)
    n_employees = 1000
    records_per_employee = np.random.randint(2, 6, size=n_employees)  # 2 to 5 records per employee
    total_records = sum(records_per_employee)

    # Generate base employee IDs
    employee_ids = [str(uuid.uuid4()) for _ in range(n_employees)]
    employee_id_series = np.repeat(employee_ids, records_per_employee)

    # Categorical options
    ranks = ['Cabin Crew', 'Cabin Manager', 'Flight Manager']
    locations = ['Jeddah', 'Riyadh', 'Dammam', 'Abu Dhabi']
    nationalities = ['PHILIPPINES', 'INDIA', 'UAE', 'UK', 'USA', 'CANADA']
    managers = [f"Manager {i}" for i in range(1, 21)]
    departments = ['Inflight', 'Training', 'Ground Services', 'Operations']
    grades = ['G1', 'G2', 'G3']
    contracts = ['Full-time', 'Part-time', 'Contract']
    flight_types = ['Long-haul', 'Short-haul', 'Mixed']
    performance_flags = ['Low Performer', 'Meets Expectations', 'High Performer']

    # Time range
    start_date = pd.to_datetime("2023-04-01")
    end_date = pd.to_datetime("2025-01-01")

    # Generate data
    df = pd.DataFrame({
        'employee_id': employee_id_series,
        'rank': np.random.choice(ranks, total_records, p=[0.7, 0.25, 0.05]),
        'location': np.random.choice(locations, total_records),
        'nationality': np.random.choice(nationalities, total_records),
        'manager': np.random.choice(managers, total_records),
        'department': np.random.choice(departments, total_records),
        'grade_level': np.random.choice(grades, total_records),
        'contract_type': np.random.choice(contracts, total_records),
        'flight_type': np.random.choice(flight_types, total_records),
        'performance_flag': np.random.choice(performance_flags, total_records, p=[0.1, 0.6, 0.3]),
        'tenure_years': np.round(np.random.exponential(5, total_records), 1).clip(0, 30),
        'engage_sentiment_score': np.round(np.random.uniform(1, 5, total_records), 2),
        'absence_days_past_6_months': np.random.poisson(2, total_records),
    })

    # Map sentiment score to category
    df['engage_sentiment'] = pd.cut(
        df['engage_sentiment_score'],
        bins=[0, 2.5, 3.5, 5],
        labels=['unfavorable', 'neutral', 'favorable']
    )

    # Generate dates
    df['nps_valid_from'] = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(total_records)]
    df['sentiment_valid_from'] = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(total_records)]

    # Generate CrewNPS
    df['crewnps'] = (
        df['engage_sentiment_score'] * 14
        - df['absence_days_past_6_months'] * 2
        + df['tenure_years'] * 0.6
        + df['performance_flag'].map({
            'Low Performer': -5,
            'Meets Expectations': 0,
            'High Performer': 5
        }).values
        + np.where(df['rank'] == 'Flight Manager', 10, 0)
        + np.random.normal(0, 6, total_records)
    ).clip(30, 92.75).round(2)

    return df

