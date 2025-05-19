# File: data_loader.py
import pandas as pd
import os

def load_result_data(path):
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df.set_index('Datetime')

def load_user_fitbit(user_file):
    if not os.path.exists(user_file):
        raise FileNotFoundError(f"Fitbit file not found: {user_file}")

    df = pd.read_csv(user_file)
    df['Start Time'] = pd.to_datetime(df['sleeping_time2'])
    df['End Time'] = pd.to_datetime(df['wake_up_time2'])
    df['date'] = df['End Time'].dt.date
    df['start_bin'] = ((df['Start Time'].dt.hour * 4 + df['Start Time'].dt.minute / 15) + 24) % 96
    df['end_bin'] = ((df['End Time'].dt.hour * 4 + df['End Time'].dt.minute / 15) + 24) % 96
    return df