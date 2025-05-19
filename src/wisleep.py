import os
import re
import time
import datetime
import pandas as pd
import pymc3 as pm
from datetime import timedelta

# === Config ===
BASE_DIR = "/your/base/data/path"
USER_DATA_DIR = f"{BASE_DIR}/Fitbit_data_new"
RESULT_CSV = f"{BASE_DIR}/fitbituserdata.csv"
SAMPLE_DIR = f"{BASE_DIR}/sample_dataset"
SAVE_DIR = f"{BASE_DIR}/sensys_results"

# === Load result dataframe ===
result = pd.read_csv(RESULT_CSV)
result['Datetime'] = pd.to_datetime(result['Datetime'])
result.set_index('Datetime', inplace=True)

def cohort_trajectory(track_df):
    """Infer trajectory start/end based on changes in 'CHRT'."""
    AP_list, AP_start, AP_end, Event_count = [], [], [], []
    prev, time_end, ev_count, count = '', 0, 0, 0

    for index, row in track_df.iterrows():
        if row['CHRT'].startswith("#"):
            continue
        if count > 0 and row['CHRT'] == prev:
            time_end = index
            ev_count += 1
        else:
            if count > 0:
                AP_end.append(time_end)
                Event_count.append(ev_count)
            AP_list.append(row['CHRT'])
            AP_start.append(index)
            time_end = index
            prev = row['CHRT']
            ev_count = 0
            count += 1

    AP_end.append(time_end)
    Event_count.append(ev_count)

    df = pd.DataFrame({
        'CHRT': AP_list,
        'start': pd.to_datetime(AP_start),
        'end': pd.to_datetime(AP_end),
        'count': Event_count
    })

    if df.empty:
        return 0, 0

    df['duration'] = df['end'] - df['start']
    loc = df['duration'].idxmax()
    return df.loc[loc, 'start'], df.loc[loc, 'end']

def model_comparison(user_id):
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.csv")
    if not os.path.exists(user_file):
        print(f"Missing file for user {user_id}")
        return

    print(f"Processing user {user_id}...")
    df = pd.read_csv(user_file)
    df['Start Time'] = pd.to_datetime(df['sleeping_time2'])
    df['End Time'] = pd.to_datetime(df['wake_up_time2'])
    df['date'] = df['End Time'].dt.date
    df['start_bin'] = ((df['Start Time'].dt.hour * 4 + df['Start Time'].dt.minute / 15) + 24) % 96
    df['end_bin'] = ((df['End Time'].dt.hour * 4 + df['End Time'].dt.minute / 15) + 24) % 96

    sleep_list, wakeup_list, sleep_true, wakeup_true, sleep_hier, wakeup_hier, count_list = [], [], [], [], [], [], []

    for filename in os.listdir(SAMPLE_DIR):
        if not filename.startswith(f"user{user_id}_"):
            continue

        data = pd.read_csv(os.path.join(SAMPLE_DIR, filename))
        if data['bin'].iloc[0] >= 50:
            continue

        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data['date'] = data['Datetime'].dt.date
        session_date = data['date'].iloc[-1]

        try:
            loc = df[df['date'] == session_date].index[0]
        except IndexError:
            continue

        bed = df.loc[loc, 'Start Time']
        wakeup = df.loc[loc, 'End Time']

        end_day = data['Datetime'].iloc[-1].date()
        start_dt = datetime.datetime.combine(end_day - timedelta(days=1), datetime.time(18))
        end_dt = datetime.datetime.combine(end_day, datetime.time(18))

        track_df = result.loc[start_dt.strftime("%Y-%m-%d %H:%M:%S"):end_dt.strftime("%Y-%m-%d %H:%M:%S")]
        start, end = cohort_trajectory(track_df)
        if start == 0 or end == 0:
            continue

        start_bin = int((start.hour * 4 + start.minute / 15 + 24) % 96)
        end_bin = int((end.hour * 4 + end.minute / 15 + 24) % 96)

        def run_model(name, T1_prior, T2_prior):
            with pm.Model(name=name) as model:
                T1 = T1_prior()
                T2 = T2_prior()
                rate1 = pm.Gamma("rate1", alpha=2.5, beta=1.0)
                rate2 = pm.Exponential("rate2", lam=0.0001)
                rate_sleep = pm.math.switch(T1 >= data['bin'].values, rate1, rate2)
                rate_awake = pm.math.switch(T2 >= data['bin'].values, rate2, rate1)
                pm.Poisson("sleep", rate_sleep, observed=data['connected'].values)
                pm.Poisson("awake", rate_awake, observed=data['connected'].values)
                trace = pm.sample(1000, tune=1000, cores=1, progressbar=False)
            return trace, model

        trace1, model1 = run_model("Normal",
                                   lambda: pm.Normal('T1', mu=35, sd=12),
                                   lambda: pm.Normal('T2', mu=59, sd=12))
        trace2, model2 = run_model("Uniform",
                                   lambda: pm.DiscreteUniform('T1', lower=start_bin, upper=start_bin + 12),
                                   lambda: pm.DiscreteUniform('T2', lower=start_bin + 13, upper=end_bin))
        trace3, model3 = run_model("Hierarchical",
                                   lambda: pm.Normal('T1', mu=43, sd=pm.Gamma('g', alpha=pm.Exponential('a', 1), beta=pm.Exponential('b', 1))),
                                   lambda: pm.Normal('T2', mu=67, sd=pm.Gamma('g', alpha=pm.Exponential('a', 1), beta=pm.Exponential('b', 1))))

        traces = {'Normal': (trace1, model1), 'Uniform': (trace2, model2), 'Hierarchical': (trace3, model3)}
        weights = {k: pm.waic(v[0], model=v[1]).WAIC for k, v in traces.items()}
        norm_factor = sum(weights.values())
        weights = {k: v / norm_factor for k, v in weights.items()}

        def weighted_time(traces, weights, param):
            avg = sum(weights[k] * traces[k][0][param].mean() for k in traces)
            return data.loc[data['bin'] >= avg]['Datetime'].iloc[0]

        sleep_list.append(weighted_time(traces, weights, 'T1'))
        wakeup_list.append(weighted_time(traces, weights, 'T2'))
        sleep_true.append(bed)
        wakeup_true.append(wakeup)
        sleep_hier.append(data.loc[data['bin'] >= trace3['T1'].mean()]['Datetime'].iloc[0])
        wakeup_hier.append(data.loc[data['bin'] >= trace3['T2'].mean()]['Datetime'].iloc[0])
        count_list.append(len(count_list) + 1)

    df_out = pd.DataFrame({
        'count': count_list,
        'sleep_avg': sleep_list,
        'wakeup_avg': wakeup_list,
        'sleep_true': sleep_true,
        'wakeup_true': wakeup_true,
        'sleep_hier': sleep_hier,
        'wakeup_hier': wakeup_hier
    })

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f"User{user_id}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"User {user_id} done. Results saved to {out_path}")

if __name__ == '__main__':
    for uid in [5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22]:
        model_comparison(uid)
