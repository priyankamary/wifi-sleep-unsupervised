import os
import datetime
from datetime import timedelta
import pandas as pd
import pymc3 as pm
from data_loader import load_result_data, load_user_fitbit
from trajectory_utils import cohort_trajectory
from bayesian_models import run_model

def process_user(user_id, config):
    user_file = os.path.join(config['user_data_dir'], f"{user_id}.csv")
    try:
        df = load_user_fitbit(user_file)
    except FileNotFoundError as e:
        print(e)
        return

    result_df = load_result_data(config['result_csv'])
    sample_dir = config['sample_dir']
    save_dir = config['save_dir']

    sleep_list, wakeup_list, sleep_true, wakeup_true, sleep_hier, wakeup_hier, count_list = [], [], [], [], [], [], []

    for fname in os.listdir(sample_dir):
        if not fname.startswith(f"user{user_id}_"):
            continue

        data = pd.read_csv(os.path.join(sample_dir, fname))
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

        track_df = result_df.loc[start_dt.strftime("%Y-%m-%d %H:%M:%S"):end_dt.strftime("%Y-%m-%d %H:%M:%S")]
        start, end = cohort_trajectory(track_df)
        if start == 0 or end == 0:
            continue

        start_bin = int((start.hour * 4 + start.minute / 15 + 24) % 96)
        end_bin = int((end.hour * 4 + end.minute / 15 + 24) % 96)

        trace1, model1 = run_model(data, lambda: pm.Normal('T1', mu=35, sd=12), lambda: pm.Normal('T2', mu=59, sd=12), 'Normal')
        trace2, model2 = run_model(data, lambda: pm.DiscreteUniform('T1', lower=start_bin, upper=start_bin+12), lambda: pm.DiscreteUniform('T2', lower=start_bin+13, upper=end_bin), 'Uniform')
        trace3, model3 = run_model(data, 
            lambda: pm.Normal('T1', mu=43, sd=pm.Gamma('g', alpha=pm.Exponential('a', 1), beta=pm.Exponential('b', 1))),
            lambda: pm.Normal('T2', mu=67, sd=pm.Gamma('g', alpha=pm.Exponential('a', 1), beta=pm.Exponential('b', 1))),
            'Hierarchical')

        traces = {'Normal': (trace1, model1), 'Uniform': (trace2, model2), 'Hierarchical': (trace3, model3)}
        weights = {k: pm.waic(v[0], model=v[1]).WAIC for k, v in traces.items()}
        norm_factor = sum(weights.values())
        weights = {k: v / norm_factor for k, v in weights.items()}

        def weighted_time(param):
            avg = sum(weights[k] * traces[k][0][param].mean() for k in traces)
            return data.loc[data['bin'] >= avg]['Datetime'].iloc[0]

        sleep_list.append(weighted_time('T1'))
        wakeup_list.append(weighted_time('T2'))
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

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"User{user_id}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"User {user_id} done. Results saved to {out_path}")
