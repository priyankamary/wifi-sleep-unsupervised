import pandas as pd

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
