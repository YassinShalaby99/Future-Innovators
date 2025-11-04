#!/usr/bin/env python3
import argparse
import pandas as pd
from datetime import timedelta

def parse_args():
    p = argparse.ArgumentParser(description="Add timestamps + features to traffic CSV.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-time", default="2024-01-01 00:00:00")
    p.add_argument("--freq-seconds", type=int, default=5)
    return p.parse_args()

def engineer_features(df):
    df['Hour From TS'] = df['timestamp'].dt.hour
    df['Weekday From TS'] = df['timestamp'].dt.day_name()
    df['Is Peak Derived'] = df['Hour From TS'].between(7,9) | df['Hour From TS'].between(16,19)
    df['Is Peak Derived'] = df['Is Peak Derived'].astype(int)

    dens = df['Traffic Density'].clip(0,1)
    inv_speed = 1 - (df['Speed'].clip(0,140)/140)
    congestion = (0.7*dens + 0.3*inv_speed) * 100
    df['Congestion Score'] = congestion.round(1).clip(0,100)

    bins = [-1, 30, 60, 100]
    labels = ['Low', 'Moderate', 'High']
    df['Congestion Level'] = pd.cut(df['Congestion Score'], bins=bins, labels=labels)
    return df

def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    start = pd.Timestamp(args.start_time)
    freq = pd.to_timedelta(args.freq_seconds, unit='s')

    df['timestamp'] = [start + freq*i for i in range(len(df))]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = engineer_features(df)
    df.to_csv(args.output, index=False)
    print(f"✅ Done: {args.output}")

if __name__ == "__main__":
    main()
