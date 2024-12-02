from builtins import input, print, range
from multiprocessing import Pool
import numpy as np
import sys
import os
import json
sys.path.append('../')
import datetime
from collections import OrderedDict
import pandas as pd
def split_trace_by_service(trace_raw_data, trace_base_dir, service_name):
    trace_data = pd.read_csv(trace_raw_data)
    service_name = service_name
    for service in service_name:
        service_data = trace_data[trace_data['servicename'] == service]
        service_data = service_data.sort_values(by='startTime_us')
        service_data.to_csv(os.path.join(trace_base_dir, f'{service}_trace.csv'), index=True)
service_name = ["compose-post-service", "home-timeline-service", "media-service",
                           "nginx-web-server", "post-storage-service", "social-graph-service",
                           "text-service", "unique-id-service", "url-shorten-service", "user-mention-service",
                           "user-service","user-timeline-service"]
trace_raw_data = r'traces.csv'
trace_base_dir = r'trace'
def trace_to_seq(df):
    window_size = 1000000
    trace_series = OrderedDict()

    for i in df['startTime_us'].values:
        trace_split_data = df[
            (df['startTime_us'] >= i) & (df['startTime_us'] < i + window_size)]
        span_data = trace_split_data['duration'].values.tolist()
        span = span_data[0]
        span_mean = np.mean(span_data)
        trace_series[str(i)] = {'duration': span, 'span_mean': span_mean}
    return trace_series
serilize_data = r'trace'
def reserve_timestamp_to_trace_serier(service_name):
    for i in service_name:
        trace_data = pd.read_csv(os.path.join(trace_base_dir, f'{i}_trace.csv'))
        trace_process_data = trace_to_seq(trace_data)
        trace_process_data_csv = pd.DataFrame.from_dict(trace_process_data, orient='index')
        trace_process_data_csv.to_csv(os.path.join(serilize_data, f'{i}_trace_seq.csv'), index=True)
serilize_data_sample = r'E:\MS\data\eadro-main\eadro-main\data\SN_process\data\trace\serilize_data_sample'
def trace_to_seq_sample(df):
    df['timestamp'] = df['startTime_s'].apply(lambda x: int(x))
    df_mean = df.groupby('timestamp')['duration'].mean()
    data = {'timestamp': df_mean.index, 'duration': df_mean.values}
    data_s = pd.DataFrame(data)
    return data_s

def reserve_timestamp_to_trace_serier(service_name):
    for i in service_name:
        trace_data = pd.read_csv(os.path.join(trace_base_dir, f'{i}_trace.csv'))
        trace_process_data = trace_to_seq_sample(trace_data)
        trace_process_data.to_csv(os.path.join(serilize_data_sample, f'{i}_trace_seq.csv'), index=False)
data_path_statuscode = r'E:\MS\data\eadro-main\eadro-main\data\SN\SN_all_spans_statuscode.csv'
trace_status_data = pd.read_csv(data_path_statuscode)
status_data = trace_status_data[trace_status_data['servicename'] == 'nginx-web-server']
data_path_statuscode_save = r'E:\MS\data\eadro-main\eadro-main\data\SN_process\data\trace'
def calcute_code_fre(df, file_path):
    df['timestamp'] = df['startTime_s'].apply(lambda x: int(x))
    df_mean = df.groupby('timestamp')['duration'].mean()
    status_count = df.groupby('timestamp')['status_code'].value_counts().unstack(fill_value=0)
    data = pd.concat([df_mean, status_count], axis=1)
    data.to_csv(os.path.join(file_path, f'nginx-web-server_status_code.csv'), index=True)
status_data_csv = calcute_code_fre(status_data, data_path_statuscode_save)
