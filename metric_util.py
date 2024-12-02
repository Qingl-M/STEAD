import pandas as pd
import json
import os
import pytz
import time
from datetime import datetime
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data
service_name = ["compose-post-service", "home-timeline-service", "media-service",
                           "nginx-web-server", "post-storage-service", "social-graph-service",
                           "text-service", "unique-id-service", "url-shorten-service", "user-mention-service",
                           "user-service","user-timeline-service"]
metric_data_dir = r'SN'
save_metric_dir = r'metric'
for service in service_name:
    service_metrics = []
    for path in os.listdir(os.path.join(metric_data_dir)):
        file_path = os.path.join(metric_data_dir, path, 'metrics')
        if not os.path.exists(file_path):
            continue
        service_metric = read_csv(os.path.join(file_path, f'{service}.csv'))
        service_metrics.append(service_metric)
    result_service_metrics = pd.concat(service_metrics, ignore_index=True)
    csv_file_path = os.path.join(save_metric_dir, f'{service}_all_metric.csv')
    result_service_metrics.to_csv(csv_file_path, index=False)

