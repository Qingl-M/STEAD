import pandas as pd
import json
import os
import pytz
import re
import time
from datetime import datetime
tz = pytz.timezone('Asia/Shanghai')
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

for ds in dataset:
    sn_logs_data_concat = {"compose-post-service": [], "home-timeline-service": [], "media-service": [],
                           "nginx-web-server": [], "post-storage-service": [], "social-graph-service": [],
                           "text-service": [],
                           "unique-id-service": [], "url-shorten-service": [], "user-mention-service": [],
                           "user-service": [],
                           "user-timeline-service": []}
    tt_logs_data_concat = {"ts-assurance-service": [], "ts-auth-service": [], "ts-basic-service": [],
                           "ts-cancel-service": [], "ts-config-service": [], "ts-contacts-service": [],
                           "ts-food-map-service": [],"ts-food-service": [], "ts-inside-payment-service": [], "ts-notification-service": [],
                           "ts-order-other-service": [],"ts-order-service": [],"ts-payment-service": [], "ts-preserve-service": [], "ts-price-service": [],
                           "ts-route-plan-service": [],"ts-route-service": [],
                           "ts-seat-service": [], "ts-security-service": [], "ts-station-service": [],
                           "ts-ticketinfo-service": [], "ts-train-service": [],
                           "ts-travel2-service": [], "ts-travel-plan-service": [],"ts-travel-service": [],
                           "ts-user-service": [], "ts-verification-code-service": []
                           }
    json_file_path = os.path.join(log_data_dir, ds, f'{ds}_logall.json')
    for path in os.listdir(os.path.join(log_data_dir, ds)):
        json_file_path = os.path.join(log_data_dir, ds, path, 'logs.json')  #每个span.json文件的路径
        if not os.path.exists(json_file_path):
            continue
        if ds == 'SN':
            sn_log_data = read_json(json_file_path)
            for service,sn_log in sn_log_data.items():
                servicename = service
                sn_logs_data_concat[service].extend(sn_log)

        if ds == 'TT':
            tt_log_data = read_json(json_file_path)
            for tservice,tt_log in tt_log_data.items():
                servicename = tservice
                tt_logs_data_concat[tservice].extend(tt_log)

    if ds == 'SN':
        with open(os.path.join(log_data_dir, ds, f'{ds}_all_logs.json'), 'w') as f:
            json.dump(sn_logs_data_concat, f)
    if ds == 'TT':
        with open(os.path.join(log_data_dir, ds, f'{ds}_all_logs.json'), 'w') as f:
            json.dump(tt_logs_data_concat, f)

def logs_to_csv(log_data_list, csv_file_path):
    data_csvs = pd.DataFrame(columns=['date', 'service', 'log_content'])
    for service,log_list in log_data_list.items():
        for log in log_list:
            log_datetime = ' '.join(log.split(' ')[:2])
            log_datetime = pd.to_datetime(log_datetime, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            log_content = log
            data_csvs = data_csvs._append({'date': log_datetime, 'service': service, 'log_content': log_content}, ignore_index=True)
    data_csvs.to_csv(csv_file_path, index=True)
    return data_csvs

