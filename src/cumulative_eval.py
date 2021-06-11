import os
import json
import pickle
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

from src.common import logger_manager, DATETIME_FORMAT
from src.evaluation import get_sequence_statistics, default_value


def process_predictions(pred_folder: str):
    th_list = np.arange(0.1, 1.0, 0.05)
    max_data = defaultdict(lambda: {
        'recall': 0.0,
        'precision': 0.0,
        'f1-score': 0.0
    })

    for pf in sorted(
        filter(lambda x: x.endswith('.p'), os.listdir(pred_folder)),
        key=lambda x: int(x.split('_')[1])
    ): #[:2]:
        print(f"Processing... {pf}")

        data = pickle.load(open(os.path.join(pred_folder, pf), 'rb'))
        stat = get_sequence_statistics(data, th_list)

        for th, value in stat['thresholds'].items():
            max_data[th]['recall'] = max(max_data[th]['recall'], value['micro-recall'])
            max_data[th]['precision'] = max(max_data[th]['precision'], value['micro-precision'])
            max_data[th]['f1-score'] = max(max_data[th]['f1-score'], value['micro-f1_score'])

    with open(os.path.join(pred_folder, f'cumulative_{datetime.now().strftime(DATETIME_FORMAT)}.csv'), 'w') as cf:
        cf.write('step,metric,value\n')
        for key, value in max_data.items():
            cf.write(f"{key},precision,{value['precision']}\n")
            cf.write(f"{key},recall,{value['recall']}\n")
            cf.write(f"{key},f1-score,{value['f1-score']}\n")

    print(max_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Folder with input model and predictions', required=True)
    args = parser.parse_args()

    logger_manager.init_eval_logger(args.model)
    process_predictions(args.model)
