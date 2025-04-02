import json
import numpy as np

filename = './consolidated_data_revised_icu_day_train_scratch.json'

with open(filename, 'r') as file:
    data = json.load(file)



for key in data.keys():
    auroc_val, auroc_test, auprc_val, auprc_test = [], [], [], []
    for inner_key in data[key].keys():
        for split in data[key][inner_key].keys():
            if split == 'val':
                if data[key][inner_key][split]['auroc'] is not None:
                    auroc_val.append(data[key][inner_key][split]['auroc'])
                if data[key][inner_key][split]['auprc'] is not None:
                    auprc_val.append(data[key][inner_key][split]['auprc'])
            elif split == 'test':
                if data[key][inner_key][split]['auroc'] is not None:
                    auroc_test.append(data[key][inner_key][split]['auroc'])
                if data[key][inner_key][split]['auprc'] is not None:
                    auprc_test.append(data[key][inner_key][split]['auprc'])


    # print(f'Average AUROC Val: {np.mean(auroc_val)}')
    # print(f'Average AUPRC Val: {np.mean(auprc_val)}')
    print(f'Average AUROC Test: {np.mean(auroc_test)}')
    print(f'Average AUPRC Test: {np.mean(auprc_test)}\n')