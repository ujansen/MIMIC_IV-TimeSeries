import pickle
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from utils import CycleIndex
import os
import pandas as pd

class Dataset:
    def __init__(self, args) -> None:
        """
        Initializes the dataset for the `strats` model with the `mimic_iv` dataset.
        Includes logic for fine-tuning and sliding window segmentation.
        """
        # Load data
        filepath = './data/processed/' + args.dataset + '.pkl'
        data, oc, train_ids, val_ids, test_ids = pickle.load(open(filepath, 'rb'))
        
        # Set up splits based on run and train fraction
        run, totalruns = list(map(int, args.run.split('o')))
        num_train = int(np.ceil(args.train_frac * len(train_ids)))
        start = int(np.linspace(0, len(train_ids) - num_train, totalruns)[run - 1])
        train_ids = train_ids[start:start + num_train]

        num_val = int(np.ceil(args.train_frac * len(val_ids)))
        start = int(np.linspace(0, len(val_ids) - num_val, totalruns)[run - 1])
        val_ids = val_ids[start:start + num_val]

        args.logger.write('\nPreparing dataset mimic_iv for training')

        # Filter labeled data in the first 24h and clean outliers
        data = data.loc[(data.minute >= 0) & (data.minute <= 5 * 24 * 60)]
        data.loc[(data.variable == 'Age') & (data.value > 200), 'value'] = 91.4

        # Retain variables only seen in the training set
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(train_variables)]
        curr_ids = data.ts_id.unique()
        train_ids = np.intersect1d(train_ids, curr_ids)
        val_ids = np.intersect1d(val_ids, curr_ids)
        test_ids = np.intersect1d(test_ids, curr_ids)
        args.logger.write('# train, val, test TS: '+str([len(train_ids), len(val_ids), len(test_ids)]))

        # train_ids, val_ids, test_ids = train_ids[:10], val_ids[:5], test_ids[:5]

        sup_ts_ids = np.concatenate((train_ids, val_ids, test_ids))
        self.ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(sup_ts_ids)}

        data = data.loc[data.ts_id.isin(sup_ts_ids)]
        data['ts_ind'] = data['ts_id'].map(self.ts_id_to_ind)

        # Labels
        oc = oc.loc[oc.ts_id.isin(sup_ts_ids)]
        oc['ts_ind'] = oc['ts_id'].map(self.ts_id_to_ind)
        oc = oc.sort_values(by='ts_ind')
        self.labels = np.array(oc['in_hospital_mortality'])

        self.N_ts = len(sup_ts_ids)

        self.windows = self.create_sliding_windows(data, window_size=720, step_size=360)
        self.N = len(self.windows)
        args.logger.write(f'# Total sliding windows: {self.N}')

        # Assign labels to windows
        self.window_labels = np.array([self.labels[window['ts_ind']] for window in self.windows])
        self.args = args

        # Splits
        # self.splits = {
        #     'train': [i for i, window in enumerate(self.windows) if window['ts_id'] in train_ids],
        #     'val': [i for i, window in enumerate(self.windows) if window['ts_id'] in val_ids],
        #     'test': [i for i, window in enumerate(self.windows) if window['ts_id'] in test_ids]
        # }


        split_identifier = f"train_frac_{args.train_frac}_run_{args.run}"
        train_ids_file = f'./index_splits/train_ids_{split_identifier}.npy'
        val_ids_file = f'./index_splits/val_ids_{split_identifier}.npy'
        test_ids_file = f'./index_splits/test_ids_{split_identifier}.npy'


        # Stratified Splitting
        if not (os.path.exists(train_ids_file) and os.path.exists(val_ids_file) and os.path.exists(test_ids_file)):
            train_val_ids, test_ids, train_val_labels, _ = train_test_split(
                sup_ts_ids,
                self.labels,
                test_size = len(test_ids),
                random_state = args.seed,
                stratify=self.labels
            )

            train_ids, val_ids, _, _ = train_test_split(
                train_val_ids,
                train_val_labels,
                test_size=int(args.train_frac * len(train_val_ids)),
                random_state=args.seed,
                stratify=train_val_labels
            )
            
            np.save(train_ids_file, train_ids, allow_pickle=True)
            np.save(val_ids_file, val_ids, allow_pickle=True)
            np.save(test_ids_file, test_ids, allow_pickle=True)
        else:
            train_ids = np.load(train_ids_file, allow_pickle=True)
            val_ids = np.load(val_ids_file, allow_pickle=True)
            test_ids = np.load(test_ids_file, allow_pickle=True)

        self.splits = {
            'train': [i for i, window in enumerate(self.windows) if window['ts_id'] in train_ids],
            'val': [i for i, window in enumerate(self.windows) if window['ts_id'] in val_ids],
            'test': [i for i, window in enumerate(self.windows) if window['ts_id'] in test_ids]
        }
        self.splits['eval_train'] = self.splits['train'][:2000]

        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)

        num_train_windows = len(self.splits['train'])
        num_train_pos = self.window_labels[self.splits['train']].sum()
        num_train_neg = num_train_windows - num_train_pos
        if num_train_pos == 0:
            args.pos_class_weight = 1.0  # Default weight if no positive samples
            args.logger.write('Warning: No positive samples in training split. Setting pos_class_weight to 1.0.')
        else:
            args.pos_class_weight = num_train_neg / num_train_pos
        args.logger.write('pos class weight: ' + str(args.pos_class_weight))

        def compute_positive_percentage(split_indices):
            if len(split_indices) == 0:
                return 0.0
            return self.window_labels[split_indices].sum() / len(split_indices)

        train_pos_percent = compute_positive_percentage(self.splits['train'])
        val_pos_percent = compute_positive_percentage(self.splits['val'])
        test_pos_percent = compute_positive_percentage(self.splits['test'])

        args.logger.write(
            '% pos class in train, val, test splits: ' +
            str([train_pos_percent, val_pos_percent, test_pos_percent])
        )

        data = self.get_static_data(data)
        data = data.sample(frac=1)
        data = data.groupby('ts_id').head(args.max_obs)

        args.logger.write('Counting # TS Variables')

        # Fine-tuning setup
        args.finetune = args.load_ckpt_path is not None
        if args.finetune:
            pt_var_path = os.path.join(os.path.dirname(args.load_ckpt_path), 'pt_saved_variables.pkl')
            variables, means_stds, max_minute = pickle.load(open(pt_var_path, 'rb'))
            self.max_minute = max_minute
        else:
            # Normalize variables if not fine-tuning
            means_stds = data.loc[data.ts_id.isin(train_ids)].groupby(
                'variable').agg({'value': ['mean', 'std']})
            means_stds.columns = [col[1] for col in means_stds.columns]
            means_stds.loc[means_stds['std'] == 0, 'std'] = 1
            data = data.merge(means_stds.reset_index(), on='variable', how='left')
            data['value'] = (data['value'] - data['mean']) / data['std']
            variables = data.variable.unique()
            self.max_minute = 12 * 60  # Maximum observation window for mimic_iv
        

        if not(args.finetune):
            variables = data.variable.unique()
        self.var_to_ind = {v:i for i,v in enumerate(variables)}
        V = len(variables)
        args.V = V
        args.logger.write('# TS variables: '+str(V))

        values = [[] for i in range(self.N_ts)]
        times = [[] for i in range(self.N_ts)]
        varis = [[] for i in range(self.N_ts)]
        data['minute'] = data['minute']/self.max_minute*2-1
        for row in data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(self.var_to_ind[row.variable])
        self.values, self.times, self.varis = values, times, varis


    def create_sliding_windows(self, data, window_size=720, step_size=360):
        """
        Segments each ts_id into overlapping sliding windows.

        Args:
            data (pd.DataFrame): Time-series data.
            window_size (int): Size of each window in minutes (default: 12 hours).
            step_size (int): Step size in minutes (default: 6 hours).

        Returns:
            list[dict]: A list of dictionaries representing each sliding window.
        """
        sliding_windows = []
        for ts_id, group in tqdm(data.groupby('ts_id'), desc='Creating Sliding Windows', file=sys.stdout):
            group = group.sort_values('minute')
            max_time = group['minute'].max().astype(int)
            start_times = range(0, max_time - window_size + 1, step_size)
            for start in start_times:
                end = start + window_size
                window_data = group[(group['minute'] >= start) & (group['minute'] < end)]
                if len(window_data) < 0.8 * window_size:  # Keep windows with at least 80% coverage
                    continue
                
                # Compute ICU Day
                day = start // (24 * 60)

                sliding_windows.append({
                    'ts_id': ts_id,
                    'ts_ind': group['ts_ind'].iloc[0],
                    'data': window_data.copy(),
                    'day': day,
                    'start_time': start,
                    'end_time': end
                })
        return sliding_windows
    
    def get_static_varis(self):
        return ['Age', 'Gender']

    def get_static_data(self, data):
        """
        Retrieves and normalizes static variables.

        Args:
            data (pd.DataFrame): Time-series data.

        Returns:
            np.ndarray: Normalized static features.
        """
        static_varis = ['Age', 'Gender'] 
        static_data = data.loc[data.variable.isin(static_varis)]  
        static_var_to_ind = {v: i for i, v in enumerate(static_varis)} 
        unique_ts_ids = static_data['ts_id'].unique()  

        ts_id_to_demo_ind = {ts_id: i for i, ts_id in enumerate(unique_ts_ids)}
        D = len(static_var_to_ind)  
        demo = np.zeros((self.N_ts, D)) 

        for ts_id, group in static_data.groupby('ts_id'):
            demo_ind = self.ts_id_to_ind[ts_id]  
            for row in group.itertuples():
                var_ind = static_var_to_ind[row.variable]  
                demo[demo_ind, var_ind] = row.value  

        train_demo_inds = [
        self.ts_id_to_ind[self.windows[i]['ts_id']]
        for i in self.splits['train']
        if self.windows[i]['ts_id'] in self.ts_id_to_ind
        ]
        means = demo[train_demo_inds].mean(axis=0, keepdims=True)
        stds = demo[train_demo_inds].std(axis=0, keepdims=True)
        stds = (stds==0) + (stds>0)*stds
        demo = (demo - means) / stds

        self.demo = demo
        self.args.D = D
        return data

    def get_batch(self, ind=None):
        """
        Retrieves a batch of data.

        Args:
            ind (list[int], optional): List of window indices for the batch.
                                       Defaults to retrieving a batch using the cycler.

        Returns:
            dict: Batch data.
        """
        if ind is None:
            ind = self.train_cycler.get_batch_ind()

        if isinstance(ind, np.ndarray):
            ind = ind.astype(int).tolist()
        else:
            ind = [int(i) for i in ind]

        values, times, varis, obs_masks, day_list = [], [], [], [], []
        for i in ind:
            window = self.windows[i]
            data = window['data']

            day = window['day']
            day_list.append(day)

            values.append(data['value'].tolist())
            times.append(data['minute'].tolist())
            varis.append([self.var_to_ind[v] for v in data['variable']])
            obs_masks.append([1] * len(data))

        # Normalize day
        max_day = 4
        day_tensor = torch.FloatTensor(day_list).unsqueeze(1) / max_day

        # Pad sequences
        num_obs = [len(v) for v in values]
        max_obs = max(num_obs)
        pad_lens = max_obs - np.array(num_obs)

        padded_values = [v + [0] * p for v, p in zip(values, pad_lens)]
        padded_times = [t + [0] * p for t, p in zip(times, pad_lens)]
        padded_varis = [v + [0] * p for v, p in zip(varis, pad_lens)]
        padded_obs_masks = [m + [0] * p for m, p in zip(obs_masks, pad_lens)]

        # Normalize times
        times = torch.FloatTensor(padded_times) / (self.max_minute) * 2 - 1

        return {
            'values': torch.FloatTensor(padded_values),
            'times': times,
            'varis': torch.LongTensor(padded_varis),
            'obs_mask': torch.FloatTensor(padded_obs_masks),
            'demo': torch.FloatTensor([self.demo[self.windows[i]['ts_ind']] for i in ind]),
            'day': day_tensor,
            'labels': torch.FloatTensor(self.window_labels[ind])
        }
