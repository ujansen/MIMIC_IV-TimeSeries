import pickle
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from utils import CycleIndex
import os
import pandas as pd

from embedding_generator import get_embeddings

class Dataset:
    def __init__(self, args) -> None:
        """
        Initializes the dataset for the `strats` model with the `mimic_iv` dataset.
        Includes logic for fine-tuning and sliding window segmentation.
        """
        # Load data
        filepath = './data/processed/' + args.dataset + '.pkl'
        discharge_filepath = './data/processed/' + args.dataset + '_disch.pkl'
        data, oc, train_ids, val_ids, test_ids = pickle.load(open(filepath, 'rb'))
        self.notes = pd.read_csv('./data/notes/final_radiology_notes.csv.gz', compression='gzip')
        _, self.admissions, _, _, _ = pickle.load(open(discharge_filepath, 'rb'))
        
        # Set up splits based on run and train fraction
        run, totalruns = list(map(int, args.run.split('o')))
        num_train = int(np.ceil(args.train_frac * len(train_ids)))
        start = int(np.linspace(0, len(train_ids) - num_train, totalruns)[run - 1])
        train_ids = train_ids[start:start + num_train]

        num_val = int(np.ceil(args.train_frac * len(val_ids)))
        start = int(np.linspace(0, len(val_ids) - num_val, totalruns)[run - 1])
        val_ids = val_ids[start:start + num_val]
        static_varis = self.get_static_varis()

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

        sup_ts_ids = np.concatenate((train_ids, val_ids, test_ids))
        ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(sup_ts_ids)}

        data = data.loc[data.ts_id.isin(sup_ts_ids)]
        data['ts_ind'] = data['ts_id'].map(ts_id_to_ind)

        static_ii = data.variable.isin(['Age', 'Gender'])
        static_data = data.loc[static_ii]
        data = data.loc[~static_ii]

        # Labels
        oc = oc.loc[oc.ts_id.isin(sup_ts_ids)]
        oc['ts_ind'] = oc['ts_id'].map(ts_id_to_ind)
        oc = oc.sort_values(by='ts_ind')
        self.labels = np.array(oc['in_hospital_mortality'])
        # y = np.array(oc['discharge_thirty'])
        # self.admissions = oc
        N = len(sup_ts_ids)

        self.N = N
        # self.y = y
        self.args = args

        self.splits = {'train':[ts_id_to_ind[i] for i in train_ids],
                       'val':[ts_id_to_ind[i] for i in val_ids],
                       'test':[ts_id_to_ind[i] for i in test_ids]}

        self.splits['eval_train'] = self.splits['train'][:2000]

        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)

        num_train, num_train_pos = len(train_ids), self.labels[self.splits['train']].sum()
        args.pos_class_weight = (num_train-num_train_pos)/num_train_pos
        args.logger.write('pos class weight: '+str(args.pos_class_weight))
        args.logger.write('% pos class in train, val, test splits: '
                          +str([num_train_pos/num_train, 
                                self.labels[self.splits['val']].sum()/len(val_ids),
                                self.labels[self.splits['test']].sum()/len(test_ids)]))

        self.get_static_data(static_data)
        data = data.sample(frac=1)
        data = data.groupby('ts_id').head(args.max_obs)

        args.logger.write('Counting # TS Variables')

        # Fine-tuning setup
        args.finetune = args.load_ckpt_path is not None
        if args.finetune:
            pt_var_path = os.path.join(os.path.dirname(args.load_ckpt_path), 'pt_saved_variables.pkl')
            variables, means_stds, max_minute = pickle.load(open(pt_var_path, 'rb'))
            self.max_minute = max_minute
            data = data.merge(means_stds.reset_index(), on='variable', how='left')
            data['value'] = (data['value'] - data['mean']) / data['std']
            data = data.drop(columns=['mean', 'std'])
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

        values = [[] for i in range(N)]
        times = [[] for i in range(N)]
        varis = [[] for i in range(N)]
        # data['minute'] = data['minute']/self.max_minute*2-1
        for row in data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(self.var_to_ind[row.variable])
        self.values, self.times, self.varis = values, times, varis

        self.data = data


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

    def get_static_data(self, static_data):
        """
        Retrieves and normalizes static variables.

        Args:
            data (pd.DataFrame): Time-series data.

        Returns:
            np.ndarray: Normalized static features.
        """
         # remove static vars from data
        static_var_to_ind = {v:i for i,v in enumerate(['Age', 'Gender'])}
        D = len(static_var_to_ind)

        demo = np.zeros((self.N, D))
        for row in tqdm(static_data.itertuples()):
            var_ind = static_var_to_ind[row.variable]
            demo[row.ts_ind, var_ind] = row.value
        
        train_ind = self.splits['train']
        means = demo[train_ind].mean(axis=0, keepdims=True)
        stds = demo[train_ind].std(axis=0, keepdims=True)
        stds = (stds==0) + (stds>0)*stds
        demo = (demo-means)/stds
        self.args.logger.write('# static features: '+str(D))
        # to save
        self.demo = demo
        self.args.D = D


    def get_readmission(self, ts_id, end_time):
        window_lens = [end_time + (7 * 24 * 60), end_time + (30 * 24 * 60)]
        subject_id = self.admissions[self.admissions['ts_id'] == ts_id]['subject_id'].iloc[0]
        admits = self.admissions[self.admissions['subject_id'] == subject_id]
        relevant_admits = [admits[(admits['minutes'] > end_time) & (admits['minutes'] < window_lens[0])],
                           admits[(admits['minutes'] > end_time) & (admits['minutes'] < window_lens[1])]]
        output = [int(len(relevant_admits[0]) > 0), int(len(relevant_admits[1]) > 0)]
        return output


    def get_notes(self, ts_id, end_time):
        subject_id = self.admissions[self.admissions['ts_id'] == ts_id]['subject_id'].iloc[0]
        # subject_admissions = self.admissions[self.admissions['subject_id'] == subject_id]
        # relevant_admissions = subject_admissions[subject_admissions['minutes'] <= start_time]
        # relevant_admissions = relevant_admissions.sort_values(by='minutes')
        
        # if len(relevant_admissions) == 0:
        #     print(f'No relevant admissions found for ts_id {ts_id}. No time before start_time: {start_time}.')
        #     return None
        # most_recent_admission_time = relevant_admissions['minutes'].iloc[-1]

        relevant_notes = self.notes[(self.notes['subject_id'] == subject_id) &
                                   (self.notes['minutes']) <= end_time]['text']
        
        if len(relevant_notes) > 0:
            # print(f'Embedding generated for ts_id {ts_id}')
            embedding = get_embeddings(relevant_notes.iloc[-1])
            return embedding
        print(f'No notes found for ts_id {ts_id}')
        return None

 

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
        bsz = len(ind)
        missing_notes = 0
        
        values, times, varis, obs_masks, day_list, valid_demo, valid_labels, valid_notes, notes_masks = [], [], [], [], [], [], [], [], []

        for i in ind:
            ts_data = self.data[self.data['ts_ind'] == i].copy()
            ts_id = ts_data['ts_id'].iloc[0]
            ts_data = ts_data.sort_values('minute')
            
            max_time = ts_data['minute'].max()
            min_time = ts_data['minute'].min()

            num_possible_windows = (max_time - min_time - 720) // 360 + 1
            if num_possible_windows > 0:
                t0 = min_time + 360 * np.random.randint(num_possible_windows)
            else:
                t0 = min_time

            start_time = t0
            end_time = start_time + 720

            window_data = ts_data[(ts_data['minute'] >= start_time) & (ts_data['minute'] < end_time)]
            if len(window_data) == 0:
            # Skip if no data in the window
                print(f"No data in window for ts_id {i}. Start: {start_time}, End: {end_time}. Skipping.")
                continue

            day = start_time // (24 * 60)
            day_list.append(day)

            values.append(window_data['value'].tolist())
            times.append(window_data['minute'].tolist())
            varis.append([self.var_to_ind[v] for v in window_data['variable']])
            obs_masks.append([1] * len(window_data))

            valid_demo.append(self.demo[i])
            valid_labels.append(self.labels[i])
            notes = self.get_notes(ts_id, end_time)
            default_notes = torch.zeros(768)

            if notes is None:
                missing_notes += 1
                notes = default_notes  # Tensor of zeros
                notes_mask = 0
            else: 
                if not isinstance(notes, torch.Tensor):
                    notes = torch.FloatTensor(notes)
                notes_mask = 1
            valid_notes.append(notes)
            notes_masks.append(notes_mask)
            # valid_labels.append(self.get_readmission(ts_id, end_time))
            # print("VALID LABELS IS: ", valid_labels)
        
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

        self.args.logger.write(f"Missing notes in batch: {missing_notes}/{bsz} ({missing_notes/bsz:.2%})")

        return {
            'values': torch.FloatTensor(padded_values),
            'times': times,
            'varis': torch.LongTensor(padded_varis),
            'obs_mask': torch.FloatTensor(padded_obs_masks),
            'demo': torch.FloatTensor(valid_demo),
            'notes': torch.stack(valid_notes),
            'notes_mask': torch.FloatTensor(notes_masks).unsqueeze(1),
            'day': day_tensor,
            'labels': torch.FloatTensor(valid_labels)
        }