import pickle
import sys
import numpy as np
from utils import CycleIndex
import torch
from dataset import Dataset
import os
from tqdm import tqdm
import pandas as pd
from embedding_generator import get_embeddings

class PretrainDataset(Dataset):
    def __init__(self, args):
        # super().__init__(args)  # Initialize the base Dataset class if necessary

        # Read data
        filepath = './data/processed/' + args.dataset + '.pkl'
        discharge_filepath = './data/processed/' + args.dataset + '_disch.pkl'
        data, _, train_ids, val_ids, test_ids = pickle.load(open(filepath, 'rb'))
        self.notes = pd.read_csv('./data/notes/final_radiology_notes.csv.gz', compression='gzip')
        _, self.admissions, _, _, _ = pickle.load(open(discharge_filepath, 'rb'))
        args.logger.write('\nPreparing dataset ' + args.dataset + ' for pretraining')

        static_varis = self.get_static_varis()

        data = data.loc[(data.minute >= 0) & (data.minute <= 5 * 24 * 60)]
        data.loc[(data.variable == 'Age') & (data.value > 200), 'value'] = 91.4
        self.max_minute = 12 * 60

        # Remove test data and update train_ids for pretraining
        data = data.loc[~data.ts_id.isin(test_ids)]
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(train_variables)]
        train_ids = np.intersect1d(train_ids, data.ts_id.unique())
        val_ids = np.intersect1d(val_ids, data.ts_id.unique())
        args.logger.write('# train, val TS: '+str([len(train_ids), len(val_ids)]))

        # Create ts_id to index mapping
        # unsup_ts_ids = np.concatenate((train_ids, val_ids))
        # self.ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(unsup_ts_ids)}
        # data = data.loc[data.ts_id.isin(unsup_ts_ids)]
        # data['ts_ind'] = data['ts_id'].map(self.ts_id_to_ind)
        # N_ts = len(unsup_ts_ids)

        unsup_ts_ids = np.concatenate((train_ids, val_ids))
        ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(unsup_ts_ids)}
        data = data.loc[data.ts_id.isin(unsup_ts_ids)]
        data['ts_ind'] = data['ts_id'].map(ts_id_to_ind)

        static_ii = data.variable.isin(['Age', 'Gender'])
        static_data = data.loc[static_ii]
        data = data.loc[~static_ii]

        N = len(unsup_ts_ids)

        # self.admissions = self.admissions[self.admissions['ts_id'].isin(unsup_ts_ids)]

        # Save attributes
        # self.N_ts = N_ts  # Number of unique time series
        self.N = N
        self.args = args
        self.static_varis = static_varis
        self.splits = {'train':[ts_id_to_ind[i] for i in train_ids],
                       'val':[ts_id_to_ind[i] for i in val_ids]}

        # Create sliding windows
        # self.windows = self.create_sliding_windows(data, window_size=12 * 60, step_size=6 * 60)
        # self.N = len(self.windows)
        # args.logger.write(f'# Total sliding windows: {self.N}')
        # self.splits = {
        # 'train': [i for i, window in enumerate(self.windows) if window['ts_id'] in train_ids],
        # 'val': [i for i, window in enumerate(self.windows) if window['ts_id'] in val_ids]
        # }

        args.logger.write('Counting # TS Variables')

        self.get_static_data_pretrain(static_data)

        # Initialize lists for window data
        means_stds = data.loc[data.ts_id.isin(train_ids)].groupby(
                                'variable').agg({'value':['mean', 'std']})
        means_stds.columns = [col[1] for col in means_stds.columns]
        means_stds.loc[means_stds['std']==0, 'std'] = 1
        data = data.merge(means_stds.reset_index(), on='variable', how='left')
        data['value'] = (data['value']-data['mean'])/data['std']

        # prepare time series inputs
        variables = data.variable.unique()
        pickle.dump([variables, means_stds, self.max_minute], 
                    open(os.path.join(args.output_dir, 'pt_saved_variables.pkl'),'wb'))
        var_to_ind = {v:i for i,v in enumerate(variables)}
        V = len(variables)
        args.V = V
        args.logger.write('# TS variables: '+str(V))
        values = [[] for _ in range(N)]
        times = [[] for _ in range(N)]
        varis = [[] for _ in range(N)]
        data = data.sample(frac=1).sort_values(by='minute')
        for row in data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(var_to_ind[row.variable])
        self.values, self.times, self.varis = values, times, varis

        self.timestamps = [np.array(sorted(list(set(x)))[:-1]) for x in self.times]
        self.timestamps = [x[x>=720] for x in self.timestamps] # atleast 12 hrs
        delete = [i for i in range(self.N) if len(self.timestamps[i])==0]
        self.splits = {k:np.setdiff1d(v,delete) for k,v in self.splits.items()}
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)
        self.V = args.V
        self.max_obs = args.max_obs

        self.data = data
    
    def get_static_data_pretrain(self, static_data):
        static_var_to_ind = {v:i for i,v in enumerate(['Age', 'Gender'])}
        D = len(static_var_to_ind)

        demo = np.zeros((self.N, D))
        for row in tqdm(static_data.itertuples()):
            print(row)
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


    def get_notes(self, ts_id, start_time):
        subject_ids = self.admissions[self.admissions['ts_id'] == ts_id]['subject_id']
        if len(subject_ids) == 0:
            print(f'Skipping notes for ts_id {ts_id} as no subject_id found')
            return None
        subject_id = subject_ids.iloc[0]
        subject_admissions = self.admissions[self.admissions['subject_id'] == subject_id]
        relevant_admissions = subject_admissions[subject_admissions['minutes'] <= start_time]
        relevant_admissions = relevant_admissions.sort_values(by='minutes')
        
        if len(relevant_admissions) == 0:
            return None
        most_recent_admission_time = relevant_admissions['minutes'].iloc[-1]

        relevant_notes = self.notes[(self.notes['subject_id'] == subject_id) &
                                   (self.notes['minutes']) < most_recent_admission_time]['text']
        
        if len(relevant_notes) > 0:
            embedding = get_embeddings(relevant_notes.iloc[-1])
            return embedding
        return None

    def get_batch(self, ind=None):
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
        bsz = len(ind)
        missing_notes = 0

        input_values = []
        input_times = []
        input_varis = []
        day_list = []
        valid_notes = []
        notes_masks = []
        forecast_values = torch.zeros((bsz, self.V))
        forecast_mask = torch.zeros((bsz, self.V), dtype=torch.int)

        for b,i in enumerate(ind):
            t1 = np.random.choice(self.timestamps[i]) # minutes
            curr_times = self.times[i]
            for ix in range(len(curr_times)-1,-1,-1):
                if curr_times[ix]==t1:
                    t1_ix = ix+1 # start of prediction window
                    break
            t0_ix = max(0,t1_ix-self.max_obs)
            if self.args.dataset=='mimic_iv': # obs window max length is 24 hrs
                while curr_times[t0_ix]<t1-self.max_minute:
                    t0_ix += 1
            if self.args.dataset=='mimic_iv' and t1>self.max_minute: # shift times
                diff = t1-self.max_minute
                input_times.append(list(np.array(self.times[i][t0_ix:t1_ix])-diff))
            else:
                input_times.append(self.times[i][t0_ix:t1_ix])
            input_values.append(self.values[i][t0_ix:t1_ix])
            input_varis.append(self.varis[i][t0_ix:t1_ix])

            day = t1 // (24 * 60)  # ICU day based on t1
            day_list.append(day)

            ts_data = self.data[self.data['ts_ind'] == i].copy()
            ts_id = ts_data['ts_id'].iloc[0]

            notes = self.get_notes(ts_id, t1)
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

            t2 = t1+120 # prediction window is 2 hrs
            for t2_ix in range(t1_ix, len(curr_times)):
                if curr_times[t2_ix]>t2:
                    break
            # t2_ix: last+1 for prediction window
            curr_varis = self.varis[i]
            curr_values = self.values[i]
            for ix in range(t2_ix-1,t1_ix-1,-1):
                vari = curr_varis[ix]
                val = curr_values[ix]
                forecast_mask[b,vari] = 1
                forecast_values[b,vari] = val

        num_obs = list(map(len, input_values))
        max_obs = max(num_obs)
        pad_lens = max_obs-np.array(num_obs)
        values = [x+[0]*(l) for x,l in zip(input_values,pad_lens)]
        times = [x+[0]*(l) for x,l in zip(input_times,pad_lens)]
        varis = [x+[0]*(l) for x,l in zip(input_varis,pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        times = times/self.max_minute*2-1
        varis = torch.LongTensor(varis)
        obs_mask = [[1]*l1+[0]*l2 for l1,l2 in zip(num_obs,pad_lens)]
        obs_mask = torch.LongTensor(obs_mask)

        max_day = 4 
        day_tensor = torch.FloatTensor(day_list).unsqueeze(1) / max_day

        self.args.logger.write(f"Missing notes in batch: {missing_notes}/{bsz} ({missing_notes/bsz:.2%})")

        return {
            'values': values,
            'times': times,
            'varis': varis,
            'obs_mask': obs_mask,
            'demo': torch.FloatTensor(self.demo[ind]),
            'notes': torch.stack(valid_notes),
            'notes_mask': torch.FloatTensor(notes_masks).unsqueeze(1),
            'day': day_tensor,
            'forecast_values': forecast_values,
            'forecast_mask': forecast_mask
        }
