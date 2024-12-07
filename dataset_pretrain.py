import pickle
import sys
import numpy as np
from utils import CycleIndex
import torch
from dataset import Dataset
import os
from tqdm import tqdm

class PretrainDataset(Dataset):
    def __init__(self, args):
        # super().__init__(args)  # Initialize the base Dataset class if necessary

        # Read data
        filepath = './data/processed/' + args.dataset + '.pkl'
        data, _, train_ids, val_ids, test_ids = pickle.load(open(filepath, 'rb'))
        args.logger.write('\nPreparing dataset ' + args.dataset + ' for pretraining')

        static_varis = self.get_static_varis()

        data = data.loc[(data.minute >= 0) & (data.minute <= 5 * 24 * 60)]
        data.loc[(data.variable == 'Age') & (data.value > 200), 'value'] = 91.4
        self.max_minute = 24 * 60

        # Remove test data and update train_ids for pretraining
        data = data.loc[~data.ts_id.isin(test_ids)]
        train_ids = np.setdiff1d(data.ts_id.unique(), val_ids)

        # train_ids, val_ids = train_ids[:500], val_ids[:100]

        # Keep variables seen in training set only
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(train_variables)]
        val_ids = data.loc[data.ts_id.isin(val_ids)].ts_id.unique()
        args.logger.write('# train, val TS: '+str([len(train_ids), len(val_ids)]))

        # Create ts_id to index mapping
        unsup_ts_ids = np.concatenate((train_ids, val_ids))
        self.ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(unsup_ts_ids)}
        data = data.loc[data.ts_id.isin(unsup_ts_ids)]
        data['ts_ind'] = data['ts_id'].map(self.ts_id_to_ind)
        N_ts = len(unsup_ts_ids)

        # Save attributes
        self.N_ts = N_ts  # Number of unique time series
        self.args = args
        self.static_varis = static_varis

        # Create sliding windows
        self.windows = self.create_sliding_windows(data, window_size=12 * 60, step_size=6 * 60)
        self.N = len(self.windows)
        args.logger.write(f'# Total sliding windows: {self.N}')
        self.splits = {
        'train': [i for i, window in enumerate(self.windows) if window['ts_id'] in train_ids],
        'val': [i for i, window in enumerate(self.windows) if window['ts_id'] in val_ids]
        }

        args.logger.write('Counting # TS Variables')

        data = self.get_static_data(data)

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
        self.var_to_ind = {v:i for i,v in enumerate(variables)}
        V = len(variables)
        args.V = V
        args.logger.write('# TS variables: '+str(V))
        values = [[] for _ in range(self.N_ts)]
        times = [[] for _ in range(self.N_ts)]
        varis = [[] for _ in range(self.N_ts)]
        data = data.sample(frac=1).sort_values(by='minute')
        for row in data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(self.var_to_ind[row.variable])
        self.values, self.times, self.varis = values, times, varis

        self.timestamps = []  
        for window in tqdm(self.windows, desc='Creating timestamps', file=sys.stdout):
            ts_id = window['ts_id']  
            start, end = window['start_time'], window['end_time']  
            ts_times = self.times[self.ts_id_to_ind[ts_id]]  
            window_times = np.array([t for t in ts_times if start <= t < end])
            self.timestamps.append(window_times)

        self.timestamps = [x[x >= 720] for x in self.timestamps]  
        delete = [i for i in range(self.N) if len(self.timestamps[i]) == 0]  
        self.splits = {k: np.setdiff1d(v, delete) for k, v in self.splits.items()}
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)
        self.V = args.V
        self.max_obs = args.max_obs


    def create_sliding_windows(self, data, window_size=720, step_size=360):
        """
        Create overlapping sliding windows for time-series data.

        Args:
            data (pd.DataFrame): The original time-series data.
            window_size (int): Size of each window in minutes (default: 720 for 12 hours).
            step_size (int): Step size in minutes (default: 360 for 6 hours).

        Returns:
            list[dict]: A list of sliding window dictionaries.
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
                sliding_windows.append({
                    'ts_id': ts_id,
                    'ts_ind': self.ts_id_to_ind[ts_id],
                    'data': window_data.copy(),
                    'start_time': start,
                    'end_time': end
                })
        return sliding_windows

    def get_batch(self, ind=None):
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
        bsz = len(ind)

        input_values = []
        input_times = []
        input_varis = []
        forecast_values = torch.zeros((bsz, self.V))
        forecast_mask = torch.zeros((bsz, self.V), dtype=torch.int)

        for b, window_idx in enumerate(ind):
            window = self.windows[window_idx]
            data = window['data'].sort_values('minute')
            ts_ind = window['ts_ind']

            # Observation window (up to max_obs)
            t1 = np.random.choice(self.timestamps[window_idx])  # minutes
            curr_times = self.times[ts_ind]
            for ix in range(len(curr_times) - 1, -1, -1):
                if curr_times[ix] == t1:
                    t1_ix = ix + 1  # start of prediction window
                    break
            t0_ix = max(0, t1_ix - self.max_obs)

            if self.args.dataset == 'mimic_iv':  # obs window max length is 24 hrs
                while t0_ix < len(curr_times) and curr_times[t0_ix] < t1 - self.max_minute:
                    t0_ix += 1
            if self.args.dataset == 'mimic_iv' and t1 > self.max_minute:  # shift times
                diff = t1 - self.max_minute
                input_times.append([x - diff for x in self.times[window_idx][t0_ix:t1_ix]])
            else:
                input_times.append(self.times[window_idx][t0_ix:t1_ix])
            input_values.append(self.values[window_idx][t0_ix:t1_ix])
            input_varis.append(self.varis[window_idx][t0_ix:t1_ix])

            # Forecast window
            t2 = t1 + 120  # 2 hours ahead for forecasting
            forecast_data = data[(data['minute'] > t1) & (data['minute'] <= t2)]
            for _, row in forecast_data.iterrows():
                vari = self.var_to_ind[row['variable']]
                forecast_mask[b, vari] = 1
                forecast_values[b, vari] = row['value']

        # Padding for batching
        num_obs = list(map(len, input_values))
        max_obs = max(num_obs)
        pad_lens = max_obs - np.array(num_obs)
        values_padded = [x + [0] * l for x, l in zip(input_values, pad_lens)]
        times_padded = [x + [0] * l for x, l in zip(input_times, pad_lens)]
        varis_padded = [x + [0] * l for x, l in zip(input_varis, pad_lens)]
        obs_mask = [[1] * l1 + [0] * l2 for l1, l2 in zip(num_obs, pad_lens)]

        # Convert to tensors
        values_tensor = torch.FloatTensor(values_padded)
        times_tensor = torch.FloatTensor(times_padded) / self.max_minute * 2 - 1  # Normalize time
        varis_tensor = torch.LongTensor(varis_padded)
        obs_mask_tensor = torch.FloatTensor(obs_mask)

        # Fetch static data corresponding to each window
        demo_tensor = torch.FloatTensor([self.demo[self.windows[i]['ts_ind']] for i in ind])

        return {
            'values': values_tensor,
            'times': times_tensor,
            'varis': varis_tensor,
            'obs_mask': obs_mask_tensor,
            'demo': demo_tensor,
            'forecast_values': forecast_values,
            'forecast_mask': forecast_mask,
        }
