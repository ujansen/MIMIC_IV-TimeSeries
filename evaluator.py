import sys
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np

class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = '+split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0,num_samples,self.args.eval_batch_size),
                    desc='Running forward pass', file=sys.stdout)
        true, pred = [], []
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start+self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            if len(batch) == 0:
                continue
            true.append(batch['labels'])
            del batch['labels']
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            with torch.no_grad():
                pred.append(model(**batch).cpu())
        if len(true) == 0 or len(pred) == 0:  # Check for empty results
            self.args.logger.write("Warning: No valid data in evaluation. Skipping metrics calculation.")
            return {'auroc': None, 'auprc': None, 'minrp': None, 'accuracy': None}
        
        true, pred = torch.cat(true), torch.cat(pred)

        unique_classes = true.unique()
        if len(unique_classes) < 2:
            self.args.logger.write(
                "Warning: Only one class present in y_true. Skipping AUROC and PR curve calculation."
            )
            return {'auroc': None, 'auprc': None, 'minrp': None, 'accuracy': None}
        
        pred_labels = (pred > 0.5).float()
        correct = (pred_labels == true).sum().item()
        accuracy = correct / len(true)

        precision, recall, _ = precision_recall_curve(true, pred)
        if np.isnan(precision).any() or np.isnan(recall).any():
            self.args.logger.write("Warning: Precision or recall contains NaN. Skipping PR AUC calculation.")
            pr_auc, minrp = None, None
        else:
            pr_auc = auc(recall, precision)
            minrp = np.minimum(precision, recall).max()

        roc_auc = roc_auc_score(true, pred)
        result = {'auroc': roc_auc, 'auprc': pr_auc, 'minrp': minrp, 'accuracy': accuracy}
        
        if train_step is not None:
            self.args.logger.write('Result on '+split+' split at train step '
                            +str(train_step)+': '+str(result))
        return result
