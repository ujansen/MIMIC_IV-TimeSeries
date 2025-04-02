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
        # print("PREDICTIONS ARE: ", pred)

        unique_classes = true.unique()
        if len(unique_classes) < 2:
            self.args.logger.write(
                "Warning: Only one class present in y_true. Skipping AUROC and PR curve calculation."
            )
            return {'auroc': None, 'auprc': None, 'minrp': None, 'accuracy': None}
        
        pred_labels = (pred >= 0.5).float()
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

        # result = {
        #     'auroc': {},
        #     'auprc': {},
        #     'minrp': {},
        #     'accuracy': {}
        # }

        # days = ['7 days', '30 days']

        # for c in range(2):
        #     true_c = true[:, c]
        #     pred_c = pred[:, c]
            
        #     # Skip if only one class present (metrics undefined)
        #     if len(true_c.unique()) < 2:
        #         self.args.logger.write(f"Warning: Only one class present in class {c}. Skipping metrics.")
        #         result['auroc'][days[c]] = None
        #         result['auprc'][days[c]] = None
        #         result['minrp'][days[c]] = None
        #         result['accuracy'][days[c]] = None
        #         continue

        #     # AUROC
        #     result['auroc'][days[c]] = roc_auc_score(true_c, pred_c)

        #     # AUPRC and MinRP
        #     precision, recall, _ = precision_recall_curve(true_c, pred_c)
        #     if np.isnan(precision).any() or np.isnan(recall).any():
        #         self.args.logger.write(f"Warning: NaN in precision/recall for class {c}. Skipping PR metrics.")
        #         result['auprc'][days[c]] = None
        #         result['minrp'][days[c]] = None
        #     else:
        #         result['auprc'][days[c]] = auc(recall, precision)
        #         result['minrp'][days[c]] = np.minimum(precision, recall).max()

        #     # Accuracy
        #     pred_labels = (pred_c >= 0.5).float()
        #     result['accuracy'][days[c]] = (pred_labels == true_c).float().mean().item()

        # if train_step is not None:
        #     self.args.logger.write('Result on ' + split + ' split at train step '
        #                            + str(train_step) + ': ' + str(result))
        # return result
