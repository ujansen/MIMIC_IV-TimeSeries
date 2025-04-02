import torch.nn as nn
import argparse
from utils import Logger
import torch
import torch.nn.functional as F


def count_parameters(logger: Logger, model: nn.Module):
    """Print no. of parameters in model, no. of traininable parameters,
     no. of parameters in each dtype."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write('\nModel details:')
    logger.write('# parameters: '+str(total))
    logger.write('# trainable parameters: '+str(trainable)+', '\
                 +str(100*trainable/total)+'%')

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: 
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    logger.write('#params by dtype:')
    for k, v in dtypes.items():
        logger.write(str(k)+': '+str(v)+', '+str(100*v/total)+'%')


class TimeSeriesModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.demo_emb = nn.Sequential(nn.Linear(args.D, args.hid_dim*2),
                                        nn.Tanh(),
                                        nn.Linear(args.hid_dim*2, args.hid_dim))
        self.day_emb = nn.Linear(1, args.hid_dim)
        self.notes_emb = nn.Sequential(nn.Linear(args.notes_dim, args.hid_dim*2),
                                        nn.Tanh(),
                                        nn.Linear(args.hid_dim*2, args.hid_dim))

        # Change to 2 when including demo/day embedding
        # 1 = demo embedding only
        # 2 = demo and day embedding
        # 3 = demo and day and notes embedding
        ts_emb_size = args.hid_dim*2
        notes_emb_size = args.hid_dim
        
        self.pretrain = args.pretrain==1
        self.finetune = args.load_ckpt_path is not None
        if self.pretrain:
            self.forecast_head = nn.Linear(notes_emb_size, args.V)
        elif self.finetune:
            self.forecast_head = nn.Linear(ts_emb_size, args.V)
            self.notes_head = nn.Sequential(nn.Linear(notes_emb_size, args.hid_dim),
                                        nn.Tanh(),
                                        nn.Linear(args.hid_dim, args.V))
            self.binary_head = nn.Linear(args.V * 2, 1) # Keep 2 for multilabel 1 for binary
            self.pos_class_weight = torch.tensor(args.pos_class_weight)
        else:
            self.binary_head = nn.Linear(args.hid_dim * 3, 1) # Keep 2 for multilabel 1 for binary
            self.pos_class_weight = torch.tensor(args.pos_class_weight)

    def binary_cls_final(self, logits, labels):
        if labels is not None:
            return F.binary_cross_entropy_with_logits(logits, labels,
                                                    pos_weight=self.pos_class_weight)
        else:
            return torch.sigmoid(logits)
        
    def forecast_final(self, ts_emb, forecast_values, forecast_mask):
        pred = self.forecast_head(ts_emb) # bsz, V
        return (forecast_mask*(pred-forecast_values)**2).sum()/forecast_mask.sum()
