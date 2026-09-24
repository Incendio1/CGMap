import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics

from data_loader import load_net_specific_data
from models.CGMap import CGMap
from models.MTGCN import MTGCNNet
from models.EMOGI import EMOGINet
from models.Chebnet import ChebNet
from models.GAT import GATNet
from models.GCN import GCNNet
from models.CGMega import CGMega
from models.JKNet import JKNet
from models.AGNN import AGNN
from models.GATV2 import GATv2
from models.Arma import Arma
from models.TAGNN import TAGCN
from utils import OPP_edge_info


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, default='CGMap')
    p.add_argument('--dataset', type=str, default='PPNet')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.0016)
    p.add_argument('--w_decay', type=float, default=0.00063)
    p.add_argument('--dropout', type=float, default=0.32)
    p.add_argument('--hidden', type=int, default=413)
    p.add_argument('--alpha', type=float, default=0.45)
    p.add_argument('--gamma', type=float, default=3.0)
    p.add_argument('--theta', type=float, default=0.75)

    cv = p.add_mutually_exclusive_group()
    cv.add_argument(
        '--cross_validation',
        dest='cross_validation',
        action='store_true',
        help='Use dataset["split_set"]. This is the default.'
    )
    cv.add_argument(
        '--no_cross_validation',
        dest='cross_validation',
        action='store_false',
        help='Use dataset["mask"] instead.'
    )
    p.set_defaults(cross_validation=True)

    p.add_argument(
        '--i_w',
        nargs=4,
        type=float,
        default=[0.05, 0.05, 0.6, 2.3]
    )
    p.add_argument('--initial_weights', nargs=4, type=float, default=None)

    p.add_argument('--OPP_layer', type=int, default=10)
    p.add_argument('--layers', nargs='+', type=int, default=[1])
    p.add_argument(
        '--agg',
        type=str,
        choices=['sum', 'weighted_sum'],
        default='sum'
    )
    p.add_argument(
        '--Init',
        type=str,
        choices=['PPR', 'PPR_raw', 'SignedPPR', 'NPPR', 'Random', 'WS'],
        default='SignedPPR'
    )
    p.add_argument('--gpr_beta', type=float, default=0.1)
    p.add_argument('--gpr_A', type=float, default=0.73)
    p.add_argument('--gpr_r', type=float, default=0.67)
    p.add_argument('--gpr_B', type=float, default=0.546)
    p.add_argument('--gpr_s', type=float, default=0.99)
    p.add_argument('--gpr_weights', nargs='+', type=float, default=None)

    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--base_seed', type=int, default=2026)
    p.add_argument('--output_root', type=str, default='results')

    return p


def get_device(device_id):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{device_id}')
    return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_mask(mask, device):
    if torch.is_tensor(mask):
        return mask.to(device)
    return torch.as_tensor(mask, device=device)


def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = torch.exp(-bce)
    loss = alpha * (1.0 - p_t) ** gamma * bce

    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()


def metric_from_logits(logits, y, mask):
    prob = torch.sigmoid(logits[mask]).view(-1).detach().cpu().numpy()
    truth = y[mask].view(-1).detach().cpu().numpy()

    auc = metrics.roc_auc_score(truth, prob)
    precision, recall, _ = metrics.precision_recall_curve(truth, prob)
    aupr = metrics.auc(recall, precision)

    return float(auc), float(aupr)


def prepare_opp(data, args, device):
    edge_path = Path('OPP_info') / f'hop_edge_index_{args.dataset}_{args.OPP_layer}'
    att_path = Path('OPP_info') / f'hop_edge_att_{args.dataset}_{args.OPP_layer}'

    if not edge_path.exists() or not att_path.exists():
        print(f'[OPP] Building cache up to hop {args.OPP_layer}...')
        OPP_edge_info(data, args)

    hop_edge_index = list(torch.load(str(edge_path), map_location='cpu'))
    hop_edge_att = list(torch.load(str(att_path), map_location='cpu'))

    for layer in sorted(set(args.layers)):
        idx = layer - 1
        hop_edge_index[idx] = hop_edge_index[idx].long().to(device)
        hop_edge_att[idx] = hop_edge_att[idx].float().view(-1).to(device)

    args.hop_edge_index = hop_edge_index
    args.hop_edge_att = hop_edge_att

    return hop_edge_index, hop_edge_att


def train_step(model, model_name, data, tr_mask, args, device):
    if model_name == 'MTGCN':
        pred, rl, c1, c2 = model(data)
        bce = F.binary_cross_entropy_with_logits(
            pred[tr_mask],
            data.y[tr_mask].view(-1, 1)
        )
        return bce / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)

    if model_name == 'EMOGI':
        pred = model(data)
        return F.binary_cross_entropy_with_logits(
            pred[tr_mask],
            data.y[tr_mask].view(-1, 1),
            pos_weight=torch.tensor([45.0], device=device)
        )

    if model_name == 'CGMap':
        pred = model(data)
        y = data.y[tr_mask].view(-1, 1)

        pos = y.sum()
        neg = y.numel() - pos
        pos_weight = (pos / neg.clamp_min(1.0)).detach()

        bce = F.binary_cross_entropy_with_logits(
            pred[tr_mask],
            y,
            pos_weight=pos_weight
        )

        foc = focal_loss(
            pred[tr_mask],
            y,
            alpha=args.alpha,
            gamma=args.gamma
        )

        return args.theta * bce + (1.0 - args.theta) * foc

    pred = model(data)

    return F.binary_cross_entropy_with_logits(
        pred[tr_mask],
        data.y[tr_mask].view(-1, 1)
    )


@torch.no_grad()
def get_logits(model, model_name, data):
    model.eval()

    if model_name == 'MTGCN':
        out, _, _, _ = model(data)
        return out

    return model(data)


def main():
    args = build_parser().parse_args()

    device = get_device(args.device)

    if not args.cross_validation:
        raise ValueError(
            'This script supports cross-validation only. '
            'Please use --cross_validation.'
        )

    model_dict = {
        'CGMap': CGMap,
        'MTGCN': MTGCNNet,
        'EMOGI': EMOGINet,
        'GAT': GATNet,
        'GCN': GCNNet,
        'CGMega': CGMega,
        'JKNet': JKNet,
        'AGNN': AGNN,
        'GATv2': GATv2,
        'Chebnet': ChebNet,
        'Arma': Arma,
        'TAGNN': TAGCN
    }

    if args.model not in model_dict:
        raise ValueError(
            f'Unknown model {args.model}. '
            f'Available: {list(model_dict)}'
        )

    data = load_net_specific_data(args)

    args.num_features = int(data.x.size(1))

    if args.model == 'CGMap':
        prepare_opp(data, args, device)

    data = data.to(device)

    layer_tag = '-'.join(map(str, args.layers))

    if args.model == 'CGMap':
        run_name = (
            f'{args.dataset}_{args.model}_{args.Init}_layers_{layer_tag}'
        )
    else:
        run_name = f'{args.dataset}_{args.model}'

    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_matrix = np.zeros((args.rounds, args.folds), dtype=float)
    aupr_matrix = np.zeros((args.rounds, args.folds), dtype=float)

    metric_rows = []

    global_fold = 0

    for rnd in range(args.rounds):
        for fold in range(args.folds):
            seed = args.base_seed + global_fold
            global_fold += 1

            set_seed(seed)

            tr_mask, te_mask = data.mask[rnd][fold]

            tr_mask = move_mask(tr_mask, device)
            te_mask = move_mask(te_mask, device)

            if tr_mask.dtype != torch.bool:
                if tr_mask.numel() != data.x.size(0):
                    tmp = torch.zeros(
                        data.x.size(0),
                        dtype=torch.bool,
                        device=device
                    )
                    tmp[tr_mask.long()] = True
                    tr_mask = tmp
                else:
                    tr_mask = tr_mask.bool()

            if te_mask.dtype != torch.bool:
                if te_mask.numel() != data.x.size(0):
                    tmp = torch.zeros(
                        data.x.size(0),
                        dtype=torch.bool,
                        device=device
                    )
                    tmp[te_mask.long()] = True
                    te_mask = tmp
                else:
                    te_mask = te_mask.bool()

            model = model_dict[args.model](args).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.w_decay
            )

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()

                loss = train_step(
                    model,
                    args.model,
                    data,
                    tr_mask,
                    args,
                    device
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            logits = get_logits(model, args.model, data)

            auc, aupr = metric_from_logits(
                logits,
                data.y,
                te_mask
            )

            auc_matrix[rnd, fold] = auc
            aupr_matrix[rnd, fold] = aupr

            metric_rows.append({
                'Round': rnd + 1,
                'Fold': fold + 1,
                'AUC': auc,
                'AUPR': aupr,
                'Seed': seed
            })

            print(
                f'Round--{rnd + 1} '
                f'CV--{fold + 1} '
                f'AUC: {auc:.5f}, '
                f'AUPR: {aupr:.5f}'
            )

        print(
            f'Round--{rnd + 1} '
            f'Mean AUC: {auc_matrix[rnd].mean():.5f}, '
            f'Mean AUPR: {aupr_matrix[rnd].mean():.5f}'
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / 'cv_metrics.csv', index=False)

    print(
        f'\n{args.model} '
        f'{args.rounds} rounds x {args.folds} folds'
        f' -- Mean AUC: {auc_matrix.mean():.6f}, '
        f'Mean AUPR: {aupr_matrix.mean():.6f}'
    )


if __name__ == '__main__':
    main()