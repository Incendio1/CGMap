import argparse
import pandas as pd
import torch.nn.functional as F
from sklearn import metrics
from models.MTGCN import MTGCNNet
from models.EMOGI import EMOGINet
from models.Chebnet import ChebNet
from models.PMLP import PMLP
from models.GAT import GATNet
from models.GCN import GCNNet
from models.CGMap import CGMap
from models.CGMega import CGMega
from models.JKNet import JKNet
from models.AGNN import AGNN
from models.GATV2 import GATv2
from models.Arma import Arma
from models.TAGNN import TAGCN
from data_loader import load_net_specific_data
from os import path
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CGMap', help='Model name, options:[MTGCN, EMOGI, Chebnet, GAT, GCN, SVM].')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--device', type=int, default=0, help='The id of GPU.')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--theta', type=float, default=0.95)
parser.add_argument('--cross_validation', type=bool, default=True, help='Run 5-CV test.')
parser.add_argument('--w_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', type=str, default='PPNet')
parser.add_argument('--OPP_layer', type=int, default=10)
parser.add_argument('--layers', nargs='+', type=int)
parser.add_argument('--OPP_dataset', type=str, default='PPNet')
parser.add_argument('--agg', type=str, default='sum')
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--dr', type=float, default=0)
parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS'],
                        default='PPR')
parser.add_argument('--num_layers', type=int, default=10)

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dict = {
              'CGMap': CGMap,
              'MTGCN': MTGCNNet,
              'EMOGI': EMOGINet,
              'GAT': GATNet,
              'GCN': GCNNet,
              'CGMega': CGMega,
              'JKNet': JKNet,
              'PMLP': PMLP,
              'AGNN': AGNN,
              'GATv2': GATv2,
              'Chebnet': ChebNet,
              'Arma': Arma,
              'TAGNN': TAGCN
             }
data = load_net_specific_data(args)
features = data.x.numpy()

if args.model == 'CGMap':
    edge_file = './OPP_info/hop_edge_index_' + args.dataset + '_' + str(args.OPP_layer)
    if(path.exists(edge_file) == False):
        OPP_edge_info(data, args)
    hop_edge_index = torch.load('./OPP_info/hop_edge_index_' + args.dataset + '_' + str(args.OPP_layer))
    hop_edge_att = torch.load('./OPP_info/hop_edge_att_' + args.dataset + '_' + str(args.OPP_layer))
data = data.to(device)


def save_predicted_driver_genes(model, data, threshold=0.5, output_file=f'{args.dataset}_{args.model}.csv'):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(data)
        pred_prob = torch.sigmoid(output).cpu().numpy()  # Apply sigmoid to get probabilities
        gene_names = data.node_names
        predicted_labels = (pred_prob > threshold).astype(int)
        results = list(zip(gene_names, predicted_labels, pred_prob))
        df = pd.DataFrame(results, columns=['Gene', 'Predicted Label', 'Probability'])
        df_sorted = df.sort_values(by='Probability', ascending=False)
        df_sorted.to_csv(output_file, index=False)
        print(f"Predicted driver genes saved to {output_file}")

    return df


def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = torch.exp(-bce_loss)  # Probability of the correct class
    focal_loss = alpha * (1 - p_t) ** gamma * bce_loss
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


@torch.no_grad()
def test(data,model_name,mask):
    model.eval()
    if model_name == 'MTGCN':
        x, _, _, _ = model(data)

    elif model_name == 'PMLP':
        x = model(data)
    else:
        x = model(data)
    pred = torch.sigmoid(x[mask])
    precision, recall, _thresholds = metrics.precision_recall_curve(data.y[mask].cpu().numpy(),
                                                                    pred.cpu().detach().numpy())
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy()), area


AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))
pred_labels_all = []
for i in range(10):
    for cv_run in range(5):
        tr_mask, te_mask = data.mask[i][cv_run]
        model = model_dict[args.model](args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        for epoch in range(1, args.epochs + 1):
            # Training model
            model.train()
            optimizer.zero_grad()
            if args.model == 'MTGCN':
                pred, rl, c1, c2 = model(data)
                loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1)) / (c1 * c1) + rl / (
                            c2 * c2) + 2 * torch.log(c2 * c1)
            elif args.model == 'EMOGI':
                pred = model(data)
                loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1),
                                                          pos_weight=torch.tensor([45]).to(device))
            elif args.model == 'PMLP':
                pred = model(data)
                loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1))

            elif args.model == 'CGMap':
                for layer in args.layers:
                    hop_edge_index[layer - 1] = hop_edge_index[layer - 1].type(torch.LongTensor).to(device)
                    hop_edge_att[layer - 1] = hop_edge_att[layer - 1].to(device)
                args.hop_edge_index = hop_edge_index
                args.hop_edge_att = hop_edge_att
                pred = model(data)
                pos_weight = torch.tensor([data.y[tr_mask].sum() / (len(data.y[tr_mask]) - data.y[tr_mask].sum())]).to(device)
                bce_loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1), pos_weight=pos_weight)
                foc_loss = focal_loss(pred[tr_mask], data.y[tr_mask].view(-1, 1), alpha=0.45, gamma=6.0)
                loss = args.theta * bce_loss + (1-args.theta) * foc_loss
            else:
                pred = model(data)
                loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if args.model == 'MTGCN':
                pred, _, _, _ = model(data)
            else:
                pred = model(data)
        AUC[i][cv_run], AUPR[i][cv_run] = test(data, args.model, te_mask)
        print('Round--%d CV--%d  AUC: %.5f, AUPR: %.5f' % (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))
    print('Round--%d Mean AUC: %.5f, Mean AUPR: %.5f' % (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))
print('%s 10 rounds for 5CV-- Mean AUC: %.4f, Mean AUPR: %.4f' % (args.model, AUC.mean(), AUPR.mean()))
