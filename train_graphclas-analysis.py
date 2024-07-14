import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.autograd import Variable
from torch.utils.data import DataLoader

# custom modules
from maskgae.utils import set_seed, tab_printer, get_dataset
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from maskgae.mask import MaskEdge, MaskPath

# custom dataloader
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from enc_dec.geo_pretrain_gformer_decoder_analysis import GraphFormerDecoder


def build_pretrain_model(args, num_feature, num_node, device):
    if args.mask == 'Path':
        mask = MaskPath(p=args.p, 
                        num_nodes=num_node, 
                        start=args.start,
                        walk_length=args.encoder_layers+1)
    elif args.mask == 'Edge':
        mask = MaskEdge(p=args.p)
    else:
        mask = None # vanilla GAE
    
    encoder = GNNEncoder(num_feature, args.encoder_channels, args.hidden_channels,
                        num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                        bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    internal_encoder = GNNEncoder(num_feature, args.input_dim, args.input_dim,
                            num_layers=args.internal_encoder_layers, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                            num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    pretrain_model = MaskGAE(input_dim=args.input_dim, 
                    num_node=num_node,
                    encoder=encoder, 
                    internal_encoder=internal_encoder,
                    edge_decoder=edge_decoder, 
                    degree_decoder=degree_decoder, 
                    mask=mask).to(device)
    return pretrain_model


def pretrain_linkpred(pretrain_model, splits, args, device='cpu'):
    print('Start Training (Link Prediction Pretext Training)...')
    optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    test_data = splits['test'].to(device)
    
    pretrain_model.reset_parameters()

    loss = pretrain_model.train_step(train_data, optimizer,
                                alpha=args.alpha, 
                                batch_size=args.batch_size)
    torch.save(pretrain_model.state_dict(), args.save_path)

    test_auc, test_ap = pretrain_model.test_step(test_data, 
                                        test_data.pos_edge_label_index, 
                                        test_data.neg_edge_label_index, 
                                        batch_size=batch_size)   
    
    print(f'Link Prediction Pretraining Results:\n'
          f'AUC: {test_auc:.2%}',
          f'AP: {test_ap:.2%}')
    return test_auc, test_ap


def pretrain_foundation(args, device, num_feature=8):
    # Dataset Selection
    dataset = args.dataset
    graph_output_folder = dataset + '-graph-data'
    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list)

    # Build Pretrain Model
    pretrain_model = build_pretrain_model(args, num_feature, num_node, device)

    if args.pretrain==1:
        # Training dataset basic parameters
        # [num_feature, num_node]
        num_feature = 8
        final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
        gene_name_list = list(final_annotation_gene_df['Gene_name'])
        num_node = len(gene_name_list)
        form_data_path = './' + graph_output_folder + '/form_data'
        # Read these feature label files
        print('--- LOADING TRAINING FILES ... ---')
        xAll = np.load(form_data_path + '/xAll.npy')
        yAll = np.load(form_data_path + '/yAll.npy')
        all_edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long()
        internal_edge_index = torch.from_numpy(np.load(form_data_path + '/internal_edge_index.npy') ).long()
        ppi_edge_index = torch.from_numpy(np.load(form_data_path + '/ppi_edge_index.npy') ).long()
        upper_index = 0
        dl_input_num = xAll.shape[0]
        batch_size = args.pretain_batch_size

        for index in range(0, dl_input_num, batch_size):
            if (index + batch_size) < dl_input_num:
                upper_index = index + batch_size
            else:
                upper_index = dl_input_num
            geo_datalist = read_batch(index, dl_input_num, xAll, yAll, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index, graph_output_folder)
            dataset_loader, node_num, feature_dim = GeoGraphLoader.load_graph(geo_datalist, args) # read by batch size
            for batch_idx, data in enumerate(dataset_loader):
                train_data, val_data, test_data = T.RandomLinkSplit(num_test=0.1, num_val=0.0,
                                                                is_undirected=False,
                                                                split_labels=True,
                                                                add_negative_train_samples=False)(data)
                if args.full_data:
                # Use full graph for pretraining
                    splits = dict(train=data, test=test_data)
                else:
                    splits = dict(train=train_data, test=test_data)
                print(f'Starting {index} - {upper_index}')
                pretrain_linkpred(pretrain_model, splits, args, device=device)
                print(f'Pretraining {upper_index} done!')
    else:
        pretrain_model.load_state_dict(torch.load(args.save_path))
    return pretrain_model


def analyze_graphclas_model(analysis_dataset_loader, batch_random_final_dl_input_df, pretrain_model, model, device, fold_n, dataset):
    batch_loss = 0
    all_ypred = np.zeros((1, 1))
    for batch_idx, data in enumerate(analysis_dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
        # Use pretrained model to get the embedding
        z = pretrain_model.internal_encoder(x, internal_edge_index)
        embedding = pretrain_model.encoder.get_embedding(z, ppi_edge_index, mode='last') # mode='cat'
        output, ypred = model(x, embedding, edge_index, batch_random_final_dl_input_df, fold_n, dataset)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        all_ypred = np.vstack((all_ypred, ypred.cpu().numpy().reshape(-1, 1)))
        all_ypred = np.delete(all_ypred, 0, axis=0)
    return model, batch_loss, batch_acc, all_ypred


def build_graphclas_model(args, num_node, device):
    model = GraphFormerDecoder(pretrain_input_dim = args.pretrain_input_dim,
                               pretrain_output_dim = args.pretrain_output_dim,
                               input_dim=args.train_input_dim, 
                               hidden_dim=args.train_hidden_dim, 
                               embedding_dim=args.train_embedding_dim, 
                               num_node=num_node, 
                               num_head=1, device=device, num_class=2).to(device)
    return model


def analyze_model(args, fold_n, dataset, pretrain_model, model, device, graph_output_folder, i):
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    # Test model on test dataset
    form_data_path = './' + graph_output_folder + '/form_data'
    xAll = np.load(form_data_path + '/xAll.npy')
    yAll = np.load(form_data_path + '/yAll.npy')
    all_edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long()
    internal_edge_index = torch.from_numpy(np.load(form_data_path + '/internal_edge_index.npy') ).long()
    ppi_edge_index = torch.from_numpy(np.load(form_data_path + '/ppi_edge_index.npy') ).long()
    random_final_dl_input_df = pd.read_csv('./' + graph_output_folder + '/random-survival-label.csv')

    dl_input_num = xAll.shape[0]
    train_batch_size = args.train_batch_size
    # Clean result previous epoch_i_pred files
    # [num_feature, num_node]
    num_feature = 8
    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list)
    # Number of samples
    survival_label_list = sorted(list(set(random_final_dl_input_df['individualID'])))
    survival_label_num = [x for x in range(1, len(survival_label_list)+1)]
    survival_label_map_df = pd.DataFrame({'individualID': survival_label_list, 'individualID_Num': survival_label_num})
    survival_label_map_df.to_csv('./' + graph_output_folder + '/survival_label_map_dict.csv', index=False, header=True)
    batch_included_survival_label_list = []
    # Run test model
    model.eval()
    upper_index = 0
    batch_loss_list = []
    for index in range(0, dl_input_num, train_batch_size):
        if (index + train_batch_size) < dl_input_num:
            upper_index = index + train_batch_size
        else:
            upper_index = dl_input_num
        geo_datalist = read_batch(index, upper_index, xAll, yAll, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index, graph_output_folder)
        analysis_dataset_loader, node_num, feature_dim = GeoGraphLoader.load_graph(geo_datalist, args)
        batch_random_final_dl_input_df = random_final_dl_input_df.iloc[index : upper_index]
        print('ANALYZE MODEL...')
        model, batch_loss, batch_acc, batch_ypred = analyze_graphclas_model(analysis_dataset_loader, batch_random_final_dl_input_df, pretrain_model, model, device, fold_n, dataset)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        print('BATCH ACCURACY: ', batch_acc)
        tmp_batch_survival_label_list = sorted(list(set(batch_random_final_dl_input_df['individualID'])))
        batch_included_survival_label_list += tmp_batch_survival_label_list
        batch_included_survival_label_list = sorted(list(set(batch_included_survival_label_list)))
        # import pdb; pdb.set_trace()
        if batch_included_survival_label_list == survival_label_list:
            print(len(batch_included_survival_label_list))
            print(batch_included_survival_label_list)
            break



def arg_parse():
    parser = argparse.ArgumentParser()
    # pre-training parameters
    parser.add_argument('--dataset', nargs='?', default='UCSC', help='Datasets. (default: UCSC)')
    parser.add_argument('--mask', nargs='?', default='Path', help='Masking stractegy, `Path`, `Edge` or `None` (default: Path)')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')
    parser.add_argument('--pretrain', type=int, default=0, help='Whether to pretrain the model. (default: False)')

    parser.add_argument('--layer', nargs='?', default='gcn', help='GNN layer, (default: gcn)')
    parser.add_argument('--encoder_activation', nargs='?', default='elu', help='Activation function for GNN encoder, (default: elu)')

    parser.add_argument('--input_dim', type=int, default=8, help='Input feature dimension. (default: 8)')
    parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder layers. (default: 128)')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Channels of hidden representation. (default: 64)')
    parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 128)')
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--internal_encoder_layers', type=int, default=3, help='Number of layers for internal encoder. (default: 3)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for pre-training. (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size for link prediction training. (default: 2**16)')
    parser.add_argument('--num_workers', dest = 'num_workers', type = int, default=0, help = 'Number of workers to load data.')

    parser.add_argument('--start', nargs='?', default='node', help='Which Type to sample starting nodes for random walks, (default: node)')
    parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--graphclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=5, help='Number of pre-training epochs. (default: 5)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
    parser.add_argument('--eval_period', type=int, default=30, help='(default: 30)')
    parser.add_argument('--save_path', nargs='?', default='MaskGAE-GraphClas.pt', help='save path for model. (default: MaskGAE-GraphClas.pt)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--full_data', action='store_true', help='Whether to use full data for pretraining. (default: False)')

    # training parameters
    parser.add_argument('--fold_n', dest='fold_n', type=int, default=1, help='Fold number for training. (default: 1)')
    parser.add_argument('--train_dataset', dest='train_dataset', type=str, default='UCSC', help='Dataset for training. (default: UCSC)')
    parser.add_argument('--num_train_epoch', dest='num_train_epoch', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, default=32, help='Batch size of training.')
    parser.add_argument('--train_lr', dest='train_lr', type=float, default=0.005, help='Learning rate for training. (default: 0.005)')
    parser.add_argument('--train_input_dim', dest='train_input_dim', type=int, default=8, help='Input dimension of training. (default: 64)')
    parser.add_argument('--train_hidden_dim', dest='train_hidden_dim', type=int, default=30, help='Hidden dimension of training. (default: 64)')
    parser.add_argument('--train_embedding_dim', dest='train_embedding_dim', type=int, default=30, help='Embedding dimension of training. (default: 64)')
    parser.add_argument('--train_result_path', nargs='?', dest='train_result_path', default='pretrain+gformer', help='save path for model result. (default: pretrain+gformer)')
    parser.add_argument('--pretrain_input_dim', dest='pretrain_input_dim', type=int, default=64, help='Output dimension of training. (default: 64)')
    parser.add_argument('--pretrain_output_dim', dest='pretrain_output_dim', type=int, default=8, help='Output dimension of pretraining. (default: 64)')

    # test parameters by loading model (both pretrained and trained)
    parser.add_argument('--load', dest='load', type=int, default=0, help='Whether to load the model. (default: 0)')
    return parser.parse_args()


def analyze_trained_model(args, fold_n, dataset, pretrain_model, device):
    print('\n------------- LOAD MODEL AND ANALYZE -------------')
    dataset = args.train_dataset
    graph_output_folder = dataset + '-graph-data'
    final_annotation_gene_df = pd.read_csv(os.path.join(graph_output_folder, 'map-all-gene.csv'))
    gene_name_list = list(final_annotation_gene_df['Gene_name'])
    num_node = len(gene_name_list)
    model = build_graphclas_model(args, num_node, device)
    folder_name = 'epoch_' + str(args.num_train_epoch) + '_fold_' + str(fold_n)
    path = './' + dataset + '-result/' + args.train_result_path + '/%s' % (folder_name)
    model.load_state_dict(torch.load(path + '/best_train_model.pt'))
    analyze_model(args, fold_n, dataset, pretrain_model, model, device, graph_output_folder, i=0)


if __name__ == "__main__":
    # Set arguments and print
    args = arg_parse()
    print(tab_printer(args))
    # Check device
    if args.device < 0:
        device = 'cpu'
    else:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Pretrain model
    pretrain_model = pretrain_foundation(args, device, num_feature=8)

    k = 5
    dataset = args.train_dataset
    for fold_n in np.arange(1, k + 1):
        os.makedirs('./' + dataset + '-analysis/fold_' + str(fold_n), exist_ok=True)
        graph_output_folder = dataset + '-graph-data'
        yTr = np.load('./' + graph_output_folder + '/form_data/yTr' + str(fold_n) + '.npy')
        unique_numbers, occurrences = np.unique(yTr, return_counts=True)
        num_class = len(unique_numbers)
        print("num:" ,num_class)

        analyze_trained_model(args, fold_n, dataset, pretrain_model, device)
    