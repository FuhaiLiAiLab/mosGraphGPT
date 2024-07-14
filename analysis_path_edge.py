import os
import torch
import numpy as np
import pandas as pd

def generate_edge_type(row, node_mapping):
    from_node = row['From']
    to_node = row['To']
    from_type = node_mapping.get(from_node, 'Unknown')
    to_type = node_mapping.get(to_node, 'Unknown')
    return f"{from_type}-{to_type}"

class PathNetAnalyse():
    def __init__(self):
        pass

    def prepare_network(self, dataset):
        ### Get [node_num_dict] for whole net nodes
        graph_output_folder = dataset + '-graph-data'
        gene_num_dict_df = pd.read_csv('./' + graph_output_folder + '/map-all-gene.csv')
        gene_num_dict_df = gene_num_dict_df.rename(columns={'Gene_name': 'node_name', 'Gene_num': 'node_num', 'NodeType': 'node_type'})
        node_num_dict_df = gene_num_dict_df.copy()
        node_num_dict_df = node_num_dict_df[['node_num', 'node_name', 'node_type']]
        node_num_dict_df.to_csv('./' + dataset + '-analysis/node_num_dict.csv', index=False, header=True)

    
    def average_edge(self, dataset, k=5):
        ### ['avg' is averaged weight over k folds]
        if os.path.exists('./' + dataset + '-analysis/avg') == False:
            os.mkdir('./' + dataset + '-analysis/avg')
        node_dict_file = './ROSMAP-analysis/node_num_dict.csv'
        node_dict = pd.read_csv(node_dict_file)
        node_mapping = dict(zip(node_dict['node_num'], node_dict['node_type']))
        # patient list
        survival_label_map_df = pd.read_csv('./' + dataset + '-graph-data/survival_label_map_dict.csv')
        survival_label_num = survival_label_map_df.shape[0]
        for survival_num in range(1, survival_label_num + 1):
            fold_survival_edge_df_list = []
            for fold_num in range(1, k + 1):
                fold_survival_edge_df = pd.read_csv('./' + dataset + '-analysis/fold_' + str(fold_num) + '/survival' + str(survival_num) +'.csv')
                fold_survival_edge_df_list.append(fold_survival_edge_df)

            fold_survival_group_df = pd.concat(fold_survival_edge_df_list)
            fold_survival_group_df = fold_survival_group_df.apply(pd.to_numeric, errors='coerce')
            # import pdb; pdb.set_trace()
            fold_survival_group_df = pd.concat(fold_survival_edge_df_list).groupby(level=0).mean()
            # convert column ['From', 'To'] to int
            fold_survival_group_df[['From', 'To']] = fold_survival_group_df[['From', 'To']].astype(int)
            # add edge type
            fold_survival_group_df['EdgeType'] = fold_survival_group_df.apply(generate_edge_type, axis=1, args=(node_mapping,))
            fold_survival_group_df.to_csv('./' + dataset + '-analysis/avg/survival' + str(survival_num) +'.csv', index=False, header=True)


### DATASET SELECTION
dataset = 'ROSMAP'
PathNetAnalyse().prepare_network(dataset)
PathNetAnalyse().average_edge(dataset)