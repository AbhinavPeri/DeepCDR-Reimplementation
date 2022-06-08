import os

import numpy as np
import csv
import hickle as hkl
import pandas as pd
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

from util.config import NeededFiles
from util.utils import normalized_adj


class DeepCDRDataModule(pl.LightningDataModule):

    def __init__(self, raw_files: NeededFiles, max_atoms: int, batch_size=32):
        super().__init__()
        self.train_dataset, self.val_dataset, self.test_dataset = [None, None, None]
        self.raw_files = raw_files
        self.max_atoms = max_atoms
        self.batch_size = batch_size
        self.n_batches = None
        self.full_dataset = None
        self.n_val = None
        self.n_test = None

    def prepare_data(self) -> None:
        self.full_dataset = DeepCDRDataset(self.raw_files, self.max_atoms)
        self.n_batches = len(self.full_dataset) // self.batch_size
        assert self.n_batches >= 3
        self.n_val = self.batch_size * (1 + (15 * (self.n_batches - 3) // 100))
        self.test_dataset = self.full_dataset[:self.n_val]
        self.val_dataset = self.full_dataset[self.n_val:2 * self.n_val]
        self.train_dataset = self.full_dataset[2 * self.n_val:]

    def get_debug_batch(self):
        self.prepare_data()
        return next(iter(self.train_dataloader()))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=16, drop_last=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=16, drop_last=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=16, drop_last=True)


class DeepCDRDataset(Dataset):

    def __init__(self, raw_files: NeededFiles, max_atoms: int, generate_data=False):
        super().__init__()
        self.data = None
        self.raw_files = raw_files
        self.max_atoms = max_atoms
        if not os.listdir(path='data/preprocessed') or generate_data:
            self.process()
        else:
            self.data = hkl.load(
                'data/preprocessed/preloaded_data.hkl')

    def process(self):
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = self.metadata_generate(False)
        # Extract features for training and test
        drug_data, mutation_data, gexpr_data, methylation_data, ic50_scores = self.feature_extract(data_idx[:100], drug_feature, mutation_feature, gexpr_feature, methylation_feature)
        self.data = []
        for i in range(len(drug_data)):
            self.data.append([drug_data[i], mutation_data[i], gexpr_data[i], methylation_data[i], ic50_scores[i]])
        hkl.dump(self.data, open('data/preprocessed/preloaded_data.hkl', 'w'))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def metadata_generate(self, filtered):
        # drug_id --> pubchem_id
        reader = csv.reader(open(self.raw_files.drg_info, 'r'))
        rows = [item for item in reader]
        drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

        # map cellline --> cancer type
        cellline2cancertype = {}
        for line in open(self.raw_files.cell_ln_info).readlines()[1:]:
            cellline_id = line.split('\t')[1]
            TCGA_label = line.strip().split('\t')[-1]
            # if TCGA_label in TCGA_label_set:
            cellline2cancertype[cellline_id] = TCGA_label

        # load demap cell lines genomic mutation features
        mutation_feature = pd.read_csv(self.raw_files.gene_mutation, sep=',', header=0, index_col=[0])
        cell_line_id_set = list(mutation_feature.index)

        # load drug features
        drug_pubchem_id_set = []
        drug_feature = {}
        for each in os.listdir(self.raw_files.drg_features):
            drug_pubchem_id_set.append(each.split('.')[0])
            feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (self.raw_files.drg_features, each))
            drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
        assert len(drug_pubchem_id_set) == len(drug_feature.values())

        # load gene expression faetures
        gexpr_feature = pd.read_csv(self.raw_files.gene_exp, sep=',', header=0, index_col=[0])

        # only keep overlapped cell lines
        mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]

        # load methylation
        methylation_feature = pd.read_csv(self.raw_files.meth, sep=',', header=0, index_col=[0])
        assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
        experiment_data = pd.read_csv(self.raw_files.cancer_resp, sep=',', header=0, index_col=[0])
        # filter experiment raw
        drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
        experiment_data_filtered = experiment_data.loc[drug_match_list]

        data_idx = []
        for each_drug in experiment_data_filtered.index:
            for each_cellline in experiment_data_filtered.columns:
                pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
                if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                    if not np.isnan(experiment_data_filtered.loc[
                                        each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                        ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                        data_idx.append((each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))
        nb_celllines = len(set([item[0] for item in data_idx]))
        nb_drugs = len(set([item[1] for item in data_idx]))
        print(
            '%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), nb_celllines, nb_drugs))
        return mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx

    def calculate_graph_feat(self, feat_mat, adj_list):
        assert feat_mat.shape[0] == len(adj_list)
        feat = np.zeros((self.max_atoms, feat_mat.shape[-1]), dtype='float32')
        adj_mat = np.zeros((self.max_atoms, self.max_atoms), dtype='float32')
        feat[:feat_mat.shape[0], :] = feat_mat
        for i in range(len(adj_list)):
            nodes = adj_list[i]
            for each in nodes:
                adj_mat[i, int(each)] = 1
        assert np.allclose(adj_mat, adj_mat.T)
        adj_ = adj_mat[:len(adj_list), :len(adj_list)]
        adj_2 = adj_mat[len(adj_list):, len(adj_list):]
        norm_adj_ = normalized_adj(adj_)
        norm_adj_2 = normalized_adj(adj_2)
        adj_mat[:len(adj_list), :len(adj_list)] = norm_adj_
        adj_mat[len(adj_list):, len(adj_list):] = norm_adj_2
        return Data(x=torch.Tensor(feat), edge_index=torch.Tensor(adj_mat).to_sparse_coo().indices())

    def feature_extract(self, data_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature):
        nb_instance = len(data_idx)
        nb_mutation_feature = mutation_feature.shape[1]
        nb_gexpr_features = gexpr_feature.shape[1]
        nb_methylation_features = methylation_feature.shape[1]
        drug_data = [[] for item in range(nb_instance)]
        mutation_data = np.zeros((nb_instance, 1, nb_mutation_feature, 1), dtype='float32')
        gexpr_data = np.zeros((nb_instance, nb_gexpr_features), dtype='float32')
        methylation_data = np.zeros((nb_instance, nb_methylation_features), dtype='float32')
        target = np.zeros(nb_instance, dtype='float32')
        for idx in range(nb_instance):
            cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]
            # modify
            feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]
            # fill drug raw,padding to the same size with zeros
            drug_data[idx] = self.calculate_graph_feat(feat_mat, adj_list)
            # randomlize X A
            mutation_data[idx, 0, :, 0] = mutation_feature.loc[cell_line_id].values
            gexpr_data[idx, :] = gexpr_feature.loc[cell_line_id].values
            methylation_data[idx, :] = methylation_feature.loc[cell_line_id].values
            target[idx] = ln_IC50
        return drug_data, torch.squeeze(torch.Tensor(mutation_data), -1), torch.Tensor(gexpr_data), torch.Tensor(
            methylation_data), torch.Tensor(target)
