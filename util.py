import numpy as np
import torch
from ogb.utils import smiles2graph
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import Dataset
from torch_geometric.data import Data


def tanimoto_similarity(smile1, smile2):
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)
    morgan_fp_gen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=2)
    fp1 = morgan_fp_gen.GetFingerprint(mol1)
    fp2 = morgan_fp_gen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def cosine_similarity(embedding_a, embedding_b):
    return torch.nn.functional.cosine_similarity(embedding_a, embedding_b, dim=-1)


def map2atc3_label(word2idx, atc3_to_index, l1, l2):
    num_atc3 = len(atc3_to_index)
    l1_mapped = [0] * num_atc3
    l2_mapped = [0] * num_atc3

    for code, presence in zip(word2idx.keys(), l1):
        if presence == 1:
            atc3 = code[:4]
            l1_mapped[atc3_to_index[atc3]] = 1

    for code, presence in zip(word2idx.keys(), l2):
        if presence == 1:
            atc3 = code[:4]
            l2_mapped[atc3_to_index[atc3]] = 1

    return l1_mapped, l2_mapped

def graph_batch_from_smile(smiles_list):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    return Data(**result)


# 自定义 Dataset 类
class PatientDataset(Dataset):
    def __init__(self, records):
        """
        初始化数据集
        :param records: 患者记录，格式为 [[[诊断列表], [手术列表], [药物列表]], ...]
        """
        self.records = records

    def __len__(self):
        """返回数据集大小"""
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# 自定义 collate_fn
def collate_fn(batch):
    return batch
