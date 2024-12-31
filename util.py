from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import Dataset
import torch
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


# 将 SMILES 转换为图数据对象
def mol_to_graph_data_obj(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 生成节点特征（原子特性）
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),  # 原子编号
            atom.GetDegree(),  # 原子连接的键数
            atom.GetFormalCharge(),  # 形式电荷
            int(atom.GetHybridization()),  # 杂化类型
            int(atom.GetIsAromatic())  # 是否芳香性
        ])
    x = torch.tensor(atom_features, dtype=torch.float)

    # 生成边特征（键特性）
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # 无向图
        edge_attr.append([
            bond.GetBondTypeAsDouble(),  # 键类型（单键、双键等）
            bond.GetIsConjugated(),  # 是否共轭键
            bond.IsInRing()  # 是否环结构
        ])
        edge_attr.append([
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.IsInRing()
        ])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_atc4_emb(mapping, mol_encoder, output_dim):
    # 用于存储 ATC4 的嵌入表示
    atc4_embeddings = {}
    for atc4, smiles_list in mapping['ATC4_to_SMILES'].items():
        # 存储该 ATC4 下所有 SMILES 的嵌入
        smile_embeddings = []
        for smile in smiles_list:
            try:
                # 将 SMILES 转换为图数据对象
                graph_data = mol_to_graph_data_obj(smile)
                # 检查图的有效性
                if graph_data.edge_index is None or graph_data.edge_index.numel() == 0:
                    print(f"Skipping SMILES with no edges: {smile}")
                    continue
                if graph_data.x is None or graph_data.x.size(0) == 0:
                    print(f"Invalid graph (no nodes) for SMILES: {smile}")
                    continue
                # 使用模型提取分子嵌入
                embedding = mol_encoder(graph_data)
                smile_embeddings.append(embedding)
            except ValueError as e:
                print(f"Invalid SMILES for {atc4}: {smile}, Error: {e}")
                continue  # 跳过无效的 SMILES
            except Exception as e:
                print(f"Unexpected error for SMILES {smile} in ATC4 {atc4}: {e}")
                continue
        # 如果有合法的 SMILES，则计算该 ATC4 的整体表示（取平均值）
        if smile_embeddings:
            atc4_embedding = torch.stack(smile_embeddings).mean(dim=0)  # 平均嵌入
            atc4_embeddings[atc4] = atc4_embedding
        else:
            print(f"No valid SMILES found for ATC4: {atc4}, assigning zero vector.")
            atc4_embeddings[atc4] = torch.zeros(output_dim)  # 默认零向量
    return atc4_embeddings


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
