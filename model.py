import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv


class PatientRepresentModel(nn.Module):
    def __init__(self, diag_voc, pro_voc, hidden_dim=256, device='cpu'):
        super(PatientRepresentModel, self).__init__()
        self.device = device
        self.embeddings = nn.ModuleList([
            nn.Embedding(diag_voc, hidden_dim),
            nn.Embedding(pro_voc, hidden_dim)
        ])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True) for _ in range(2)
        ])
        self.get_query = nn.Sequential(nn.ReLU(), nn.Linear(2 * hidden_dim, hidden_dim))
        self.init_weights()

    def forward(self, input):
        diag_seq = []
        pro_seq = []
        for adm in input:
            diag = adm[0]
            pro = adm[1]
            diag = torch.LongTensor(diag).unsqueeze(dim=0).to(self.device)
            pro = torch.LongTensor(pro).unsqueeze(dim=0).to(self.device)
            diag_embed = self.dropout(self.embeddings[0](diag))
            pro_embed = self.dropout(self.embeddings[1](pro))
            # 平均池化嵌入，用于每次就诊的表示
            diag_seq.append(diag_embed.sum(dim=1).unsqueeze(dim=0))
            pro_seq.append(pro_embed.sum(dim=1).unsqueeze(dim=0))

        diag_seq = torch.cat(diag_seq, dim=1)
        pro_seq = torch.cat(pro_seq, dim=1)

        # 使用 GRU 进行时序编码
        diag_encoded, _ = self.encoders[0](diag_seq)
        pro_encoded, _ = self.encoders[1](pro_seq)

        # 计算患者的表示，取最后时间步的隐状态
        patient_representations = torch.cat([diag_encoded, pro_encoded], dim=-1).squeeze(dim=0)
        query = self.get_query(patient_representations)[-1:, :]
        return query

    def init_weights(self):

        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


# 定义GNN用于药物嵌入
class MoleculeEncoder(nn.Module):
    def __init__(self, node_features=9, hidden_dim=256, output_dim=256, device='cpu'):
        super(MoleculeEncoder, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        ).to(device)
        # 第二层 GINConv
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.ReLU()
            )
        ).to(device)
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.ReLU()
            )
        ).to(device)
        self.device = device

    def forward(self, data):
        # 将图数据移动到设备上
        data.to(self.device)
        x, edge_index = data.x, data.edge_index

        # 第一层 GINConv
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 第二层 GINConv
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # 读出操作（全图表示）
        x = torch.sum(x, dim=0)  # 将节点嵌入进行求和，作为整个图的表示
        return x


class myModel(nn.Module):
    def __init__(self, diag_voc_size, pro_voc_size, med_voc_size, med_voc, graph_data_dict, node_features=5,
                 hidden_dim1=256, hidden_dim2=256,
                 mol_dim=256, device='cpu'):
        """
        初始化患者-药物匹配模型
        :param diag_voc_size: 诊断词表大小
        :param pro_voc_size: 手术词表大小
        :param node_features: 药物分子图节点特征维度
        :param hidden_dim1: 患者表示模块隐藏层维度
        :param hidden_dim2: 药物分子嵌入模块隐藏层纬度
        :param mol_dim: 药物分子嵌入维度
        :param atc4_mapping: ATC4 到 SMILES 的映射，用于生成药物嵌入
        """
        super(myModel, self).__init__()
        # 患者表示模块
        self.patient_model = PatientRepresentModel(diag_voc=diag_voc_size, pro_voc=pro_voc_size, hidden_dim=hidden_dim1,
                                                   device=device).to(device)

        # 药物分子表示模块
        self.molecule_encoder = MoleculeEncoder(node_features, hidden_dim2, mol_dim, device=device).to(device)

        # 融合模块：将患者表示与药物表示映射到相同的空间
        self.patient_projector = nn.Linear(hidden_dim1, mol_dim)  # 将患者表示映射到药物表示的维度
        self.atc4_projector = nn.Linear(mol_dim, mol_dim)  # 可选：进一步处理药物表示

        # LayerNorm
        self.patient_layernorm = nn.LayerNorm(mol_dim)  # 对患者表示进行规范化
        self.atc4_layernorm = nn.LayerNorm(mol_dim)  # 对药物嵌入进行规范化
        self.pred_layernorm = nn.LayerNorm(med_voc_size)

        self.med_voc = med_voc
        self.graph_data_dict = graph_data_dict
        self.output_dim = mol_dim
        self.device = device
        self.atc4_emb_matrix = self._build_atc4_embedding(graph_data_dict=self.graph_data_dict,
                                                          output_dim=self.output_dim).to(self.device)

    def _build_atc4_embedding(self, graph_data_dict, output_dim):
        """
        根据 ATC4 和 SMILES 映射生成药物嵌入
        """
        atc4_emb = {}
        for atc4, graph_data in graph_data_dict.items():
            graph_emb = self.molecule_encoder(graph_data)  # 使用 MoleculeEncoder 生成图嵌入
            atc4_emb[atc4] = graph_emb

        # 使用 med_voc.word2idx 的序号来生成矩阵
        vocab_size = len(self.med_voc.word2idx)
        atc4_matrix = torch.zeros((vocab_size, output_dim))  # 初始化矩阵，大小为 (词汇表大小, 嵌入维度)

        for atc4, idx in self.med_voc.word2idx.items():
            if atc4 in atc4_emb:
                atc4_matrix[idx] = atc4_emb[atc4]  # 将对应的 ATC4 嵌入放入矩阵的正确位置
        return nn.Parameter(atc4_matrix, requires_grad=True).to(self.device)

    def forward(self, input):

        # 患者嵌入表示
        patient_emb = self.patient_model(input).to(self.device)

        # 将患者表示映射到药物表示的维度
        patient_emb_proj = self.patient_projector(patient_emb).to(self.device)
        patient_emb_proj = self.patient_layernorm(patient_emb_proj)
        atc4_emb = self.atc4_projector(self.atc4_emb_matrix).to(self.device)
        atc4_emb = self.atc4_layernorm(atc4_emb)
        # 匹配分数计算
        predictions = torch.sigmoid(self.pred_layernorm(torch.mm(patient_emb_proj, atc4_emb.t())))
        return predictions
