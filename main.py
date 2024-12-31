from util import PatientDataset, collate_fn, get_atc4_emb
import dill, random, torch
from torch.utils.data import DataLoader
from model import myModel
import numpy as np
from sklearn.metrics import jaccard_score  # 用于计算 Jaccard 相似度

# =========================================
# setup
# 设置随机种子
random.seed(42)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
node_features = 5
hidden_dim1 = 256
hidden_dim2 = 128
mol_dim = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===========================================
# load source file
data_path = './data/output/records.pkl'
voc_path = './data/output/voc.pkl'
mapping_path = './data/output/ATC4_mappings.pkl'
graph_data_path = './data/output/graph_data.pkl'

with open(voc_path, 'rb') as f:
    voc = dill.load(f)
med_voc = voc['med_voc']
diag_voc = voc['diag_voc']
pro_voc = voc['pro_voc']
med_voc_size = len(med_voc.word2idx)

with open(data_path, 'rb') as f:
    data = dill.load(f)

with open(mapping_path, 'rb') as f:
    mapping = dill.load(f)

with open(graph_data_path, 'rb') as f:
    graph_data_dict = dill.load(f)
# ===========================================
# get data
# 总数据量
num_records = len(data)
# 计算各部分大小
train_size = int(train_ratio * num_records)
val_size = int(val_ratio * num_records)
test_size = num_records - train_size - val_size
# 随机打乱数据（确保不同运行的结果一致）
indices = list(range(num_records))
random.shuffle(indices)
# 根据索引划分数据
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]
# 生成对应的数据集
train_records = [data[i] for i in train_indices]
val_records = [data[i] for i in val_indices]
test_records = [data[i] for i in test_indices]
print(f"训练集大小: {len(train_records)}, 验证集大小: {len(val_records)}, 测试集大小: {len(test_records)}")
# 创建对应的 Dataset 对象
train_dataset = PatientDataset(train_records)
val_dataset = PatientDataset(val_records)
test_dataset = PatientDataset(test_records)
# 配置 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ==========================================
# model
model = myModel(
    diag_voc_size=len(diag_voc.word2idx),
    pro_voc_size=len(pro_voc.word2idx),
    med_voc_size=med_voc_size,
    med_voc=med_voc,
    graph_data_dict=graph_data_dict,
    node_features=node_features,
    hidden_dim1=hidden_dim1,
    hidden_dim2=hidden_dim2,
    mol_dim=mol_dim,
    device=device
).to(device)

# ========================================
# train
# 使用 BCEWithLogitsLoss 作为损失函数
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    jaccard_total = 0.0  # 累计 Jaccard 相似度
    total_samples = 0  # 样本总数

    for step, batch in enumerate(train_loader):
        print(step)
        optimizer.zero_grad()
        batch_loss = 0.0  # 当前 batch 的总损失
        batch_jaccard = 0.0  # 当前 batch 的 Jaccard 相似度

        for batch_step, patient_data in enumerate(batch):
            for idx, adm in enumerate(patient_data):
                input = patient_data[ : idx+1]
                pred = model(input).squeeze(0)
                pred_probs = torch.sigmoid(pred).detach().cpu().numpy()
                loss_bce_target = np.zeros((1, med_voc_size))
                med = adm[2]
                loss_bce_target[:, med] = 1
                loss_bce_target = torch.from_numpy(loss_bce_target).float().squeeze(0).to(device)
                batch_loss += criterion(pred, loss_bce_target)
                # 计算 Jaccard 相似度
                # 将 logits 转为二值预测
                pred_labels = (pred_probs >= 0.5).astype(int)  # 阈值为 0.5，转为二值
                true_labels = loss_bce_target.cpu().numpy()  # 真实标签转为 numpy
                batch_jaccard += jaccard_score(true_labels, pred_labels)  # 计算 Jaccard 相似度
                total_samples += 1
        train_loss += batch_loss.item()
        jaccard_total += batch_jaccard  # 累加 Jaccard 相似度
        batch_loss.backward()
        optimizer.step()
        print(f"Jaccard: {jaccard_total / total_samples:.4f}")
        # 打印每个 epoch 的平均损失和 Jaccard 相似度
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader):.4f}, "
          f"Jaccard: {jaccard_total / total_samples:.4f}")
