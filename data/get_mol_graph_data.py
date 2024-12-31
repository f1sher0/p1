import dill
from util import graph_batch_from_smile

def preprocess_and_save_graph_data_dill(atc4_mapping, save_path):
    """
    预处理 SMILES 数据并保存为 graph_data
    :param atc4_mapping: ATC4 到 SMILES 的映射
    :param save_path: 保存路径
    """
    graph_data_dict = {}

    for atc4, smiles_list in atc4_mapping['ATC4_to_SMILES'].items():
        graph_data = graph_batch_from_smile(smiles_list)
        graph_data_dict[atc4] = graph_data  # 每个 ATC4 存储其对应的图数据列表

    # 使用 dill 保存 graph_data 到文件
    with open(save_path, 'wb') as f:
        dill.dump(graph_data_dict, f)
    print(f"Graph data saved to {save_path}")

if __name__ == "__main__":
    with open('./output/ATC4_mappings.pkl', 'rb') as f:
        atc4_mapping = dill.load(f)
    preprocess_and_save_graph_data_dill(atc4_mapping, './output/graph_data.pkl')