import pandas as pd
import dill


def med_process(med_file):
    """
    :param med_file: prescription med file from MIMIC
    :return: preliminary processed med file (drop unnecessary columns and duplicate rows)
    """
    med_pd = pd.read_csv(med_file, dtype={"NDC": "category"})
    med_pd.drop(
        columns=[
            "ROW_ID",
            "DRUG_TYPE",
            "DRUG_NAME_POE",
            "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD",
            "PROD_STRENGTH",
            "DOSE_VAL_RX",
            "DOSE_UNIT_RX",
            "FORM_VAL_DISP",
            "FORM_UNIT_DISP",
            "GSN",
            "FORM_UNIT_DISP",
            "ROUTE",
            "ENDDATE",
        ],
        axis=1,
        inplace=True,
    )
    med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)
    med_pd.fillna(method="pad", inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd["ICUSTAY_ID"] = med_pd["ICUSTAY_ID"].astype("int64")
    med_pd["STARTDATE"] = pd.to_datetime(
        med_pd["STARTDATE"], format="%Y-%m-%d %H:%M:%S"
    )
    med_pd.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"], inplace=True
    )
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=["ICUSTAY_ID"])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

def filter_100_most_med(med_pd):
    med_count = (
        med_pd.groupby(by=["ATC4"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    med_pd = med_pd[med_pd["ATC4"].isin(med_count.loc[:100, "ATC4"])]

    return med_pd.reset_index(drop=True)

def codeMapping2atc4(med_pd):
    """
    :param med_pd: preliminary processed med file with code in NDC format
    :return: med file with code in ATC4 format
    """
    with open(ndc2RXCUI_file, "r") as f:
        ndc2RXCUI = eval(f.read())
    med_pd["RXCUI"] = med_pd["NDC"].map(ndc2RXCUI)
    med_pd.dropna(inplace=True)

    RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
    RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
    RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)

    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
    med_pd.drop(columns=["NDC", "RXCUI"], inplace=True)
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def process_visit_lg2(med_pd):
    a = (
        med_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby(by="SUBJECT_ID")["HADM_ID"]
        .unique()
        .reset_index()
    )
    a["HADM_ID_Len"] = a["HADM_ID"].map(lambda x: len(x))
    a = a[a["HADM_ID_Len"] > 1]
    return a


def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["SEQ_NUM", "ROW_ID"], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = (
            diag_pd.groupby(by=["ICD9_CODE"])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
            .sort_values(by=["count"], ascending=False)
            .reset_index(drop=True)
        )
        diag_pd = diag_pd[diag_pd["ICD9_CODE"].isin(diag_count.loc[:1999, "ICD9_CODE"])]
        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)
    return diag_pd


def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={"ICD9_CODE": "category"})
    pro_pd.drop(columns=["ROW_ID"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], inplace=True)
    pro_pd.drop(columns=["SEQ_NUM"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def combine_process(med_pd, diag_pd, pro_pd):
    med_pd_key = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_pd_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    pro_pd_key = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()

    combined_key = med_pd_key.merge(
        diag_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
    )
    combined_key = combined_key.merge(
        pro_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
    )

    diag_pd = diag_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    med_pd = med_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    pro_pd = pro_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    # flatten and merge
    diag_pd = (
        diag_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"]
        .unique()
        .reset_index()
    )
    med_pd = med_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ATC4"].unique().reset_index()
    pro_pd = (
        pro_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"]
        .unique()
        .reset_index()
        .rename(columns={"ICD9_CODE": "PRO_CODE"})
    )
    med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: list(x))
    pro_pd["PRO_CODE"] = pro_pd["PRO_CODE"].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data = data.merge(pro_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data["ATC4_num"] = data["ATC4"].map(lambda x: len(x))

    return data


# get vocabulary
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def ATC_process(med_voc):
    # 获取所有 ATC4
    all_atc4 = set(med_voc.word2idx.keys())
    # 加载 ATC_CID.xlsx 和 DrugInfo.csv 文件
    atc_cid_df = pd.read_excel('./input/ATC_CID.xlsx')
    drug_info_df = pd.read_csv('./input/DrugInfo.csv')
    atc_cid_df['ATC4'] = atc_cid_df['ATC4'].str[:5]

    # 删除 ATC_CID 中 CID 为 -1 的条目
    filtered_atc_cid = atc_cid_df[(atc_cid_df['ATC4'].isin(all_atc4)) & (atc_cid_df['CID'] != -1)]
    # 1. 创建 ATC4 -> 对应的所有 ATC5 代码映射
    atc4_to_atc5_mapping = filtered_atc_cid.groupby('ATC4')['ATC5'].apply(list).to_dict()
    # 2. 创建 ATC4 -> 对应的所有 ATC5 的 SMILES 映射
    # 合并 ATC_CID 和 DrugInfo 表，按 CID 合并
    merged_df = filtered_atc_cid.merge(drug_info_df, on='CID', how='inner')
    # 创建映射：ATC4 -> 对应的 SMILES 列表
    atc4_to_smiles_mapping = merged_df.groupby('ATC4')['isosmiles'].apply(list).to_dict()

    # 获取两个映射的交集键
    valid_atc4_keys = set(atc4_to_atc5_mapping.keys()) & set(atc4_to_smiles_mapping.keys())
    # 过滤出交集键对应的映射
    filtered_atc4_to_atc5_mapping = {k: atc4_to_atc5_mapping[k] for k in valid_atc4_keys}
    filtered_atc4_to_smiles_mapping = {k: atc4_to_smiles_mapping[k] for k in valid_atc4_keys}

    # 打包映射为字典
    final_mapping = {
        'ATC4_to_ATC5': atc4_to_atc5_mapping,
        'ATC4_to_SMILES': atc4_to_smiles_mapping
    }
    # 保存结果到 pkl 文件
    with open('./output/ATC4_mappings.pkl', 'wb') as f:
        dill.dump(final_mapping, f)
    print("get ATC4 mapping done")
    return valid_atc4_keys


def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    for index, row in df.iterrows():
        diag_voc.add_sentence(row["ICD9_CODE"])
        med_voc.add_sentence(row["ATC4"])
        pro_voc.add_sentence(row["PRO_CODE"])
    valid_atc4_keys = ATC_process(med_voc)
    # 根据 valid_atc4_keys 过滤 med_voc
    med_voc = filter_voc_by_valid_keys(med_voc, valid_atc4_keys)
    dill.dump(
        obj={"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
        file=open(vocabulary_file, "wb"),
    )
    return diag_voc, med_voc, pro_voc, valid_atc4_keys


def filter_voc_by_valid_keys(voc, valid_keys):
    """
    根据有效的键过滤 voc 对象。
    :param voc: Voc 对象
    :param valid_keys: 有效的键集合
    :return: 过滤后的 Voc 对象
    """
    # 创建新的 voc
    filtered_voc = Voc()

    # 遍历 word2idx，筛选有效的键
    for word in voc.word2idx:
        if word in valid_keys:
            filtered_voc.add_sentence([word])

    return filtered_voc


# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = []  # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df["SUBJECT_ID"].unique():
        item_df = df[df["SUBJECT_ID"] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row["ICD9_CODE"]])
            admission.append([pro_voc.word2idx[i] for i in row["PRO_CODE"]])
            admission.append([med_voc.word2idx[i] for i in row["ATC4"]])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open(ehr_sequence_file, "wb"))
    return records


if __name__ == "__main__":
    # MIMIC dataset
    med_file = "./input/PRESCRIPTIONS.csv"
    diag_file = "./input/DIAGNOSES_ICD.csv"
    procedure_file = "./input/PROCEDURES_ICD.csv"

    # auxiliary files
    RXCUI2atc4_file = "./input/RXCUI2atc4.csv"
    ndc2RXCUI_file = "./input/ndc2RXCUI.txt"

    # output files
    vocabulary_file = "./output/voc.pkl"
    ehr_sequence_file = "./output/records.pkl"

    # process of med
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(
        med_pd_lg2[["SUBJECT_ID"]], on="SUBJECT_ID", how="inner"
    ).reset_index(drop=True)
    med_pd = codeMapping2atc4(med_pd)

    print("complete medication processing")

    # process of diagnosis
    diag_pd = diag_process(diag_file)
    print("complete diagnosis processing")

    # process procedure
    pro_pd = procedure_process(procedure_file)
    print("complete procedure processing")

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    print("complete combining")

    # create vocab
    diag_voc, med_voc, pro_voc, valid_atc4 = create_str_token_mapping(data)
    print("obtain voc")

    data = data[data["ATC4"].apply(lambda atc_list: all(atc in valid_atc4 for atc in atc_list))]

    # create ehr sequence data
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    print("obtain ehr sequence data")
