import numpy as np
import math
import pandas as pd

sentence_store = {}
seen_summ = {}
sent_cnt = 0
base_path = ""
unseen_sent_id = 10000
relevant_count = 0

def string_id(s, val, is_summ):
    global sent_cnt, relevant_count
    s = s.strip()

    if is_summ:
        if not s in sentence_store:
            global unseen_sent_id
            relevant_count += 1 if val == 1 else 0
            use = unseen_sent_id
            unseen_sent_id += 1
            return use
        sent_id = sentence_store[s]
        if s in seen_summ:
            return False
        seen_summ[s] = True
        return sent_id

    if s in sentence_store:
        return False

    sentence_store[s] = sent_cnt
    relevant_count += 1 if val == 1 else 0
    sent_cnt += 1
    return sentence_store[s]

def get_csv(name, is_summ=False):
    use_path = base_path if is_summ else "QG"
    path = f"{use_path}-{name}.csv"

    df = pd.read_csv(path, header=None)    
    result = []

    for i in range(df.shape[0]):
        val = df.iloc[i, 2]
        val = 0 if math.isnan(val) else val
        val = int(val)
        sent_id = string_id(df.iloc[i, 0], val, is_summ)
        if sent_id == False:
            continue
        result.append((sent_id, val)) # float to int

    return np.array(result)

def get_csvs(name):
    global relevant_count, sentence_store, sent_cnt, seen_summ
    sent_cnt = 0
    seen_summ.clear()
    relevant_count = 0
    sentence_store.clear()
    unsumm = get_csv(f"{name}-U")
    summ = get_csv(f"{name}-S", True)
    return unsumm, summ


def print_metrics(name):
    unsumm, summ = get_csvs(name)
    relevant_retrievals = 0
    retrievals = 0

    for sent_id, val in summ:
        retrievals += 1
        if val == 1:
            relevant_retrievals += 1

    precision = relevant_retrievals / retrievals
    recall = relevant_retrievals / relevant_count

    # print(f"For dataset {name}\nPrecision: {precision}; recall: {recall}")
    print(f"{recall}", end=",")

files = ["BERT", "EXT", "W2V", "TF"]

for name in files:
    base_path = name
    print("FILE: " + name)
    print_metrics('NLP')
    print_metrics('COA')
