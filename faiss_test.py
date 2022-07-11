import numpy as np
import faiss                   # 使Faiss可调用
import pickle

name = "0710_50"
print("loading...")
xb_entity = pickle.load(open(f"src/fassi_entity_time_emb_label_{name}.pkl", "rb"))
xb_rel = pickle.load(open(f"src/fassi_rel_time_emb_label_{name}.pkl", "rb"))
print("done.")
dim, measure = 400, faiss.METRIC_L2
param = 'HNSW64'
index_entity = faiss.index_factory(dim, param, measure)
index_rel = faiss.index_factory(dim, param, measure)
print("begin adding...")
index_entity.add(xb_entity[:, 1:-1].copy())
index_rel.add(xb_rel[:, 1:-1].copy())
# D, I = index_entity.search(xb_entity[:5], 4)
# D, I = index_rel.search(xb_rel[:5], 4)
print("done.")
faiss.write_index(index_entity, f"entity.index_{name}")
faiss.write_index(index_rel, f"rel.index_{name}")
# index = faiss.read_index("large.index")