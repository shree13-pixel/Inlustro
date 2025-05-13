
import faiss
import numpy as np

def build_faiss_index(features):
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features)
    return index

def find_matches(index, query_feature, image_paths, threshold=0.7):
    D, I = index.search(np.array([query_feature]), k=1)
    distance = D[0][0]
    if distance < threshold:
        return image_paths[I[0][0]], distance
    else:
        return None, distance
