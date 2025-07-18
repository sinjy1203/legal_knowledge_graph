import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cluster_entities(entities, entity_vectors, distance_threshold=0.25):
    vectors_array = np.array(entity_vectors)
    
    # AgglomerativeClustering 수행
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine", 
        linkage="single", 
        distance_threshold=distance_threshold
    )
    
    # 클러스터 레이블 예측
    cluster_labels = clustering.fit_predict(vectors_array)
    
    # 클러스터별로 엔티티 그룹화
    unique_clusters = np.unique(cluster_labels)
    clusters = []
    
    for cluster_id in unique_clusters:
        cluster_entities = [entities[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        clusters.append(cluster_entities)
    
    return clusters
