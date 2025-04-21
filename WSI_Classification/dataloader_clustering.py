from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from typing import List, Tuple
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import torch 
from torch.utils.data import Dataset
import os
import numpy as np
import random

class WSIDataset(Dataset):
    def __init__(self, save_dir: str, fold_ids: List[str]):
        self.data = []
        self.save_dir = save_dir
        self.fold_ids = fold_ids
        self._load_data()

    def _load_data(self):
        for wsi_folder in os.listdir(self.save_dir):
            wsi_folder_path = os.path.join(self.save_dir, wsi_folder)
            if not os.path.isdir(wsi_folder_path) and len(os.listdir(wsi_folder_path)) <= 15:
                # print(f"Skipping {wsi_folder} due to less than 18 patches")
                continue
            for wsi_file in os.listdir(wsi_folder_path):
                if wsi_file.endswith('.pt'):
                    # wsi_id = wsi_folder
                    # for tcga-CV comment above and use below line
                    wsi_id = wsi_file[:12]
                    if wsi_id not in self.fold_ids:
                        continue
                    try:
                        wsi_features = torch.load(os.path.join(wsi_folder_path, wsi_file))
                        # check if loaded features is not one feature vector then average them to make one feature vector
                        if isinstance(wsi_features, torch.Tensor) and wsi_features.dim() > 1:
                            # print(f"WSI ID: {wsi_id} | Features Shape: {wsi_features.shape}")
                            wsi_features = torch.mean(wsi_features, dim=0)
                        label = 0 if '_nonMSI' in wsi_file else 1
                        self.data.append((wsi_features, label, wsi_id))
                    except Exception as e:
                        print(f"Error loading {os.path.join(wsi_folder_path, wsi_file)}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label, wsi_id = self.data[idx]
        return features, label, wsi_id
    
    def apply_clustering(self, clustering_algorithm: str, num_clusters: int = 3, num_selected_patches: int = 0):
        """
        Apply clustering on the patches of each WSI and create a consistent WSI representation.
        Args:
        - clustering_algorithm: The clustering algorithm to use ('kmeans', 'dbscan', 'pca').
        - num_clusters: Number of clusters to create (only for k-means or similar algorithms).
        - num_selected_patches: Number of top patches to use for averaging within each cluster (optional).
        """
        clustered_data = []
        wsi_ids = set([wsi_id for _, _, wsi_id in self.data])  # Unique WSI IDs

        for wsi_id in wsi_ids:
            # Extract all patches for the WSI
            wsi_patches = [features for features, _, id in self.data if id == wsi_id]
            wsi_patches = torch.stack(wsi_patches)
            patch_array = wsi_patches.cpu().numpy()
            # Step 0: Skip WSI if the number of patches is less than the specified number of clusters
            if len(patch_array) < num_clusters:
                # print(f"Skipping WSI {wsi_id} because it has fewer patches ({len(patch_array)}) than the specified number of clusters ({num_clusters}).")
                continue

            # Step 1: Perform clustering
            if clustering_algorithm == 'kmeans':
                clustering_model = KMeans(n_clusters=num_clusters, random_state=42)
                clustering_model.fit(patch_array)
                cluster_labels = clustering_model.labels_
                cluster_centroids = clustering_model.cluster_centers_
            elif clustering_algorithm == 'dbscan':
                clustering_model = DBSCAN(eps=0.1, min_samples=2)
                cluster_labels = clustering_model.fit_predict(patch_array)
                unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                num_clusters = unique_clusters
                print(f"Unique clusters: {unique_clusters}")
                cluster_centroids = np.array([np.mean(patch_array[cluster_labels == i], axis=0) for i in range(num_clusters)])
                
            elif clustering_algorithm == 'pca':
                pca = PCA(n_components=num_clusters)
                transformed_features = pca.fit_transform(patch_array)
                cluster_labels = np.argmax(transformed_features, axis=1)
                cluster_centroids = pca.components_
            else:
                raise ValueError(f"Unsupported clustering algorithm: {clustering_algorithm}")

            # Step 2: Aggregate features within each cluster
            selected_features = []
            cluster_sums = []
            for cluster_idx in range(num_clusters):
                cluster_patches = patch_array[cluster_labels == cluster_idx]
                if len(cluster_patches) == 0:
                    continue
                # Optionally select top-ranked patches
                if num_selected_patches > 0:
                    distances = cdist(cluster_patches, [cluster_centroids[cluster_idx]], metric='euclidean').flatten()
                    sorted_indices = np.argsort(distances)
                    cluster_patches = cluster_patches[sorted_indices[:num_selected_patches]]

                # Average the cluster features and calculate cluster sum for sorting
                cluster_average = np.mean(cluster_patches, axis=0)
                cluster_sum = np.average(cluster_average)
                selected_features.append(cluster_average)
                cluster_sums.append(cluster_sum)

            # Step 3: Sort clusters by their sum values
            sorted_indices = np.argsort(cluster_sums)
            sorted_features = np.array(selected_features)[sorted_indices]

            # Step 4: Create a consistent WSI-level representation
            concatenated_features = np.concatenate(sorted_features)
            label = [label for _, label, id in self.data if id == wsi_id][0]  # Assume all patches have the same label
            clustered_data.append((torch.tensor(concatenated_features), label, wsi_id))

            # Print debug information (optional)
            # print(f"WSI ID: {wsi_id} | Total Patches: {len(patch_array)} | Clusters: {num_clusters} | Sorted Features Shape: {concatenated_features.shape}")

        # Update the dataset with clustered data
        self.data = clustered_data