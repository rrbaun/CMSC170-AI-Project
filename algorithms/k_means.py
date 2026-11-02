import math
from tabulate import tabulate

class TabularKMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k                      
        self.max_iters = max_iters      
        self.centroids = None           
        self.cluster_assignments = None 

    def fit(self, X):
        n_samples = len(X)
        if n_samples < self.k:
            raise ValueError("Number of samples is less than number of clusters k.")

        # Initial centroid selection: First k rows of X
        self.centroids = [list(X[i]) for i in range(self.k)]
        self.cluster_assignments = [0] * n_samples
        
        print(f"\nRunning K-Means with k={self.k} across {n_samples} records...")

        for iteration in range(1, self.max_iters + 1):
            new_assignments = []
            
            for x in X:
                distances = [self._euclidean_distance(x, centroid) for centroid in self.centroids]
                min_dist_idx = distances.index(min(distances))
                new_assignments.append(min_dist_idx)

            print(f"Iteration {iteration}: Updating assignments...")

            if new_assignments == self.cluster_assignments and iteration > 1:
                print(f"[K-Means] Converged successfully at iteration {iteration}!")
                break

            self.cluster_assignments = new_assignments

            new_centroids = []
            for c_idx in range(self.k):
                members = [X[i] for i in range(n_samples) if self.cluster_assignments[i] == c_idx]
                if members:
                    mean_vector = [sum(dim) / len(members) for dim in zip(*members)]
                    new_centroids.append(mean_vector)
                else:
                    new_centroids.append(self.centroids[c_idx])
                    
            self.centroids = new_centroids

        return self.cluster_assignments, self.centroids, iteration

    def compute_silhouette(self, X):
        n = len(X)
        if n <= self.k or self.k <= 1:
            return 0.0

        silhouette_scores = []
        for i in range(n):
            curr_cluster = self.cluster_assignments[i]
            same_cluster = [idx for idx, c in enumerate(self.cluster_assignments) if c == curr_cluster and idx != i]

            if not same_cluster:
                silhouette_scores.append(0.0)
                continue

            a_i = sum(self._euclidean_distance(X[i], X[idx]) for idx in same_cluster) / len(same_cluster)

            b_i = float('inf')
            for c_idx in range(self.k):
                if c_idx == curr_cluster:
                    continue
                other_cluster = [idx for idx, c in enumerate(self.cluster_assignments) if c == c_idx]
                if other_cluster:
                    avg_dist = sum(self._euclidean_distance(X[i], X[idx]) for idx in other_cluster) / len(other_cluster)
                    b_i = min(b_i, avg_dist)

            if b_i == float('inf'):
                s_i = 0.0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

            silhouette_scores.append(s_i)

        return sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else 0.0

    def _euclidean_distance(self, v1, v2):
        sq_sum = 0.0
        for a, b in zip(v1, v2):
            diff = a - b
            sq_sum += diff * diff
        return math.sqrt(sq_sum)


def run_kmeans(X, feature_columns, regions, causes):
    """
    Handles user interaction, execution, silhouette computation, and reporting for K-Means.
    """
    print("\n" + "="*40)
    print("RUNNING METHOD: K-MEANS CLUSTERING")
    print("="*40)

    # 1. Ask user for k
    k_input = input("Enter the number of clusters, k: ")
    try:
        k = int(k_input)
    except ValueError:
        k = 3
        print(f"Invalid input. Defaulting to k={k}")

    # 2. Run K-Means
    kmeans = TabularKMeans(k=k, max_iters=100)
    assignments, centroids, iterations = kmeans.fit(X)

    # 3. Compute Silhouette Score
    print("Computing silhouette score...")
    silhouette_score = kmeans.compute_silhouette(X)

    # 4. Generate Output Report & Geographic Breakdown
    cluster_table_data = []
    for c_idx in range(k): 
        member_indices = [i for i, c in enumerate(assignments) if c == c_idx]
        if len(member_indices) < 100:  # Focus on small/outlier clusters
            print(f"\nCluster {c_idx} (Size: {len(member_indices)}):")
            for idx in member_indices[:5]: # Show first 5 examples
                print(f"  - Region: {regions[idx]} | Cause: {causes[idx]}")

        member_count = assignments.count(c_idx)
        centroid_vec = centroids[c_idx]
        
        feature_scores = list(zip(feature_columns, centroid_vec))
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        top_5 = ", ".join([feat for feat, score in sorted_features[:5]])

        cluster_table_data.append([c_idx, member_count, top_5])

    print(f"\nIteration stages until convergence: {iterations}")
    print("\nCluster Details:")
    print(tabulate(cluster_table_data, headers=["Cluster Label", "Count of Records", "Top 5 Features (Age/Sex Bracket)"], tablefmt="plain"))
    print(f"\nOverall Silhouette score: {silhouette_score:.4f}")