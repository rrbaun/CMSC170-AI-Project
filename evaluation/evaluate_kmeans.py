import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Ensure the figures directory exists
os.makedirs("figures", exist_ok=True)

# 2. K-Means Silhouette vs K plot data
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]
sil_scores = [0.9495, 0.8626, 0.7979, 0.7076, 0.7192, 0.6804, 0.6840, 0.5564, 0.6916]

plt.figure(figsize=(8, 4.5))
plt.plot(k_vals, sil_scores, marker='o', linewidth=2.2, color='#1b4f72', label='Silhouette Score')
plt.axvline(x=6, color='#c0392b', linestyle='--', linewidth=1.5, label='Selected K = 6 (Score = 0.7192)')
plt.scatter([6], [0.7192], color='#c0392b', zorder=5, s=70)

# 3. Add silhouette score text labels on top of every data point
for k, s in zip(k_vals, sil_scores):
    plt.annotate(f"{s:.4f}", (k, s), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8.5)

plt.title("K-Means Clustering: Silhouette Score vs. Number of Clusters (K)", fontsize=12, fontweight='bold')
plt.xlabel("Number of Clusters (K)", fontsize=10.5)
plt.ylabel("Mean Silhouette Score", fontsize=10.5)
plt.xticks(k_vals)
plt.ylim(0.50, 1.04)  # Gives a little breathing room at the top for labels like K=2
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save directly to the figures folder
plt.savefig("figures/kmeans_silhouette_vs_k.png", dpi=300)
plt.close()

print("Figure saved successfully to figures/kmeans_silhouette_vs_k.png!")