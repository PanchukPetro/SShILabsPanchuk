import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Завантаження даних Iris
iris = load_iris()
X = iris.data

target = iris.target  # Істинні мітки класів для відображення
num_clusters = 3

kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=42)
kmeans.fit(X)

# Візуалізація вхідних даних (зменшення до 2D для графічного відображення)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
cluster_centers_reduced = pca.transform(kmeans.cluster_centers_)

# Візуалізація вхідних даних за справжніми класами
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], marker='o', facecolors = 'none', edgecolors='black', s=80)
plt.title("Вхідні дані")
plt.show()

step_size = 0.01

x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1

x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

output = kmeans.predict(pca.inverse_transform(np.c_[x_vals.ravel(), y_vals.ravel()]))

# Графічне відображення областей та виділення їх кольором
output = output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(),
                   y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired,
           aspect='auto',
           origin='lower')

# Відображення вхідних точок
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], marker='o', facecolors='none',
            edgecolors='black', s=80)

# Відображення центрів кластерів
plt.scatter(cluster_centers_reduced[:, 0], cluster_centers_reduced[:, 1], marker="o", s=210, linewidth=4, color='black', zorder=12, facecolors='black')

plt.title("Кордони кластерів")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
