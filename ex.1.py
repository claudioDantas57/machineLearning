import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = np.array([


    [-25,-46], #sp
    [-22,-43],#rj
    [-20,-40], #ctb
    [-30,-51], #poa
    [-19,-43], #bh
    [-15,-47], #bsb
    [-12,-38], #sal
    [-18,-36], #rec
    [-16,-42], #goi
    [-13,-60], #rio branco
    [-22,-47], #porto velho
    [-23,-38], #man
    [-21,-47], #camp
    [-23,-51], #for
    [-27,-48], #maringá
    [-21,-43], #floripa
    [-11,-44], #juizFora
    [-10,-67], #belém
    [-29,-63], #ribeirão
    [-18,-58], #aracaju
    [-13,-55],
    [-40,-45],#SLuis
    [-42,-50]
    ]
)
kmeans =KMeans(n_clusters = 3,
               init = 'k-means++', n_init = 10,
               max_iter = 300)
pred_y = kmeans.fit_predict(dataset)


plt.scatter(dataset[:,1], dataset[:,0], c = pred_y)
plt.xlim(-75,-30)
plt.ylim(-50,10)
plt.grid()

print(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0])
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], s = 70, c = 'red')
plt.show()

