import faiss
import time
import numpy as np

def train_kmeans(x, k, max_point_per_centroid=100000, min_points_per_centroid=20,
                    verbose=True, niter=20, ngpu=1, seed=10):
        '''
        Runs kmeans on one or several GPUs, 
        x: features, [data_num, dim]
        k: cluster number
        niter: kmeans iteration number
        max_point_per_centroid: max number of points in per cluster
        min_points_per_centroid: min number of points in per cluster
        return centroid, pseudo lablels
        '''
        d = x.shape[1]
        clus = faiss.Clustering(d, k)
        clus.verbose=verbose
        clus.seed=seed
        clus.niter=niter

        #  otherwise the kmeans implementation sub-samples the training set
        clus.max_points_per_centroid=max_point_per_centroid
        clus.min_points_per_centroid=min_points_per_centroid

        res=[faiss.StandardGpuResources() for i in range(ngpu)]

        flat_config = []
        for i in range(ngpu):
            cfg=faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = i
            flat_config.append(cfg)

        if ngpu == 1:
            index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
        else:
            indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpu)]
            index = faiss.IndexReplicas()
            for sub_index in indexes:
                index.addIndex(sub_index)

        #  perform the training
        clus.train(x, index)
        centroids = faiss.vector_float_to_array(clus.centroids)

        D, I = index.search(x, 1) #  for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I] #  label assignments

        return centroids.reshape(k, d), im2cluster

if __name__=='__main__':

    start = time.time()
    cf = np.random.rand(1000, 2048)
    train_kmeans(cf, k=50)
    print('using {:.2f}s'.format(time.time() - start))