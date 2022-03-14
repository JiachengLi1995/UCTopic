import math
import torch
import torch.nn.functional as F
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score

class KMeans:
    '''
    Kmeans clustering algorithm implemented with PyTorch
    Parameters:
    n_clusters: int, 
        Number of clusters
    max_iter: int, default: 100
        Maximum number of iterations
    tol: float, default: 0.0001
        Tolerance
    mode: {'euclidean', 'cosine'}, default: 'euclidean'
        Type of distance measure
    minibatch: {None, int}, default: None
        Batch size of MinibatchKmeans algorithm
        if None perform full KMeans algorithm
        
    Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
        cluster centroids
    '''
    def __init__(self, n_clusters, max_iter=100, tol=0.0001, mode="euclidean", minibatch=None):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.mode = mode
        self.minibatch = minibatch
        self._loop = False

        self.centroids = None

    @staticmethod
    def cos_sim(a, b):
        """
        Compute cosine similarity of 2 sets of vectors
        Parameters:
        a: torch.Tensor, shape: [m, n_features]
        b: torch.Tensor, shape: [n, n_features]
        """
        a_norm = a.norm(dim=-1, keepdim=True)
        b_norm = b.norm(dim=-1, keepdim=True)
        a = a / (a_norm + 1e-8)
        b = b / (b_norm + 1e-8)
        return a @ b.transpose(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
        Compute euclidean similarity of 2 sets of vectors
        Parameters:
        a: torch.Tensor, shape: [m, n_features]
        b: torch.Tensor, shape: [n, n_features]
        """
        return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

    def remaining_memory(self):
        """
        Get remaining memory in gpu
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            remaining = info.free
        except:
            remaining = torch.cuda.memory_allocated()
        return remaining

    def max_sim(self, a, b, return_sim=False):
        """
        Compute maximum similarity (or minimum distance) of each vector
        in a with all of the vectors in b
        Parameters:
        a: torch.Tensor, shape: [m, n_features]
        b: torch.Tensor, shape: [n, n_features]
        """
        device = a.device.type
        batch_size = a.shape[0]
        if self.mode == 'cosine':
            sim_func = self.cos_sim
        elif self.mode == 'euclidean':
            sim_func = self.euc_sim

        if device == 'cpu':
            sim = sim_func(a, b)
            max_sim_v, max_sim_i = sim.max(dim=-1)
            if return_sim:
                return max_sim_v, max_sim_i, sim
            else:
                return max_sim_v, max_sim_i
        else:
            if a.dtype == torch.float:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
            elif a.dtype == torch.half:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
            ratio = math.ceil(expected / self.remaining_memory())
            subbatch_size = math.ceil(batch_size / ratio)
            msv, msi, sim = [], [], []
            for i in range(ratio):
                if i*subbatch_size >= batch_size:
                    continue
                sub_x = a[i*subbatch_size: (i+1)*subbatch_size]
                sub_sim = sim_func(sub_x, b)
                sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
                if return_sim:
                    sim.append(sub_sim.cpu())
                else:
                    del sim
                msv.append(sub_max_sim_v)
                msi.append(sub_max_sim_i)
            if ratio == 1:
                max_sim_v, max_sim_i = msv[0], msi[0]
                if return_sim:
                    sim = sim[0]
            else:
                max_sim_v = torch.cat(msv, dim=0)
                max_sim_i = torch.cat(msi, dim=0)
                if return_sim:
                    sim = torch.cat(sim, dim=0)

            if return_sim:
                return max_sim_v, max_sim_i, sim
            else:
                return max_sim_v, max_sim_i

    def fit_predict(self, X, centroids=None, verbose=0):
        """
        Combination of fit() and predict() methods.
        This is faster than calling fit() and predict() seperately.
        Parameters:
        X: torch.Tensor, shape: [n_samples, n_features]
        centroids: {torch.Tensor, None}, default: None
            if given, centroids will be initialized with given tensor
            if None, centroids will be randomly chosen from X
        Return:
        labels: torch.Tensor, shape: [n_samples]
        """
        batch_size, emb_dim = X.shape
        device = X.device.type
        start_time = time()
        if centroids is None:
            self.centroids = X[np.random.choice(batch_size, size=[self.n_clusters], replace=False)]
        else:
            self.centroids = centroids
        num_points_in_clusters = torch.ones(self.n_clusters, device=device)
        closest = None

        if verbose > 0:
            tqdm_iter = tqdm(range(self.max_iter), desc='Clustering')
        else:
            tqdm_iter = range(self.max_iter)
        for i in tqdm_iter:
            iter_time = time()
            if self.minibatch is not None:
                x = X[np.random.choice(batch_size, size=[self.minibatch], replace=False)]
            else:
                x = X
            closest = self.max_sim(a=x, b=self.centroids)[1]
            matched_clusters, counts = closest.unique(return_counts=True)

            c_grad = torch.zeros_like(self.centroids)
            if self._loop:
                for j, count in zip(matched_clusters, counts):
                    c_grad[j] = x[closest==j].sum(dim=0) / count
            else:
                if self.minibatch is None:
                    expanded_closest = closest[None].expand(self.n_clusters, -1)
                    mask = (expanded_closest==torch.arange(self.n_clusters, device=device)[:, None]).float()
                    c_grad = mask @ x / mask.sum(-1)[..., :, None]
                    c_grad[c_grad!=c_grad] = 0 # remove NaNs
                else:
                    expanded_closest = closest[None].expand(len(matched_clusters), -1)
                    mask = (expanded_closest==matched_clusters[:, None]).float()
                    c_grad[matched_clusters] = mask @ x / mask.sum(-1)[..., :, None]

            error = (c_grad - self.centroids).pow(2).sum()
            if self.minibatch is not None:
                lr = 1/num_points_in_clusters[:,None] * 0.9 + 0.1
            else:
                lr = 1

            num_points_in_clusters[matched_clusters] += counts
            if error <= self.tol:
                break

            self.centroids = self.centroids * (1-lr) + c_grad * lr

        _, closest, scores = self.max_sim(a=X, b=self.centroids, return_sim=True)
        
        return closest, F.softmax(scores, dim=-1)

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to
        Parameters:
        X: torch.Tensor, shape: [n_samples, n_features]
        Return:
        labels: torch.Tensor, shape: [n_samples]
        """
        return self.max_sim(a=X, b=self.centroids)[1]

    def fit(self, X, centroids=None):
        """
        Perform kmeans clustering
        Parameters:
        X: torch.Tensor, shape: [n_samples, n_features]
        """
        self.fit_predict(X, centroids)


def get_silhouette_score(features: torch.Tensor, n_clusters: int, max_iter: int=300):

    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
    labels, kmeans_scores = kmeans.fit_predict(features)
    s_score = silhouette_score(features.cpu().numpy(), labels.numpy())
    return s_score, kmeans_scores, kmeans.centroids.cpu().numpy()

def get_kmeans(features: torch.Tensor, n_clusters: int, max_iter: int=300, verbose: int=1):

    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)
    labels, kmeans_scores = kmeans.fit_predict(features, verbose=verbose)

    return kmeans_scores, kmeans.centroids.cpu().numpy()

