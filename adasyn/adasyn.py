# This approach comes from ADASYN: Adaptive Synthetic Sampling
# Approach for Imbalanced Learning by Haibo He, Yang Bai,
# Edwardo A. Garcia, and Shutao Li. The descriptions are most
# from their paper. To read the paper, pleasego to:
# http://www.ele.uri.edu/faculty/he/PDFfiles/adasyn.pdf

from sklearn.neighbors import NearestNeighbors
import numpy as np
import random


class Adasyn:
    """
    ADASYN: Adaptive Synthetic Sampling

    Parameters
    -----------
    X           2D array
                feature space X

    Y           array
                label, y is either -1 or 1

    dth         float in (0,1]
                preset threshold
                maximum tolerated degree of class imbalance ratio

    b           float in [0, 1]
                desired balance level after generation of the synthetic data

    K           Integer
                number of nearest neighbors

    mutually_exclusive_groups   list of lists
                Groups of column indices that are mutually exclusive.
                For each group, only one feature can be active (non-zero) at a time.

    Attributes
    ----------
    ms          Integer
                the number of minority class examples

    ml          Integer
                the number of majority class examples

    d           float in n (0, 1]
                degree of class imbalance, d = ms/ml

    minority    Integer label
                the class label which belong to minority

    neighbors   K-Nearest Neighbors model

    synthetic   2D array
                array for synthetic samples
    """
    def __init__(self, X, Y, dth, b, K, mutually_exclusive_groups=None):
        self.X = X
        self.Y = Y
        self.K = K
        self.ms, self.ml, self.d, self.minority = self.calculate_degree()
        self.dth = dth
        self.b = b
        self.neighbors = NearestNeighbors(n_neighbors=self.K).fit(self.X)
        self.synthetic = []
        self.mutually_exclusive_groups = mutually_exclusive_groups or []

    def calculate_degree(self):
        pos, neg = 0, 0
        for i in range(0, len(self.Y)):
            if self.Y[i] == 1:
                pos += 1
            elif self.Y[i] == -1:
                neg += 1
        ml = max(pos, neg)
        ms = min(pos, neg)
        d = 1. * ms / ml
        if pos > neg:
            minority = -1
        else:
            minority = 1
        return ms, ml, d, minority

    def sampling(self):
        if self.d < self.dth:
            # a: calculate the number of synthetic data examples
            #    that need to be generated for the minority class
            G = (self.ml - self.ms) * self.b

            # b: for each xi in minority class, find K nearest neighbors
            # based on Euclidean distance in n-d space and calculate ratio
            # ri = number of examples in K nearest neighbors of xi that
            # belong to majority class, therefore ri in [0,1]
            r = []
            for i in range(0, len(self.Y)):
                if self.Y[i] == self.minority:
                    delta = 0
                    neighbors = self.neighbors.kneighbors([self.X[i]], self.K, return_distance=False)[0]
                    for neighbors_index in neighbors:
                        if self.Y[neighbors_index] != self.minority:
                            delta += 1
                    r.append(1. * delta/self.K)

            # c: normalize ri to get density distribution
            r = np.array(r)
            sum_r = np.sum(r)
            if sum_r == 0:
                raise ValueError("NaN values appear. Please "
                                 "try to use SMOTE or other methods.""")
            r = r / sum_r

            # d: calculate the number of synthetic data examples that
            # need to be generated for each minority example xi
            g = r * G

            # e: for each minority class data example, generate gi
            # synthetic data examples
            index = 0
            for i in range(0, len(self.Y)):
                if self.Y[i] == self.minority:
                    neighbors = self.neighbors.kneighbors([self.X[i]], self.K, return_distance=False)[0]
                    xzi_set = []
                    for j in neighbors:
                        if self.Y[j] == self.minority:
                            xzi_set.append(j)

                    for g_index in range(0, int(g[index])):
                        # 增加随机性：随机选择多个邻居进行插值
                        if len(xzi_set) > 1:
                            # 随机选择2-3个邻居进行加权平均
                            num_neighbors = min(random.randint(2, 3), len(xzi_set))
                            selected_neighbors = random.sample(xzi_set, num_neighbors)
                            
                            # 为每个邻居分配随机权重
                            weights = np.random.dirichlet(np.ones(num_neighbors))
                            
                            # 计算加权平均的邻居点
                            xzi_weighted = np.zeros_like(self.X[i])
                            for j, neighbor_idx in enumerate(selected_neighbors):
                                xzi_weighted += weights[j] * np.array(self.X[neighbor_idx])
                        else:
                            # 如果只有一个邻居，使用原来的方法
                            random_num = random.randint(0, len(xzi_set) - 1)
                            xzi_weighted = np.array(self.X[xzi_set[random_num]])
                        
                        xi = np.array(self.X[i])
                        
                        # 增加随机性：使用多个随机参数
                        random_lambda1 = random.random()
                        random_lambda2 = random.random()
                        
                        # 使用更复杂的插值公式
                        synthetic_sample = xi + (xzi_weighted - xi) * random_lambda1
                        
                        # 添加额外的随机扰动
                        noise_scale = 0.1 * np.std(self.X, axis=0)
                        noise = np.random.normal(0, noise_scale)
                        synthetic_sample = synthetic_sample + noise * random_lambda2

                        # Apply mutually exclusive constraints
                        synthetic_sample = self._apply_mutually_exclusive_constraints(synthetic_sample)

                        self.synthetic.append(synthetic_sample.tolist())
                    index += 1

    def _apply_mutually_exclusive_constraints(self, sample):
        """
        Apply mutually exclusive constraints to generated sample.
        For each group of mutually exclusive features, only the feature
        with the highest value will be set to 1, others to 0.

        Parameters
        ----------
        sample : numpy array
            The synthetic sample to constrain

        Returns
        -------
        constrained_sample : numpy array
            Sample with mutually exclusive constraints applied
        """
        constrained_sample = sample.copy()

        for group in self.mutually_exclusive_groups:
            if len(group) > 0:
                # Get values for this group
                group_values = constrained_sample[group]

                # Find the index with maximum value
                max_idx = np.argmax(group_values)

                # Set all to 0, then set max to 1
                for i, col_idx in enumerate(group):
                    if i == max_idx:
                        constrained_sample[col_idx] = 1
                    else:
                        constrained_sample[col_idx] = 0

        return constrained_sample
