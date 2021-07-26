from typing import Tuple
import numpy as np
import sklearn.cluster
import pandas as pd
from zedsuite.zutil import Llk, Lsmash
from zedsuite.genesess import GenESeSS
from zedsuite.quantizer import Quantizer
from mantis.utils import RANDOM_NAME, os_remove


class AnomalyDetection:
    def __init__(
        self,
        *,
        anomaly_sensitivity: float = 1,
        clustering_alg: sklearn.cluster = sklearn.cluster.KMeans(),
        quantize: bool = True,
        eps: float = 0.1,
        verbose: bool = False
    ) -> None:
        self.anomaly_sensitivity = anomaly_sensitivity
        self.clustering_alg = clustering_alg
        self.quantize = quantize
        self.eps = eps
        self.verbose = verbose
        self.temp_dir = "zed_temp"

        # values calculated in self.fit()
        self._fitted = False
        self.quantizer = None
        self.dist_matrix = None
        self.clusters = None
        self.cluster_files = None
        self.cluster_PFSAs = None
        self.PFSA_llk_means = None
        self.PFSA_llk_stds = None

        # can be accessed after calling self.predict()
        self.curr_cluster_llk_vec = None

    def fit(self, X: pd.DataFrame, y=None):
        X_quantized = self.__quantize(X, multiline=True)
        self.dist_matrix = Lsmash(
            seq=X_quantized,
            data_type="symbolic",
            sae=False
        ).run()
        self.clusters = self.clustering_alg.fit(self.dist_matrix).labels_
        self.cluster_files = self.__generate_cluster_files(X_quantized, self.clusters)
        self.cluster_PFSAs = self.__generate_cluster_PFSAs(self.cluster_files, eps=self.eps)
        self.PFSA_llk_means, self.PFSA_llk_stds = self.__calculate_PFSA_stats(self.cluster_PFSAs, self.cluster_files)
        self._fitted = True

        return self

    def predict(self, X: pd.Series) -> bool:
        X_quantized = self.__quantize(X)
        seqfile = RANDOM_NAME(path=self.temp_dir, clean=False)
        X_quantized.to_csv(seqfile, sep=" ", line_terminator=" ", index=False, header=False)

        cluster_llk_vec = []
        anomaly_vec = []
        for i in range(len(self.cluster_PFSAs)):
            curr_llk = Llk(seqfile=seqfile, pfsafile=self.cluster_PFSAs[i]).run()[0]
            cluster_llk_vec.append(curr_llk)
            # classify llk as anomaly if greater than X standard deviations above the mean
            upper_bound = self.PFSA_llk_means[i] + (self.PFSA_llk_stds[i] * self.anomaly_sensitivity)
            anomaly_vec.append(1 if curr_llk > upper_bound else 0)

        os_remove(seqfile)

        self.curr_cluster_llk_vec = cluster_llk_vec
        return np.sum(anomaly_vec) == len(self.cluster_PFSAs)

    def __quantize(self, X: pd.DataFrame, quantize_type: str = "complex") -> pd.DataFrame:
        if type(X) is pd.Series:
            X = X.copy().to_frame().reset_index(drop=True).T
        if not self.quantize or quantize_type is None:
            return X.copy()
        if quantize_type == "complex":
            # use quantizer binary
            if not self._fitted:
                self.quantizer = Quantizer(n_quantizations=1, epsilon=-1)
                self.quantizer.fit(X)
            return pd.concat([quantized for quantized in self.quantizer.transform(X)], axis=1)
        else:
            # basic differentiation
            X = X.astype(float).diff(axis=1).fillna(0)
            return X.apply(lambda row : row.apply(lambda n : 1 if n > 0 else 0), axis=1)

    def __generate_cluster_files(self, X: pd.DataFrame, clusters: list[int]) -> list[str]:
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        X["cluster"] = clusters
        cluster_files = []
        for i in range(n_clusters):
            cluster_files.append(RANDOM_NAME(path=self.temp_dir))
            X[X["cluster"] == i] \
                .drop("cluster", axis=1) \
                .to_csv(cluster_files[i], sep=" ", header=False, index=False, float_format="%g")
        return cluster_files

    def __generate_cluster_PFSAs(self, cluster_files: list[int], eps: float = 0.1) -> list[str]:
        PFSAs = []
        for cluster_file in cluster_files:
            PFSA_file = RANDOM_NAME(path=self.temp_dir)
            PFSAs.append(PFSA_file)
            alg = GenESeSS(
                datafile=cluster_file,
                outfile=PFSA_file,
                data_type="symbolic",
                data_dir="row",
                force=True,
                eps=eps,
            )
            alg.run()
        return PFSAs

    def __calculate_PFSA_stats(self, PFSAs: list[str], cluster_files: list[str]) -> Tuple[list[float], list[float]]:
        # calculate the means and standard deviations of llks for each PFSA
        # to later determine if a sequence is an anomaly
        PFSA_llk_means = []
        PFSA_llk_stds = []
        for i in range(len(PFSAs)):
            llks = np.array(Llk(seqfile=cluster_files[i], pfsafile=PFSAs[i]).run())
            PFSA_llk_means.append(np.mean(llks))
            PFSA_llk_stds.append(np.std(llks))
        return PFSA_llk_means, PFSA_llk_stds
