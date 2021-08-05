from typing import Union
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
        self.data_file = ""
        self.dist_matrix = pd.DataFrame()
        self.clusters = []               # list of cluster labels
        self.cluster_files = []          # list of file paths to clusters
        self.cluster_PFSAs = []          # list of file paths to cluster PFSAs
        self.cluster_PFSAs_info = []     # list of dicts of cluster PFSAs info for printing
        self.PFSA_llk_means = []         # list of mean llk for each cluster
        self.PFSA_llk_stds = []          # list of std of llk for each cluster

        # can be accessed after calling self.predict()
        self.curr_cluster_llk_vec = None


    def fit(self, X, y=None):
        X_quantized = self.__quantize(X)
        self.__calculate_dist_matrix(X if type(X) is str else X_quantized)
        self.__calculate_cluster_labels()
        self.__write_cluster_files(X_quantized)
        self.__calculate_cluster_PFSAs()
        self.__calculate_PFSA_stats()
        self._fitted = True
        return self


    def predict(self, X=None) -> Union[bool, list[bool]]:
        seqfile = ""
        if X is None:
            seqfile = self.data_file
        else:
            if type(X) is str and not self.quantize:
                seqfile = X
            else:
                seqfile = RANDOM_NAME(path=self.temp_dir, clean=False)
                self.__quantize(X).to_csv(
                    seqfile,
                    sep=" ",
                    line_terminator=(" " if type(X) is pd.Series else "\n"),
                    index=False,
                    header=False
                )

        cluster_llk_vec = []
        anomaly_vec = np.zeros(
            len(self.clusters),
            dtype=np.int8
        )
        for i in range(len(self.cluster_PFSAs)):
            curr_llks = Llk(seqfile=seqfile, pfsafile=self.cluster_PFSAs[i]).run()
            cluster_llk_vec.append(curr_llks)
            # classify llk as anomaly if greater than X standard deviations above the mean
            upper_bound = self.PFSA_llk_means[i] + (self.PFSA_llk_stds[i] * self.anomaly_sensitivity)
            for j, llk in enumerate(curr_llks):
                anomaly_vec[j] += 1 if llk > upper_bound else 0
        self.curr_cluster_llk_vec = cluster_llk_vec

        predictions = [x == len(self.cluster_PFSAs) for x in anomaly_vec]

        if len(predictions) == 1:
            predictions = predictions[0]

        if seqfile != self.data_file:
            os_remove(seqfile)

        return predictions


    def print_PFSAs(self) -> None:
        properties = ["%ANN_ERR", "%MRG_EPS", "%SYN_STR", "%SYM_FRQ", "%PITILDE", "%CONNX"]
        for i in range(len(self.cluster_PFSAs_info)):
            print(f"Cluster {i} PFSA:")
            for prop in properties:
                print(f"{prop}: {self.cluster_PFSAs_info[i][prop]}")
            if i != len(self.cluster_PFSAs_info) - 1:
                print("\n")


    def __quantize(self, X, quantize_type: str = "complex") -> pd.DataFrame:
        if not self.quantize or quantize_type is None:
            if type(X) is str:
                return pd.read_csv(X, sep=" ", header=None, low_memory=False).dropna(how="all", axis=1)
            else:
                return X.copy()
        if self.verbose:
            print("Quantizing...")
        # Quantizer() expects a DataFrame
        if type(X) is pd.Series:
            X = X.copy().to_frame().reset_index(drop=True).T
        if quantize_type == "complex":
            # use cythonized quantizer binary
            if not self._fitted:
                self.quantizer = Quantizer(n_quantizations=1, epsilon=-1)
                self.quantizer.fit(X)
            return pd.concat([quantized for quantized in self.quantizer.transform(X)], axis=1)
        else:
            # basic differentiation
            X = X.astype(float).diff(axis=1).fillna(0)
            return X.apply(lambda row : row.apply(lambda n : 1 if n > 0 else 0), axis=1)


    def __calculate_dist_matrix(self, X) -> None:
        if self.verbose:
            print("Calculating distance matrix...")
        # don't create file if it already exists i.e. X is a file path
        if type(X) is str and not self.quantize:
            self.data_file = X
        else:
            self.data_file = RANDOM_NAME(path=self.temp_dir)
            X.to_csv(self.data_file, sep=" ", index=False, header=False)

        self.dist_matrix = Lsmash(seqfile=self.data_file, data_type="symbolic", sae=False).run().round(8)


    def __calculate_cluster_labels(self) -> None:
        if self.verbose:
            print("Clustering distance matrix...")
        self.clusters = self.clustering_alg.fit(self.dist_matrix).labels_


    def __write_cluster_files(self, X: pd.DataFrame) -> None:
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        X["cluster"] = self.clusters
        cluster_files = []
        for i in range(n_clusters):
            if self.verbose:
                print(f"Writing cluster {i + 1}/{n_clusters} to file...")
            cluster_files.append(RANDOM_NAME(path=self.temp_dir))
            X[X["cluster"] == i] \
                .drop("cluster", axis=1) \
                .to_csv(cluster_files[i], sep=" ", header=False, index=False, float_format="%g")
        self.cluster_files = cluster_files


    def __calculate_cluster_PFSAs(self) -> None:
        cluster_PFSAs = []
        for i, cluster_file in enumerate(self.cluster_files):
            if self.verbose:
                print(f"Generating cluster PFSA {i + 1}/{len(self.cluster_files)}...")
            PFSA_file = RANDOM_NAME(path=self.temp_dir)
            cluster_PFSAs.append(PFSA_file)
            alg = GenESeSS(
                datafile=cluster_file,
                outfile=PFSA_file,
                data_type="symbolic",
                data_dir="row",
                force=True,
                eps=self.eps,
            )
            alg.run()
            PFSA_info = {}
            PFSA_info["%ANN_ERR"] = alg.inference_error
            PFSA_info["%MRG_EPS"] = alg.epsilon_used
            PFSA_info["%SYN_STR"] = alg.synchronizing_string_found
            PFSA_info["%SYM_FRQ"] = alg.symbol_frequency
            PFSA_info["%PITILDE"] = alg.probability_morph_matrix
            PFSA_info["%CONNX"] = alg.connectivity_matrix
            self.cluster_PFSAs_info.append(PFSA_info)
        self.cluster_PFSAs = cluster_PFSAs


    def __calculate_PFSA_stats(self) -> None:
        # calculate the means and standard deviations of llks for each PFSA
        # to later determine if a sequence is an anomaly
        if self.verbose:
            print("Calculating cluster PFSA means and stds...")
        PFSA_llk_means = []
        PFSA_llk_stds = []
        for i in range(len(self.cluster_PFSAs)):
            llks = np.array(Llk(seqfile=self.cluster_files[i], pfsafile=self.cluster_PFSAs[i]).run())
            PFSA_llk_means.append(np.mean(llks))
            PFSA_llk_stds.append(np.std(llks))
        self.PFSA_llk_means, self.PFSA_llk_stds = PFSA_llk_means, PFSA_llk_stds
