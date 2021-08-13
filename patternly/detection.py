import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from zedsuite.zutil import Llk, Lsmash
from zedsuite.genesess import GenESeSS
from zedsuite.quantizer import Quantizer
from patternly._utils import RANDOM_NAME, os_remove


class AnomalyDetection:
    """ Tool for anomaly detection """
    def __init__(
        self,
        *,
        anomaly_sensitivity=1,
        clustering_alg=KMeans(),
        quantize=True,
        quantize_type="complex",
        eps=0.1,
        verbose=False
    ) -> None:
        """
        Args:
            anomaly_sensitivity (float, optional): how many standard deviations above the mean llk to consider
                an anomaly (Default = 1)
            cluster_alg (sklearn.cluster, optional): clustering algorithm to use (Default = KMeans())
            quantize (bool, optional): whether to quantize the data (Default = True)
            quantize_type (str, optional): type of quantization to use ("complex" or "simple") (Default = "complex")
            eps (float, optional): epsilon parameter for finding PFSAs (Default = 0.1)
            verbose (bool, optional): whether to print verbose output (Default = False)

        """

        self.anomaly_sensitivity = anomaly_sensitivity
        self.clustering_alg = clustering_alg
        self.quantize = quantize
        self.quantize_type = quantize_type
        self.eps = eps
        self.verbose = verbose
        self.temp_dir = "zed_temp"

        # values calculated in self.fit()
        self.fitted = False               # whether model has been fit
        self.quantizer = None              # quantizer used for quantizing data
        self.data_file = ""                # file path of original data
        self.dist_matrix = pd.DataFrame()  # calculated by lsmash, used for clustering
        self.n_clusters = 0                # number of clusters
        self.clusters = []                 # list of cluster labels
        self.cluster_files = []            # list of file paths to clusters
        self.dot_files = []                # list of file paths to PFSA dot files
        self.cluster_PFSAs = []            # list of file paths to cluster PFSAs
        self.cluster_PFSAs_info = []       # list of dicts of cluster PFSAs info for printing
        self.PFSA_llk_means = []           # list of mean llk for each cluster
        self.PFSA_llk_stds = []            # list of std of llk for each cluster

        # can be accessed after calling self.predict()
        self.curr_cluster_llk_vec = None
        self.closest_match = None


    def fit(self, X, y=None):
        """ Fit an anomaly detection model

        Args:
            X (pd.DataFrame or str): time series data to be fit
            y (pd.Series, optional): labels for X only provided for sklearn standard (Default = None)

        Returns:
            AnomalyDetection: fitted model
        """

        X_quantized = self.__quantize(X)
        self.__calculate_dist_matrix(X if type(X) is str else X_quantized)
        self.__calculate_cluster_labels()
        self.__write_cluster_files(X_quantized)
        self.__calculate_cluster_PFSAs()
        self.__calculate_PFSA_stats()
        self.fitted = True

        return self


    def predict(self, X=None, *, clean=True):
        """ Predict whether a time series sequence is anomalous

        Args:
            X (pd.DataFrame or pd.Series or str, optional): time series data to find anomalies, if None then
                predicts on original data (Default = None)
            clean (bool, optional): whether to remove temp files (Default = True)

        Returns:
            bool or list[bool]: True if time series is an anomaly, False otherwise output shape depends on input
        """

        if not self.fitted:
            raise ValueError("Model has not been fit yet")

        seqfile = ""
        num_predictions = 0
        # commonly want to find anomalies in original data
        if X is None and os.path.isfile(self.data_file):
            seqfile = self.data_file
            num_predictions = len(self.clusters)
        else:
            # if type(X) is str and not self.quantize:
            #     seqfile = X
            # else:
            is_series = type(X) is pd.Series
            seqfile = RANDOM_NAME(path=self.temp_dir, clean=True)
            X_quantized = self.__quantize(X)
            X_quantized.to_csv(
                seqfile,
                sep=" ",
                line_terminator=(" " if is_series else "\n"),
                index=False,
                header=False
            )

            if is_series:
                num_predictions = 1
                # remove trailing space because it affects llk calculation
                with open(seqfile, "r+") as f:
                    line = next(f).rstrip()
                    f.seek(0)
                    f.write(line + "\n")
            else:
                num_predictions = X_quantized.shape[0]

        cluster_llk_vec = np.empty([self.n_clusters, num_predictions], dtype=np.float64)
        anomaly_vec = np.zeros(num_predictions, dtype=np.int64)
        for i in range(self.n_clusters):
            curr_llks = Llk(seqfile=seqfile, pfsafile=self.cluster_PFSAs[i]).run()
            cluster_llk_vec[i] = curr_llks
            # classify llk as anomaly if greater than X standard deviations above the mean
            upper_bound = self.PFSA_llk_means[i] + (self.PFSA_llk_stds[i] * self.anomaly_sensitivity)
            for j, llk in enumerate(curr_llks):
                anomaly_vec[j] += 1 if llk > upper_bound else 0

        self.curr_cluster_llk_vec = cluster_llk_vec
        self.closest_match = np.argmin(cluster_llk_vec, axis=0)

        # consider to be anomaly if all llks above specified upper bound
        predictions = [x == self.n_clusters for x in anomaly_vec]

        if len(predictions) == 1:
            predictions = predictions[0]

        if seqfile != self.data_file and clean:
            os_remove(seqfile)

        return predictions


    def save_model(self, path="patternly_model.pickle"):
        """ Save model to file

        Args:
            path (str): file path to save model to
        """

        if not self.fitted:
            raise ValueError("Model has not been fit yet")

        PFSAs = {}
        for i, cluster_file in enumerate(self.cluster_PFSAs):
            with open(cluster_file, "r") as f:
                # parse PFSA file
                next_val = lambda f: next(f).split(":")[1].strip()
                ann_err = float(next_val(f))
                mrg_eps = float(next_val(f))
                syn_str = next_val(f)
                sym_frq = [float(n) for n in next_val(f).split(" ")]
                size = int(next_val(f).split("(")[1].split(")")[0])
                next(f) # skip #PITILDE line
                pitilde = [[float(val) for val in next(f).strip().split(" ")] for _ in range(size)]
                size = int(next_val(f).split("(")[1].split(")")[0])
                next(f) # skip #CONNX line
                connx = [[int(val) for val in next(f).strip().split(" ")] for _ in range(size)]

                PFSAs[i] = {
                    "%ANN_ERR": ann_err,
                    "%MRG_EPS": mrg_eps,
                    "%SYN_STR": syn_str,
                    "%SYM_FRQ": sym_frq,
                    "%PITILDE": pitilde,
                    "%CONNX": connx,
                }

        model = {
            "self": self,
            "PFSAs": PFSAs,
        }

        with open(path, "wb") as f:
            pickle.dump(model, f)


    @staticmethod
    def load_model(path="patternly_model.pickle"):
        """ Load saved model

            Args:
                path (str): path to saved model

            Returns:
                AnomalyDetection: loaded model
        """

        with open(path, "rb") as f:
            model = pickle.load(f)

        self = model["self"]
        PFSAs = model["PFSAs"]

        # write PFSA files
        for i in range(self.n_clusters):
            self.cluster_PFSAs[i] = RANDOM_NAME(path=self.temp_dir)
            with open(self.cluster_PFSAs[i], "w") as f:
                f.write(f"%ANN_ERR: {PFSAs[i]['%ANN_ERR']}\n")
                f.write(f"%MRG_EPS: {PFSAs[i]['%MRG_EPS']}\n")
                f.write(f"%SYN_STR: {PFSAs[i]['%SYN_STR']}\n")
                f.write(f"%SYM_FRQ: ")
                for sym_frq in PFSAs[i]["%SYM_FRQ"]:
                    suffix = " " if str(sym_frq) != str(PFSAs[i]["%SYM_FRQ"][-1]) else " \n"
                    f.write(f"{sym_frq}{suffix}")
                f.write(f"%PITILDE: size({len(PFSAs[i]['%PITILDE'])})\n")
                f.write(f"#PITILDE\n")
                for pitilde in PFSAs[i]["%PITILDE"]:
                    for val in pitilde:
                        suffix = " " if str(val) != str(pitilde[-1]) else " \n"
                        f.write(f"{val}{suffix}")
                f.write(f"%CONNX: size({len(PFSAs[i]['%CONNX'])})\n")
                f.write(f"#CONNX\n")
                for connx in PFSAs[i]["%CONNX"]:
                    for val in connx:
                        suffix = " " if str(val) != str(connx[-1]) else " \n"
                        f.write(f"{val}{suffix}")
                f.write("\n")

        return self


    def print_PFSAs(self):
        """ Print PFSAs found for each cluster """

        if not self.fitted:
            raise ValueError("Model has not been fit yet")

        properties = ["%ANN_ERR", "%MRG_EPS", "%SYN_STR", "%SYM_FRQ", "%PITILDE", "%CONNX"]
        for i in range(self.n_clusters):
            print(f"Cluster {i} PFSA:")
            for prop in properties:
                print(f"{prop}: {self.cluster_PFSAs_info[i][prop]}")
            if i != len(self.cluster_PFSAs_info) - 1:
                print("\n")


    def __quantize(self, X):
        """ Quantize the data into finite alphabet

            Args:
                X (pd.DataFrame or str): time series data to be quantized
        """

        if not self.quantize or self.quantize_type is None:
            if type(X) is str:
                return pd.read_csv(X, sep=" ", header=None, low_memory=False).dropna(how="all", axis=1)
            else:
                return X.copy()

        if self.verbose:
            print("Quantizing...")

        # Quantizer() expects a DataFrame
        if type(X) is pd.Series:
            X = X.copy().to_frame().reset_index(drop=True).T

        if self.quantize_type == "simple":
            # basic differentiation
            X = X.astype(float).diff(axis=1).fillna(0)
            return X.apply(lambda row : row.apply(lambda n : 1 if n > 0 else 0), axis=1)
        elif self.quantize_type == "complex":
            # use cythonized quantizer binary
            if not self.fitted:
                self.quantizer = Quantizer(n_quantizations=1, epsilon=-1)
                self.quantizer.fit(X)
            return pd.concat([quantized for quantized in self.quantizer.transform(X)], axis=1)
        else:
            raise ValueError(f"Unknown quantize type: {self.quantize_type}. Choose \"simple\" or \"complex\".")


    def __calculate_dist_matrix(self, X):
        """ Calculate distance matrix using lsmash

            Args:
                X (pd.DataFrame or str): time series data to calculate distance matrix from
        """

        if self.verbose:
            print("Calculating distance matrix...")

        # don't create file if it already exists i.e. X is a file path
        if type(X) is str and not self.quantize:
            self.data_file = X
        else:
            self.data_file = RANDOM_NAME(path=self.temp_dir)
            X.to_csv(self.data_file, sep=" ", index=False, header=False)

        self.dist_matrix = pd.DataFrame(Lsmash(seqfile=self.data_file, data_type="symbolic", sae=False).run()).round(8)


    def __calculate_cluster_labels(self):
        """ Cluster distance matrix """

        if self.verbose:
            print("Clustering distance matrix...")

        clusters = self.clustering_alg.fit(self.dist_matrix).labels_
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # reassign clusters such that cluster 0 is the most common label, 1 is the second most common, etc.
        cluster_counts = np.zeros(n_clusters, dtype=np.int64)
        for cluster in clusters:
            cluster_counts[cluster] += 1
        cluster_rank = np.full(n_clusters, n_clusters - 1, dtype=np.int64) - np.argsort(np.argsort(cluster_counts))
        clusters = [cluster_rank[cluster] for cluster in clusters]

        self.n_clusters = n_clusters
        self.clusters = clusters


    def __write_cluster_files(self, X):
        """ Write cluster time series data to files

            Args:
                X (pd.DataFrame or str): time series data to write to files according to cluster
        """

        X["cluster"] = self.clusters
        cluster_files = []
        if (self.n_clusters == 1):
            cluster_files.append(self.data_file)
        else:
            for i in range(self.n_clusters):
                if self.verbose:
                    print(f"Writing cluster {i + 1}/{self.n_clusters} to file...")
                cluster_files.append(RANDOM_NAME(path=self.temp_dir))
                X[X["cluster"] == i] \
                    .drop("cluster", axis=1) \
                    .to_csv(cluster_files[i], sep=" ", header=False, index=False, float_format="%g")

        self.cluster_files = cluster_files


    def __calculate_cluster_PFSAs(self):
        """ Infer PFSAs from clusters using genESeSS """

        cluster_PFSAs = []
        dot_files = []
        for i, cluster_file in enumerate(self.cluster_files):
            if self.verbose:
                print(f"Generating cluster PFSA {i + 1}/{self.n_clusters}...")
            PFSA_file = RANDOM_NAME(path=self.temp_dir)
            cluster_PFSAs.append(PFSA_file)
            dot_file = RANDOM_NAME(path=self.temp_dir)
            dot_files.append(dot_file)
            alg = GenESeSS(
                datafile=cluster_file,
                outfile=PFSA_file,
                data_type="symbolic",
                data_dir="row",
                force=True,
                eps=self.eps,
                dot=dot_file,
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
        self.dot_files = dot_files


    def __calculate_PFSA_stats(self):
        """ Calculate the means and standard deviations of llks for each PFSA
            to later determine if a sequence is an anomaly
        """

        if self.verbose:
            print("Calculating cluster PFSA means and stds...")

        PFSA_llk_means = []
        PFSA_llk_stds = []
        for i in range(len(self.cluster_PFSAs)):
            llks = np.array(Llk(seqfile=self.cluster_files[i], pfsafile=self.cluster_PFSAs[i]).run())
            PFSA_llk_means.append(np.mean(llks))
            PFSA_llk_stds.append(np.std(llks))

        self.PFSA_llk_means, self.PFSA_llk_stds = PFSA_llk_means, PFSA_llk_stds


    def __reduce_clusters(self):
        """ Attempt to reduce the number of clusters by combining clusters that
            generate similar PFSAs
        """

        pass


class StreamingDetection(AnomalyDetection):
    """ Tool for anomaly detection within a single data stream """
    def __init__(self, *, window_size=1000, window_overlap=0, **kwargs):
        """
        Args:
            window_size (int): size of sliding window
            window_overlap (int): overlap of sliding windows
        """

        super().__init__(**kwargs)
        self.window_size = window_size
        self.window_overlap = window_overlap

    def fit(self, X, y=None):
        X_split_streams = self.__split_streams(X)
        super().fit(X_split_streams)

    def predict(self, X=None):
        if X is None:
            return super().predict(X)
        else:
            X_split_streams = self.__split_streams(X)
            return super().predict(X_split_streams)

    def __split_streams(self, X):
        """ Split stream data into individual streams of length self.window_size
            that overlap by self.overlap

            Args:
                X (pd.DataFrame): data to be split into streams
        """

        if (self.verbose):
            print("Splitting data into individual streams...")

        beg = lambda i: (self.window_size * i) - (self.window_overlap * i)
        end = lambda i: beg(i) + self.window_size
        size = X.shape[0] // (self.window_size - self.window_overlap)
        return pd.concat(
            [X[beg(i):end(i)].reset_index(drop=True) for i in range(size)],
            axis=1
        ).T

