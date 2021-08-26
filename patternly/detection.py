import dill
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from zedsuite.zutil import Llk, Lsmash, Prun, DrawPFSA
from zedsuite.genesess import GenESeSS
from zedsuite.quantizer import Quantizer
from patternly._utils import RANDOM_NAME


class AnomalyDetection:
    """ Tool for anomaly detection """

    def __init__(
        self,
        *,
        anomaly_sensitivity=1,
        n_clusters=1,
        clustering_alg=None,
        quantize=True,
        quantize_type="complex",
        eps=0.1,
        verbose=False
    ) -> None:
        """
            Args:
                anomaly_sensitivity (float, optional): how many standard deviations above the mean llk to consider
                    an anomaly (Default = 1)
                n_clusters (int, optional): number of clusters to use with KMeans (Default = 1)
                cluster_alg (sklearn.cluster, optional): clustering algorithm to use, if None then KMeans (Default = None)
                quantize (bool, optional): whether to quantize the data (Default = True)
                quantize_type (str, optional): type of quantization to use ("complex" or "simple") (Default = "complex")
                eps (float, optional): epsilon parameter for finding PFSAs (Default = 0.1)
                verbose (bool, optional): whether to print verbose output (Default = False)

            Attributes:
                cluster_llks (np.ndarray): array of shape (n_clusters, n_predictions) containing the llk or each
                    prediction for each cluster after calling fit()
                closest_match (np.ndarray): array of shape (n_predictions) containing the closest match after
                    calling fit()
                cluster_PFSA_pngs (list[str]): list of file paths to PFSA png files
        """

        self.anomaly_sensitivity = anomaly_sensitivity
        self.n_clusters = n_clusters
        self.clustering_alg = clustering_alg
        self.quantize = quantize
        self.quantize_type = quantize_type
        self.eps = eps
        self.verbose = verbose
        self.temp_dir = "zed_temp"
        self.PFSA_prop_titles =  ["%ANN_ERR", "%MRG_EPS", "%SYN_STR", "%SYM_FRQ", "%PITILDE", "%CONNX"]

        # values calculated in self.fit()
        self.fitted = False                # whether model has been fit
        self.quantizer = None              # quantizer used for quantizing data
        self.quantized_data = None         # quantized data
        self.dist_matrix = pd.DataFrame()  # calculated by lsmash, used for clustering
        self.cluster_labels = []           # list of cluster labels
        self.cluster_PFSA_files = []       # list of file paths to cluster PFSAs
        self.cluster_PFSA_info = []        # list of dicts of cluster PFSAs info for printing
        self.cluster_PFSA_pngs = []        # list of file paths to PFSA pngs
        self.PFSA_llk_means = []           # list of mean llk for each cluster
        self.PFSA_llk_stds = []            # list of std of llk for each cluster

        # can be accessed after calling self.predict()
        self.cluster_llks = None           # list of llk of each prediction for each cluster
        self.closest_match = None          # cluster label of closest match for each prediction


    def fit(self, X, y=None):
        """ Fit an anomaly detection model

        Args:
            X (pd.DataFrame or pd.Series): time series data to be fit
            y (pd.Series, optional): labels for X only provided for sklearn standard (Default = None)

        Returns:
            AnomalyDetection: fitted model
        """

        self.quantized_data = self.__quantize(X)
        self.__calculate_dist_matrix()
        self.__calculate_cluster_labels()
        self.__calculate_cluster_PFSAs()
        self.__calculate_PFSA_stats()
        self.__reduce_clusters()
        self.fitted = True

        return self


    def predict(self, X=None, *, clean=True):
        """ Predict whether a time series sequence is anomalous

        Args:
            X (pd.DataFrame or pd.Series, optional): time series data to find anomalies, if None then
                predicts on original data (Default = None)
            clean (bool, optional): whether to remove temp files (Default = True)

        Returns:
            bool or list[bool]: True if time series is an anomaly, False otherwise output shape depends on input
        """

        if not self.fitted:
            raise ValueError("Model has not been fit yet")

        data = None
        num_predictions = 0
        # commonly want to find anomalies in original data
        if X is None:
            # occurs when model is loaded from file
            if self.quantized_data is None:
                raise ValueError("Original data not found. Pass data to predict().")
            data = self.quantized_data.drop(columns=["cluster"], axis=1)
            num_predictions = self.quantized_data.shape[0]
        else:
            data = self.__quantize(X)
            num_predictions = 1 if type(X) is pd.Series else data.shape[0]

        cluster_llks = np.empty(shape=(self.n_clusters, num_predictions), dtype=np.float32)
        for i in range(self.n_clusters):
            cluster_llks[i] = np.asarray(Llk(data=data, pfsafile=self.cluster_PFSA_files[i]).run(), dtype=np.float32)

        # consider to be anomaly if all llks above specified upper bound (X standard deviations above the mean)
        upper_bounds = self.PFSA_llk_means + (self.PFSA_llk_stds * self.anomaly_sensitivity)
        predictions = np.all(cluster_llks.T > upper_bounds, axis=1)

        self.cluster_llks = cluster_llks
        self.closest_match = np.argmin(cluster_llks, axis=0)

        if len(predictions) == 1:
            predictions = predictions[0]

        return predictions


    def save_model(self, path="patternly_model.pickle"):
        """ Save model to file

        Args:
            path (str): file path to save model to
        """

        if not self.fitted:
            raise ValueError("Model has not been fit yet")

        metadata = {
            "modeltype": type(self),
            "user_params": {
                "anomaly_sensitivity": self.anomaly_sensitivity,
                "n_clusters": self.n_clusters,
                "clustering_alg": self.clustering_alg,
                "quantize": self.quantize,
                "quantize_type": self.quantize_type,
                "eps": self.eps,
                "verbose": self.verbose,
            },
            "fitted_params": {
                "quantizer_parameters": None if not self.quantize else self.quantizer.parameters ,
                "quantizer_feature_order": None if not self.quantize else self.quantizer._feature_order,
                "cluster_labels": self.cluster_labels,
                "cluster_PFSA_info": self.cluster_PFSA_info,
                "PFSA_llk_means": self.PFSA_llk_means,
                "PFSA_llk_stds": self.PFSA_llk_stds
            }
        }

        with open(path, "wb") as f:
            dill.dump(metadata, f)


    @staticmethod
    def load_model(path="patternly_model.pickle"):
        """ Load saved model

            Args:
                path (str): path to saved model

            Returns:
                AnomalyDetection or StreamingDetection: loaded model
        """

        with open(path, "rb") as f:
            metadata = dill.load(f)

        model = metadata["modeltype"](**metadata["user_params"])
        if model.quantize:
            model.quantizer = Quantizer(n_quantizations=1, eps=-1)
            model.quantizer.parameters = metadata["fitted_params"]["quantizer_parameters"]
            model.quantizer._feature_order = metadata["fitted_params"]["quantizer_feature_order"]
        model.cluster_labels = metadata["fitted_params"]["cluster_labels"]
        model.cluster_PFSA_info = metadata["fitted_params"]["cluster_PFSA_info"]
        model.PFSA_llk_means = metadata["fitted_params"]["PFSA_llk_means"]
        model.PFSA_llk_stds = metadata["fitted_params"]["PFSA_llk_stds"]

        # write PFSA files
        model.cluster_PFSA_files = []
        for i in range(model.n_clusters):
            model.cluster_PFSA_files.append(RANDOM_NAME(path=model.temp_dir))
            with open(model.cluster_PFSA_files[i], "w") as f:
                f.write(f"{model.__format_PFSA_info(model.cluster_PFSA_info[i])}\n")
        model.generate_PFSA_pngs()

        model.fitted = True

        return model


    def print_PFSAs(self):
        """ Print PFSAs found for each cluster """

        if not self.fitted:
            raise ValueError("Model has not been fit yet")

        for i in range(self.n_clusters):
            print(f"Cluster {i} PFSA:")
            print(self.__format_PFSA_info(self.cluster_PFSA_info[i], indent_level=4))


    def generate_PFSA_pngs(self):
        """ Generates png files for PFSAs

            Returns:
                list[str]: list of file paths to png files
        """
        self.cluster_PFSA_pngs = []
        for i in range(self.n_clusters):
            self.cluster_PFSA_pngs.append(RANDOM_NAME(path=self.temp_dir))
            DrawPFSA(pfsafile=self.cluster_PFSA_files[i], graphpref=self.cluster_PFSA_pngs[i]).run()

        return self.cluster_PFSA_pngs


    def __quantize(self, X):
        """ Quantize the data into finite alphabet

            Args:
                X (pd.DataFrame): time series data to be quantized
        """

        if not self.quantize or self.quantize_type is None:
            return X.copy(deep=False).astype(np.int8)

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
            return pd.concat(
                [quantized for quantized in self.quantizer.transform(X)],
                axis=1,
                copy=False
            ).astype(np.int8)
        else:
            raise ValueError(f"Unknown quantize type: {self.quantize_type}. Choose \"simple\" or \"complex\".")


    def __calculate_dist_matrix(self):
        """ Calculate distance matrix using lsmash """

        if self.verbose:
            print("Calculating distance matrix...")

        if self.n_clusters == 1:
            self.dist_matrix = self.quantized_data
        else:
            self.dist_matrix = pd.DataFrame(
                Lsmash(data=self.quantized_data, data_type="symbolic", sae=False).run(),
                dtype=np.float32
            )


    def __calculate_cluster_labels(self):
        """ Cluster distance matrix """

        if self.verbose:
            print("Clustering distance matrix...")

        if self.n_clusters == 1:
            self.cluster_labels = np.zeros(shape=(self.dist_matrix.shape[0]))

        cluster_labels = []
        if self.clustering_alg is None:
            cluster_labels = KMeans(n_clusters=self.n_clusters).fit(self.dist_matrix).labels_
        else:
            cluster_labels = self.clustering_alg.fit(self.dist_matrix).labels_
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        # reassign clusters such that cluster 0 is the most common label, 1 is the second most common, etc.
        cluster_counts = np.zeros(n_clusters, dtype=np.int32)
        for cluster in cluster_labels:
            cluster_counts[cluster] += 1
        cluster_rank = np.full(n_clusters, n_clusters - 1, dtype=np.int32) - np.argsort(np.argsort(cluster_counts))
        cluster_labels = [cluster_rank[cluster] for cluster in cluster_labels]

        self.n_clusters = n_clusters
        self.cluster_labels = cluster_labels
        self.quantized_data["cluster"] = cluster_labels


    def __calculate_cluster_PFSAs(self):
        """ Infer PFSAs from clusters using genESeSS """

        cluster_PFSA_files = []
        PFSA_pngs = []
        for i in range(self.n_clusters):
            if self.verbose:
                print(f"Generating cluster PFSA {i + 1}/{self.n_clusters}...")
            cluster_data = self.quantized_data[self.quantized_data["cluster"] == i].drop(columns=["cluster"], axis=1)
            cluster_PFSA_file = RANDOM_NAME(path=self.temp_dir)
            dot_file = RANDOM_NAME(path=self.temp_dir)
            cluster_PFSA_files.append(cluster_PFSA_file)
            PFSA_pngs.append(dot_file)
            alg = GenESeSS(
                data=cluster_data,
                outfile=cluster_PFSA_file,
                data_type="symbolic",
                data_dir="row",
                force=True,
                eps=self.eps,
                dot=dot_file,
            )
            alg.run()
            prop_vals = [
                alg.inference_error,
                alg.epsilon_used,
                alg.synchronizing_string_found,
                alg.symbol_frequency,
                alg.probability_morph_matrix,
                alg.connectivity_matrix,
            ]
            self.cluster_PFSA_info.append(dict(zip(self.PFSA_prop_titles, prop_vals)))

        self.cluster_PFSA_files = cluster_PFSA_files
        self.cluster_PFSA_pngs = PFSA_pngs


    def __calculate_PFSA_stats(self):
        """ Calculate the means and standard deviations of llks for each PFSA
            to later determine if a sequence is an anomaly
        """

        if self.verbose:
            print("Calculating cluster PFSA means and stds...")

        PFSA_llk_means = np.empty(shape=self.n_clusters)
        PFSA_llk_stds = np.empty(shape=self.n_clusters)
        for i in range(self.n_clusters):
            cluster_data = self.quantized_data[self.quantized_data["cluster"] == i].drop(columns=["cluster"], axis=1)
            llks = np.asarray(Llk(data=cluster_data, pfsafile=self.cluster_PFSA_files[i]).run(), dtype=np.float32)
            PFSA_llk_means[i] = np.mean(llks)
            PFSA_llk_stds[i] = np.std(llks, ddof=1)

        self.PFSA_llk_means, self.PFSA_llk_stds = PFSA_llk_means, PFSA_llk_stds


    def __reduce_clusters(self):
        """ Attempt to reduce the number of clusters by combining clusters that
            generate similar PFSAs
        """

        # generate representative sequences for each cluster PFSA
        seqs_per_pfsa = 1
        seqs = pd.DataFrame()
        for pfsa in self.cluster_PFSA_files:
            curr_seqs = pd.DataFrame(Prun(pfsafile=pfsa, data_len=10000, num_repeats=seqs_per_pfsa).run())
            seqs = pd.concat([seqs, curr_seqs], axis=0)


        cluster_llks = []
        labels = range(self.n_clusters)
        for i, pfsa in enumerate(self.cluster_PFSA_files):
            curr_llks = np.asarray(Llk(data=seqs, pfsafile=pfsa).run())
            diffs = curr_llks - self.PFSA_llk_means[i]
            cluster_llks.append(curr_llks)
            # print(i, curr_llks)
            # print(i, diffs)
            # print(f"mean: {self.PFSA_llk_means[i]}, std: {self.PFSA_llk_stds[i]}")
            # print("\n")

        self.closest_match = np.argmin(cluster_llks, axis=0)
        self.mismatches = pd.concat([pd.DataFrame(self.closest_match), pd.DataFrame(labels)], axis=1)
        self.cluster_llks = cluster_llks

    def __format_PFSA_info(self, PFSA_info, indent_level=0):
        """ Format the PFSA information for output

            Args:
                PFSA_info (dict): PFSA information to format
                indent_level (int): indentation level for formatting

            Returns:
                str: formatted PFSA information
        """

        indent = indent_level * " "

        syn_str_vals = ""
        if PFSA_info["%SYN_STR"] is not None:
            for syn_str in PFSA_info["%SYN_STR"]:
                syn_str_vals += (f"{syn_str} ")

        sym_frq_vals = ""
        for sym_frq in PFSA_info["%SYM_FRQ"]:
            sym_frq_vals += (f"{sym_frq} ")

        pitilde_vals = indent
        for row in PFSA_info["%PITILDE"]:
            for val in row:
                suffix = " " if str(val) != str(row[-1]) else f" \n{indent}"
                pitilde_vals += (f"{val}{suffix}")
        pitilde_vals = pitilde_vals[:-len(indent)] if indent_level > 0 else pitilde_vals

        connx_vals = indent
        for row in PFSA_info["%CONNX"]:
            for val in row:
                suffix = " " if str(val) != str(row[-1]) else f" \n{indent}"
                connx_vals += (f"{val}{suffix}")
        connx_vals = pitilde_vals[:-len(indent)] if indent_level > 0 else connx_vals

        return (
            f"{indent}%ANN_ERR: {PFSA_info['%ANN_ERR']}\n"
            + f"{indent}%MRG_EPS: {PFSA_info['%MRG_EPS']}\n"
            + f"{indent}%SYN_STR: {syn_str_vals}\n"
            + f"{indent}%SYM_FRQ: {sym_frq_vals}\n"
            + f"{indent}%PITILDE: size({len(PFSA_info['%PITILDE'])})\n"
            + f"{indent}#PITILDE\n{pitilde_vals}"
            + f"{indent}%CONNX: size({len(PFSA_info['%CONNX'])})\n"
            + f"{indent}#CONNX\n{connx_vals}\n"
        )


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
            return super().predict()
        else:
            X_split_streams = self.__split_streams(X)
            return super().predict(X_split_streams)


    def __split_streams(self, X):
        """ Split stream data into individual streams of length self.window_size
            that overlap by self.overlap

            Args:
                X (pd.Series): data to be split into streams
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
