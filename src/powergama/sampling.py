import pandas as pd
import numpy as np
from scipy.stats.distributions import norm
from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import pairwise_distances_argmin
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KernelDensity
import sklearn.cluster
import sklearn.preprocessing


def _TMPsample_kmeans(X, samplesize):
    """K-means sampling

    Parameters
    ==========
    X : matrix
        data matrix to sample from
    samplesize : int
        size of sample

    This method relies on sklearn.cluster.KMeans

    """
    """
    TODO: Have to weight the importance, i.e. multiply timeseries with
    installed capacities in order to get proper clustering.
    """

    n_clusters=samplesize
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(X)
    # which cluster nr it belongs to:
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique, X_indecies = np.unique(k_means_labels,
                                                  return_index=True)
    #k_means_predict = k_means.predict(X)

    return k_means_cluster_centers


def _TMPsample_mmatching(X, samplesize):
    """
    The idea is to make e.g. 10000 randomsample-sets of size=samplesize
    from the originial datasat X.

    Choose the sampleset with the lowest objective:
    MINIMIZE [(meanSample - meanX)^2 + (stdvSample - stdvX)^2...]

    in terms of stitistical measures
    """

    return


def _TMPsample_meanshift(X, samplesize):
    """M matching sampling

    Parameters
    ==========
    X : matrix
        data matrix to sample from
    samplesize : int
        size of sample

    This method relies on sklearn.cluster.MeanShift

    It is a centroid-based algorithm, which works by updating candidates
    for centroids to be the mean of the points within a given region.
    These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.
    """
    #from sklearn.cluster import MeanShift, estimate_bandwidth
    #from sklearn.datasets.samples_generator import make_blobs

    # The following bandwidth can be automatically detected using
    bandwidth = sklearn.cluster.estimate_bandwidth(X, quantile=0.2,
                                                   n_samples=samplesize)

    ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    #labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    #labels_unique = np.unique(labels)
    #n_clusters_ = len(labels_unique)

    #print("number of estimated clusters : %d" % n_clusters_)

    return cluster_centers


def _TMPsample_latinhypercube(X, samplesize):
    """Latin hypercube sampling

    Parameters
    ==========
    X : matrix
        data matrix to sample from
    samplesize : int
        size of sample

    This method relies on pyDOE.lhs(n, [samples, criterion, iterations])

    """
    """
    lhs(n, [samples, criterion, iterations])

    n:an integer that designates the number of factors (required)
    samples: an integer that designates the number of sample points to generate
        for each factor (default: n)
    criterion: a string that tells lhs how to sample the points (default: None,
        which simply randomizes the points within the intervals):
        “center” or “c”: center the points within the sampling intervals
        “maximin” or “m”: maximize the minimum distance between points, but place
                          the point in a randomized location within its interval
        “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
        “correlation” or “corr”: minimize the maximum correlation coefficient
    """
    from pyDOE import lhs
    X_rows = X.shape[0]; X_cols = X.shape[1]
    X_mean = X.mean(); X_std = X.std()
    X_sample = lhs( X_cols , samples=samplesize , criterion='center' )
    kernel=False
    if kernel:
        # Fit data w kernel density
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
        kde.score_samples(X)
        # random sampling (TODO: fit to latin hypercube sample):
        kde_sample = kde.sample(samplesize)
    else:
        # Fit data w normal distribution
        for i in range(X_cols):
            X_sample[:,i] = norm(loc=X_mean[i] , scale=X_std[i]).ppf(X_sample[:,i])

    return X_sample

def sampleProfileData(data, samplesize, sampling_method):
        """ Sample data from full-year time series

        Parameters
        ==========
        data : GridData object
            Sample from data.profiles
        samplesize : int
            size of sample
        sampling_method : str
            'kmeans', 'uniform',
            EXPERIMENTAL: 'kmeans_scale', 'lhs',  ('mmatching', 'meanshift')


        Returns
        =======
            reduced data matrix according to sample size and method



        """

        """
        Harald:
        TODO: Tidy up - remove irrelevant code.
        -Note: Profiles may also include generator cost profile, not only
        generation and consumption
        -Should we cluster the profiles, or all consumers and generators? This
        is different sine many generators/consumers may use the same profile.
        Using all generators/consumers is more difficult, but probably more
        correct
        -How to determine weight between different types of variations, i.e.
        generation/consumption (MW) vs marginal costs (€)? Using normalised
        profiles with no weighing is one such choice.
        """

        X = data.profiles.copy()

        if sampling_method == 'kmeans':
            """
            Harald preferred method:
            Scale all profiles to have similar variability,
            then cluster, and finally scale back
            """
            # Consider only those profiles which are used in the model:
            profiles_in_use = data.generator['inflow_ref'].append(
                data.generator['fuelcost_ref']).append(
                data.consumer['demand_ref']).unique().tolist()
            X = X[profiles_in_use]

            #scaler = sklearn.preprocessing.MinMaxScaler()
            scaler = sklearn.preprocessing.RobustScaler()
            x_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(data=x_scaled, columns=X.columns,
                          index=X.index)
            km_norm2=sklearn.cluster.KMeans(n_clusters=samplesize,
                                            init='k-means++')
            km_norm2.fit(X)
            km_orig=scaler.inverse_transform(km_norm2.cluster_centers_)
            X_sample = pd.DataFrame(data=km_orig,columns=X.columns)
            return X_sample

        elif sampling_method == 'kmeans_scale':
            print("Using k-means with scaled profiles -> IN PROGRESS")
            #TODO: How to scale prices?
            # Multiply time series for load and VRES with their respective
            # maximum capacities in order to get the correct clusters/samples

            for k in data.profiles.columns.values.tolist():
                ref = k
                pmax = sum(data.generator.pmax[g] for g in range(len(data.generator)) if data.generator.inflow_ref[g] == ref)
                if pmax > 0:
                    X[ref] = data.profiles[ref] * pmax
#            for k,row in self.generator.iterrows():
#                pmax = row['pmax']
#                ref = row['inflow_ref']
#                if X[ref].mean()<1:
#                    X[ref] = self.profiles[ref] * pmax
            X['const'] = 1

            for k,row in data.consumer.iterrows():
                pmax = row['demand_avg']
                ref = row['demand_ref']
                X[ref] = data.profiles[ref] * pmax

                X_sample = _TMPsample_kmeans(X, samplesize)
                X_sample = pd.DataFrame(data=X_sample,
                        columns=X.columns)

            # convert back to relative values
            for k in data.profiles.columns.values.tolist():
                ref = k
                pmax = sum(data.generator.pmax[g] for g in range(len(data.generator)) if data.generator.inflow_ref[g] == ref)
                if pmax > 0:
                    X_sample[ref] = X_sample[ref] / pmax
#            for k,row in self.generator.iterrows():
#                pmax = row['pmax']
#                ref = row['inflow_ref']
#                if pmax == 0:
#                    centroids[ref] = 0
#                else:
#                    if X[ref].mean()>1:
#                        centroids[ref] = centroids[ref] / pmax
            for k,row in data.consumer.iterrows():
                pmax = row['demand_avg']
                ref = row['demand_ref']
                if pmax == 0:
                    X_sample[ref] = 0
                else:
                    X_sample[ref] = X_sample[ref] / pmax
            X_sample['const'] = 1
            return X_sample

        elif sampling_method == 'mmatching':
            print("Using moment matching... -> NOT IMPLEMENTED")
        elif sampling_method == 'meanshift':
            print("Using Mean-Shift... -> EXPERIMENTAL")
            X_sample = _TMPsample_meanshift(X, samplesize)
            return X_sample
        elif sampling_method == 'lhs':
            print("Using Latin-Hypercube... -> EXPERIMENTAL")
            X_sample = _TMPsample_latinhypercube(X, samplesize)
            X_sample = pd.DataFrame(data=X_sample,
                        columns=X.columns)
            X_sample['const'] = 1
            X_sample[(X_sample < 0)] = 0
            return X_sample
        elif sampling_method == 'uniform':
            print("Using uniform sampling (consider changing sampling method!)...")
            #Use numpy random in order to have control of ranom seed from
            # top level script (np.random.seed(..))
            timerange=np.random.choice(data.profiles.shape[0],
                                        size=samplesize,replace=False)

            #timerange = random.sample(range(8760),samplesize)
            X_sample = data.profiles.loc[timerange, :]
            X_sample.index = list(range(len(X_sample.index)))
            return X_sample
        else:
            raise Exception("Unknown sampling method")
        return
