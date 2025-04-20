# Comparison of K-means and DBSCAN Clustering on Synthetic Datasets

## About

This Python project compares the performance of two popular clustering algorithms, K-means and DBSCAN, on four different synthetic datasets. The code generates these datasets using `scikit-learn`'s `make_blobs`, `make_moons`, and `make_circles` functions, creating datasets with varying characteristics (Gaussian distribution, linear transformation, non-linear separability). The project then applies K-means and DBSCAN to each dataset, evaluates the clustering results using metrics like F1-score, Normalized Mutual Information (NMI), and Rand Index, and visualizes the cluster assignments.

## Features

* **Dataset Generation:**
    * Generates four synthetic datasets using `make_blobs`, `make_moons`, and `make_circles` with a fixed number of samples (300) and a calculated `random_state` based on student IDs.
    * Applies a linear transformation to one of the blob datasets to alter its structure.
    * Scales the features of each dataset using `StandardScaler`.
* **K-means Clustering:**
    * Implements the K-means algorithm from scratch, including centroid initialization, data point assignment, and centroid updating.
    * Uses the `KMeans` implementation from `scikit-learn` to obtain cluster labels.
    * Calculates Silhouette scores for K values from 2 to 7 to help in choosing the optimal K.
* **DBSCAN Clustering:**
    * Implements the DBSCAN algorithm from scratch, including neighbor finding and cluster expansion.
    * Applies DBSCAN to each dataset with chosen `eps` and `min_samples` parameters.
* **Evaluation Metrics:**
    * Evaluates the performance of both K-means and DBSCAN using:
        * F1-score (weighted average)
        * Normalized Mutual Information (NMI)
        * Rand Index
* **Visualization:**
    * Visualizes the original data points and the resulting cluster assignments for both K-means and DBSCAN on each dataset using `matplotlib`.
* **Summary Tables:**
    * Presents the evaluation metrics (F1-score, NMI, Rand Index) for both algorithms on each dataset in a pandas DataFrame, allowing for easy comparison.

## Libraries Used

* **numpy:** For numerical operations (e.g., matrix manipulation, distance calculations).
* **pandas:** For creating and displaying the summary tables.
* **scikit-learn (sklearn):**
    * `make_blobs`, `make_moons`, `make_circles`: For generating synthetic datasets.
    * `StandardScaler`: For scaling the data.
    * `KMeans`: For K-means clustering.
    * `f1_score`, `normalized_mutual_info_score`, `rand_score`: For evaluating clustering performance.
    * `metrics`: For calculating the Silhouette Score
* **matplotlib.pyplot:** For data visualization.

## Code Explanation

The code is structured as follows:

1.  **Import Libraries:** Imports the necessary Python libraries.
2.  **Calculate `random_state`:** Calculates a unique `random_state` for dataset generation based on a list of student IDs. This ensures consistent dataset generation across multiple runs.
3.  **Generate Datasets:**
    * Generates four synthetic datasets using `make_blobs`, `make_moons`, and `make_circles`.
    * Scales the data using `StandardScaler`.
4.  **K-means Implementation:**
    * `k_means(data, k, max_iterations)`:  Implements the K-means algorithm.
        * `initialize_centroids(data, k)`: Randomly selects initial centroids.
        * `assign_data_points(data, centroids)`: Assigns each data point to the nearest centroid.
        * `update_centroids(data, clusters, k)`: Calculates the new centroids based on the assigned data points.
    5. **DBSCAN Implementation:**
    * `dbscan(X, eps, min_samples)`: Implements the DBSCAN algorithm.
        * `get_neighbors(X, i, eps)`: Finds the neighbors of a given point within a specified radius (eps).
        * `expand_cluster(X, i, neighbors, labels, cluster_id, eps, min_samples)`: Recursively expands a cluster from a core point.
5.  **Apply Algorithms and Evaluate:**
    * For each dataset:
        * Applies K-means (using the custom implementation and scikit-learn's `KMeans`).
        * Applies DBSCAN (using the custom implementation).
        * Calculates F1-score, NMI, and Rand Index to evaluate the clustering results.
        * Prints the evaluation metrics.
        * Generates Silhouette plots to help select the best K for K-means
6.  **Visualization:**
    * Generates plots showing the original data and the cluster assignments for both K-means and DBSCAN for each dataset.
7.  **Summary Tables:**
    * Creates pandas DataFrames to display the evaluation metrics for each algorithm on each dataset.

## Usage

1.  **Prerequisites:**
    * Python 3.x
    * numpy
    * pandas
    * scikit-learn (sklearn)
    * matplotlib

2.  **Installation:**
    * Install the required libraries using pip:

        ```bash
        pip install numpy pandas scikit-learn matplotlib
        ```

3.  **Running the Code:**
    * Save the Python code to a file (e.g., `clustering_comparison.py`).
    * Run the script from the command line:

        ```bash
        python clustering_comparison.py
        ```

4.  **Output:**
    * The script will:
        * Print the evaluation metrics (F1-score, NMI, Rand Index) for K-means and DBSCAN on each dataset.
        * Display plots visualizing the cluster assignments for both algorithms on each dataset.
        * Display summary tables of the evaluation metrics.

## Results

The script outputs the following results:

* Evaluation metrics (F1-score, NMI, Rand Index) for K-means and DBSCAN on each of the four synthetic datasets.
* Plots visualizing the cluster assignments for both algorithms on each dataset.
* Summary tables in pandas DataFrames comparing the performance of the two algorithms.

The results demonstrate the performance of K-means and DBSCAN on datasets with different characteristics. K-means performs well on datasets with well-defined, spherical clusters (Datasets 1 and 2), while DBSCAN excels at identifying clusters with arbitrary shapes and handling noise (Datasets 3 and 4).

## Future Enhancements

* Explore other clustering algorithms (e.g., Agglomerative Clustering, Spectral Clustering).
* Experiment with different parameter settings for K-means and DBSCAN.
* Evaluate the algorithms on real-world datasets.
* Add more visualization options (e.g., 3D plots for higher-dimensional data).
* Implement methods for automatically determining the optimal number of clusters (K) for K-means.
* Incorporate techniques for handling high-dimensional data.
* Add more cluster validity indices.
