import os
import logging
from tqdm import tqdm
import numpy as np
import fitz  # PyMuPDF
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
import pickle

import warnings
from sklearn.exceptions import ConvergenceWarning



from classify.util import (
    load_df_from_pickle,
    load_vectorizer,
    load_np_array_from_pickle,
    load_model,
)

logger = logging.getLogger(__name__)


def check_aspect_ratio_and_mix_feature(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Extract Features. Error opening PDF {pdf_path}: {e}")
        raise ValueError("Error opening PDF")
        return (
            [],
            0,
            False,
        )  # Returning False as the third value for no significant change

    aspect_ratios = []

    page_count = len(doc)

    for page in doc:
        rect = page.rect
        aspect_ratio = rect.width / rect.height
        aspect_ratios.append(aspect_ratio)

    # Detect significant changes in aspect ratio
    change_frequency = calculate_change_frequency(aspect_ratios)

    persistent_changes_raw, persistent_changes_frequency = detect_persistent_changes(
        aspect_ratios, change_frequency
    )
    num_changes, changes_significance = detect_significant_change(aspect_ratios)
    stats = extract_aspect_ratio_features(aspect_ratios)
    categories = categorize_aspect_ratios(aspect_ratios)

    return (
        aspect_ratios,
        page_count,
        persistent_changes_raw,
        persistent_changes_frequency,
        num_changes,
        changes_significance,
        stats,
        categories,
    )


def calculate_change_frequency(aspect_ratios, threshold=0.1):
    """
    Calculate the percentage of pages exceeding a given aspect ratio change threshold.

    :param aspect_ratios: List of aspect ratios for the document's pages.
    :param threshold: Threshold for considering a change in aspect ratio significant.
    :return: Percentage of page transitions that exceed the change threshold.
    """
    if len(aspect_ratios) < 2:
        # If there's only one page or none, there can't be any transitions
        return 0.0

    # Calculate the absolute percentage change in aspect ratio between consecutive pages
    changes = [
        abs(aspect_ratios[i] - aspect_ratios[i - 1]) / aspect_ratios[i - 1]
        for i in range(1, len(aspect_ratios))
    ]

    # Count how many changes exceed the threshold
    significant_changes = sum(change > threshold for change in changes)

    # Calculate the percentage of transitions that are significant
    change_frequency = (significant_changes / (len(aspect_ratios) - 1)) * 100

    return change_frequency


def detect_significant_change(aspect_ratios, change_threshold=0.1):
    """
    Detects the number of significant aspect ratio changes and assesses their significance.

    :param aspect_ratios: List of aspect ratios for each page in the document.
    :param change_threshold: Threshold for considering a change in aspect ratio significant, based on change_frequency.
    :return: A tuple containing the number of significant changes and an aggregate measure of their significance.
    """
    if len(aspect_ratios) < 2:
        # No significant changes can be detected in a single-page document
        return 0, 0.0

    # Calculate percentage changes between consecutive aspect ratios
    changes = np.abs(np.diff(aspect_ratios) / aspect_ratios[:-1])

    # Determine significant changes based on the threshold
    significant_changes_indices = np.where(changes > change_threshold)[0]
    num_significant_changes = len(significant_changes_indices)

    # Calculate the significance of changes as the sum of changes that exceed the threshold, normalized by the number of changes
    if num_significant_changes > 0:
        significance_of_changes = (
            np.sum(changes[significant_changes_indices]) / num_significant_changes
        )
    else:
        significance_of_changes = 0.0

    return num_significant_changes, significance_of_changes


def detect_persistent_changes(aspect_ratios, change_threshold=0.1):
    """
    Detects persistent changes in aspect ratios and adjusts the count based on change frequency.

    :param aspect_ratios: List of aspect ratios for each page in the document.
    :param change_threshold: Threshold for considering a change in aspect ratio significant.
    :return: A tuple containing the raw count of persistent changes and the count adjusted by change frequency.
    """
    if len(aspect_ratios) < 2:
        # If there's only one page or none, there can't be any transitions
        return 0, 0.0

    # Calculate the absolute percentage change in aspect ratio between consecutive pages
    changes = [
        abs(aspect_ratios[i] - aspect_ratios[i - 1]) / aspect_ratios[i - 1]
        for i in range(1, len(aspect_ratios))
    ]

    # Count the number of persistent changes
    persistent_changes_raw = 0
    current_persistence = 0

    for change in changes:
        if change > change_threshold:
            current_persistence += 1
        else:
            if (
                current_persistence > 1
            ):  # Assuming persistence means more than one consecutive change
                persistent_changes_raw += 1
            current_persistence = 0

    # Check if the last sequence of pages ends with persistent changes
    if current_persistence > 1:
        persistent_changes_raw += 1

    # Calculate change_frequency to adjust the persistent_changes_raw
    change_frequency = calculate_change_frequency(aspect_ratios, change_threshold)

    # Adjusting the raw count by change frequency to get a frequency-informed measure
    # This example simply scales the raw count by the change_frequency percentage; other methods could also be applied
    persistent_changes_frequency = persistent_changes_raw * (change_frequency / 100)

    return persistent_changes_raw, persistent_changes_frequency


def extract_aspect_ratio_features(aspect_ratios):
    """
    Extracts statistical features from a list of aspect ratios.

    :param aspect_ratios: List of aspect ratios for the document's pages.
    :return: Dictionary of statistical features.
    """
    if not aspect_ratios:  # Check if the list is empty
        return {"mean": 0, "std": 0, "min": 0, "max": 0}

    aspect_ratios_array = np.array(aspect_ratios)
    return {
        "mean": np.mean(aspect_ratios_array),
        "std": np.std(aspect_ratios_array),
        "min": np.min(aspect_ratios_array),
        "max": np.max(aspect_ratios_array),
    }


def categorize_aspect_ratios(aspect_ratios):
    """
    Categorizes aspect ratios as portrait, landscape, or square.

    :param aspect_ratios: List of aspect ratios for the document's pages.
    :return: List of categories corresponding to each aspect ratio.
    """
    categories = []
    for ar in aspect_ratios:
        if ar < 0.95:
            categories.append("portrait")
        elif ar > 1.05:
            categories.append("landscape")
        else:
            categories.append("square")
    return categories


def calculate_clustering_features(aspect_ratios, n_clusters=3):
    """
    Performs KMeans clustering on aspect ratios and calculates clustering-based features.

    :param aspect_ratios: List of aspect ratios for the document's pages.
    :param n_clusters: Number of clusters to form, default is 3.
    :return: A dictionary with the calculated features: CDC, CTI, CDS, and MCP.
    """
    # Ensure aspect_ratios is a 2D array for KMeans
    try:
        aspect_ratios = np.array(aspect_ratios).reshape(-1, 1)

        # Perform KMeans clustering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)

            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(aspect_ratios)

        labels = kmeans.labels_

        # Calculate Cluster Diversity Count (CDC)
        unique_clusters = len(set(labels))

        # Calculate Cluster Transition Indicator (CTI)
        transitions = sum(
            1 for i in range(1, len(labels)) if labels[i] != labels[i - 1]
        )

        # Calculate Cluster Distribution Spread (CDS) using entropy
        cluster_counts = np.bincount(labels, minlength=n_clusters)
        proportions = cluster_counts / np.sum(cluster_counts)
        spread = entropy(proportions)

        # Calculate Majority Cluster Proportion (MCP)
        majority_cluster_proportion = max(cluster_counts) / np.sum(cluster_counts)
    except Exception as e:
        print(f"Error calculating clustering features: {e}")
        return {
            "CDC": np.nan,
            "CTI": np.nan,
            "CDS": np.nan,
            "MCP": np.nan,
        }  # Return an empty values if an error occurs

    # Return the calculated features
    return {
        "CDC": unique_clusters,
        "CTI": transitions,
        "CDS": spread,
        "MCP": majority_cluster_proportion,
    }


def detect_outliers_z_score(aspect_ratios, threshold=2):
    aspect_ratios = np.array(aspect_ratios).flatten()  # Ensures aspect_ratios is 1D
    mean_ar = np.mean(aspect_ratios)
    std_ar = np.std(aspect_ratios)
    outliers = [
        i
        for i, ar in enumerate(aspect_ratios)
        if abs((ar - mean_ar) / std_ar) > threshold
    ]

    return outliers


np.seterr(divide="ignore", invalid="ignore")


def calculate_text_density(pdf_path):
    """
    Calculates text density (words per page) for a PDF document.

    :param pdf_path: Path to the PDF document.
    :return: List of text densities for each page.
    """
    doc = fitz.open(pdf_path)
    text_densities = []
    for page in doc:
        text = page.get_text("text")
        word_count = len(text.split())
        area = page.rect.width * page.rect.height
        text_density = word_count / area if area else 0
        text_densities.append(text_density)
    return text_densities


def correlate_text_density_aspect_ratio(aspect_ratios, text_densities):
    """
    Calculates the Pearson correlation coefficient between aspect ratios and text densities of a document's pages.

    :param aspect_ratios: List of aspect ratios for each page.
    :param text_densities: List of text densities for each page.
    :return: Pearson correlation coefficient, or NaN if the calculation is not possible.
    """
    if len(aspect_ratios) != len(text_densities) or len(aspect_ratios) < 2:
        return np.nan  # Ensures there are enough data points and both lists match

    return np.corrcoef(aspect_ratios, text_densities)[0, 1]


def calculate_text_density_variability(text_densities):
    return np.std(text_densities)


def calculate_text_density_by_position(text_densities):
    """
    Calculates the average text density for the beginning, middle, and end sections of a document.

    :param text_densities: List of text densities for each page.
    :return: A tuple with average text densities for the beginning, middle, and end of the document.
    """
    third = len(text_densities) // 3
    if third == 0:
        return (0, 0, 0)  # Avoid division by zero for very short documents

    beginning = np.mean(text_densities[:third])
    middle = np.mean(text_densities[third: 2 * third])
    end = np.mean(text_densities[2 * third:])

    return beginning, middle, end


# Function to check for keyword presence
def check_keywords(text, keyword_list):
    text = text.lower()
    return int(any(keyword in text for keyword in keyword_list))


def combine_tfidf_keyword(df):
    # Step 2: TF-IDF Calculation
    vectorizer = TfidfVectorizer(
        max_features=5000
    )  # Adjust number of features as needed
    tfidf_matrix = vectorizer.fit_transform(df["tokenized_text"])

    # Step 3. Combine keyword and tfidf features into a single matrix
    # Convert binary keyword matches to a matrix
    keyword_features = df[[col for col in df.columns if "_keyword" in col]].to_numpy()
    # Combine TF-IDF features with keyword binary indicators
    combined_features = np.hstack((tfidf_matrix.toarray(), keyword_features))

    # Now `combined_features` is ready for model training, and should be aligned with your labels.
    return combined_features


# %%


# %%
def combine_tfidf_keyword_additional_features(df, vectorizer=None):
    # Step 2: TF-IDF Calculation
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df["tokenized_text"])
    else:
        tfidf_matrix = vectorizer.transform(df["tokenized_text"])

    # Convert binary keyword matches to a matrix
    keyword_features = df[[col for col in df.columns if "_keyword" in col]].to_numpy()

    # Assuming new features are already in df and are numeric
    additional_features_columns = [
        "aspect_ratio_means",
        "aspect_ratio_std",
        "aspect_ratio_min",
        "aspect_ratio_max",
        "page_counts",
        "persistent_changes_raw",
        "persistent_changes_frequency",
        "num_changes",
        "changes_significance",
        "text_density_means",
        "text_density_correlations",
        "text_density_variability",
        "text_density_beginning",
        "text_density_middle",
        "text_density_end",
        "outliers_counts",
        "unique_cluster_lists",
        "cluster_transitions_lists",
        "cluster_spreads_lists",
        "majority_cluster_proportions",
        "portrait_count",
        "landscape_count",
        "square_count",
    ]
    additional_features = df[additional_features_columns].to_numpy()

    # Combine TF-IDF features with keyword binary indicators and the additional features
    combined_features = np.hstack(
        (tfidf_matrix.toarray(), keyword_features, additional_features)
    )

    return combined_features, vectorizer


def append_data_or_nan(a_list, data):
    try:
        a_list.append(data)
    except Exception:
        # logger.info(e)
        a_list.append(np.nan)


def extract_specific_features(df):
    # Initialize empty lists to store your new features
    aspect_ratio_means = []
    aspect_ratio_std = []
    aspect_ratio_min = []
    aspect_ratio_max = []
    page_counts = []
    persistent_changes_raw_list = []
    persistent_changes_frequency_list = []
    num_changes_list = []
    changes_significance_list = []
    text_density_means = []
    text_density_correlations = []
    text_density_variability_lists = []
    text_density_beginning_lists = []
    text_density_middle_lists = []
    text_density_end_lists = []
    categories_lists = []
    portrait_counts = []
    landscape_counts = []
    square_counts = []
    outliers_counts = []
    unique_cluster_lists = []
    cluster_transitions_lists = []
    cluster_spreads_lists = []
    majority_cluster_proportions = []

    for pdf_path in tqdm(df["fname"], desc="Processing PDFs"):
        try:
            # logger.info(f"Processing aspect stuff for {pdf_path}")
            # Run your feature extraction functions
            (
                aspect_ratios,
                page_count,
                persistent_changes_raw,
                persistent_changes_frequency,
                num_changes,
                changes_significance,
                stats,
                categories,
            ) = check_aspect_ratio_and_mix_feature(pdf_path)

            text_densities = calculate_text_density(pdf_path)
            correlation = correlate_text_density_aspect_ratio(
                aspect_ratios, text_densities
            )
            text_density_variability = calculate_text_density_variability(
                text_densities
            )
            (text_density_beginning, text_density_middle, text_density_end) = (
                calculate_text_density_by_position(text_densities)
            )

            outliers = detect_outliers_z_score(aspect_ratios)
            clusters = calculate_clustering_features(aspect_ratios)

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")

            # For simplicity, let's just use some of the features as examples
        append_data_or_nan(aspect_ratio_means, stats["mean"])
        append_data_or_nan(aspect_ratio_std, stats["std"])
        append_data_or_nan(aspect_ratio_min, stats["min"])
        append_data_or_nan(aspect_ratio_max, stats["max"])
        append_data_or_nan(page_counts, page_count)
        append_data_or_nan(persistent_changes_raw_list, persistent_changes_raw)
        append_data_or_nan(
            persistent_changes_frequency_list, persistent_changes_frequency
        )
        append_data_or_nan(num_changes_list, num_changes)
        append_data_or_nan(changes_significance_list, changes_significance)

        append_data_or_nan(text_density_means, np.mean(text_densities))
        append_data_or_nan(text_density_correlations, correlation)
        append_data_or_nan(text_density_variability_lists, text_density_variability)
        append_data_or_nan(text_density_beginning_lists, text_density_beginning)
        append_data_or_nan(text_density_middle_lists, text_density_middle)
        append_data_or_nan(text_density_end_lists, text_density_end)

        append_data_or_nan(
            categories_lists, categories
        )  # This one is a bit tricky as it's a list. Might aggregate or process further.

        append_data_or_nan(
            outliers_counts, len(outliers)
        )  # Assuming aspect ratios are recalculated within the function
        append_data_or_nan(unique_cluster_lists, clusters["CDC"])
        append_data_or_nan(cluster_transitions_lists, clusters["CTI"])
        append_data_or_nan(cluster_spreads_lists, clusters["CDS"])
        append_data_or_nan(majority_cluster_proportions, clusters["MCP"])

    portrait_counts = [
        cats.count("portrait") if isinstance(cats, list) else 0
        for cats in categories_lists
    ]
    landscape_counts = [
        cats.count("landscape") if isinstance(cats, list) else 0
        for cats in categories_lists
    ]
    square_counts = [
        cats.count("square") if isinstance(cats, list) else 0
        for cats in categories_lists
    ]

    # Now add these lists as columns to your DataFrame
    df["aspect_ratio_means"] = aspect_ratio_means
    df["aspect_ratio_std"] = aspect_ratio_std
    df["aspect_ratio_min"] = aspect_ratio_min
    df["aspect_ratio_max"] = aspect_ratio_max
    df["page_counts"] = page_counts
    df["persistent_changes_raw"] = persistent_changes_raw_list
    df["persistent_changes_frequency"] = persistent_changes_frequency_list
    df["num_changes"] = num_changes_list
    df["changes_significance"] = changes_significance_list
    df["text_density_means"] = text_density_means
    df["text_density_correlations"] = text_density_correlations
    df["text_density_variability"] = text_density_variability_lists
    df["text_density_beginning"] = text_density_beginning_lists
    df["text_density_middle"] = text_density_middle_lists
    df["text_density_end"] = text_density_end_lists
    df["outliers_counts"] = outliers_counts
    df["unique_cluster_lists"] = unique_cluster_lists
    df["cluster_transitions_lists"] = cluster_transitions_lists
    df["cluster_spreads_lists"] = cluster_spreads_lists
    df["majority_cluster_proportions"] = majority_cluster_proportions
    df["portrait_count"] = portrait_counts
    df["landscape_count"] = landscape_counts
    df["square_count"] = square_counts

    return df


keywords = {
    "financial_terms": [
        "financial",
        "investment",
        "share price",
        "financial metrics",
        "investment strategy",
    ],
    "legal_statements": [
        "confidentiality statement",
        "legal disclaimer",
        "disclosure statement",
        "proprietary information",
        "intellectual property",
    ],
    "company_info": [
        "company overview",
        "company analysis",
        "business model",
        "company performance",
    ],
    "presentation_content": [
        "visual aids",
        "data charts",
        "case studies",
        "comparative analysis",
    ],
    "company_targets": ["sales targets", "company targets", "performance targets"],
    "financial_discussions": [
        "financial figures",
        "financial projections",
        "financial results",
        "financial language",
    ],
    "regulatory_references": [
        "SEC filings",
        "regulatory filings",
        "external entities",
        "lawsuits",
    ],
    "detail_descriptions": [
        "loan details",
        "product details",
        "research and development",
        "financial details",
    ],
    "company_specific": [
        "company specific",
        "industry specific",
        "company-specific analysis",
        "specific company focus",
    ],
    # "Other Clusters" category is omitted since it's broad and without specific keywords
}


def load_from_disk(include_model=False):
    dfpickle_path = "/dave/data/df.pkl"
    if not os.path.exists(dfpickle_path):
        print(
            "Go back and do the preprocessing step above before running this cell for feature extraction"
        )
    else:
        print("loading PreProcessing df from disk")
        df = load_df_from_pickle(dfpickle_path)

    # %%
    dff_pickle_path = "/dave/data/df_features.pkl"
    features_path = "/dave/data/features_array.pkl.npy"
    features_array_ppath = "/dave/data/features_array.pkl"
    tdif_vectorizer_pickle_path = "/dave/data/tdif_vectorizer.pkl"

    force = False
    if (
        os.path.exists(features_path)
        and (os.path.exists(dff_pickle_path))
        and (os.path.exists(tdif_vectorizer_pickle_path))
        and not force
    ):
        print(
            "loading dataframe with features,features numpy array, and tdif vectorizer from disk)"
        )
        features = load_np_array_from_pickle(features_path)
        df = load_df_from_pickle(dff_pickle_path)
        tdif_vectorizer = load_vectorizer(tdif_vectorizer_pickle_path)
    else:
        df = extract_specific_features(df)
        # Apply keyword matching
        for category, keyword_list in keywords.items():
            df[category + "_keyword"] = df["tokenized_text"].apply(
                check_keywords, args=(keyword_list,)
            )
        features, tdif_vectorizer = combine_tfidf_keyword_additional_features(df)
        df.to_pickle(dff_pickle_path)
        np.save(features_array_ppath, features)
        pickle.dump(tdif_vectorizer, open(tdif_vectorizer_pickle_path, "wb"))

    if include_model:
        model = load_model(type=include_model)
        return df, features, tdif_vectorizer, model
    return df, features, tdif_vectorizer


if __name__ == "__main__":
    df, features, tdif_vectorizer, model = load_from_disk(include_model="hgb")
    assert df is not None
    assert features is not None
    assert tdif_vectorizer is not None
    assert model is not None
    print("Loaded ...")
