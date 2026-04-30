# -*- coding: utf-8 -*-
"""
Plot the maximum crosstalk matrix from a MATLAB .mat file.

This script is written as a direct-run version for GitHub release.
Users only need to modify the configuration variables in the
"User settings" section and then run the script directly.

Example:
    python MainFig4_2_direct.py

Expected .mat file format:
    The .mat file should contain a matrix dataset, for example named "crost".
    The matrix is visualized as a heatmap with mode labels on both axes.

Author:
    Please replace this line with your preferred author information.
"""

from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt


# =============================================================================
# User settings
# =============================================================================

# Path to the MATLAB .mat file.
# Use r"..." for Windows paths to avoid issues with backslashes.
MAT_FILE_PATH = r"E:\文章3\Fig\Fig4\fig4_git/crost.mat"

# Dataset name inside the .mat file.
DATASET_NAME = "crost"

# Whether to transpose the loaded matrix.
# MATLAB/HDF5 files sometimes require transposition to match the intended layout.
TRANSPOSE_MATRIX = True

# Labels shown on the x- and y-axes.
LABELS = ["-4", "-3", "-2", "-1", "0", "1", "2", "3"]

# Figure title. Set to None if no title is needed.
FIGURE_TITLE = None

# Output figure path. Use None if you do not want to save the figure.
SAVE_FIGURE_PATH = "maxcrosstalk.eps"

# Figure resolution for saved output.
SAVE_DPI = 600

# Colormap used for the heatmap.
COLORMAP = plt.cm.Greens

# Font settings.
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 10

# Annotation settings.
# For crosstalk in dB, negative values are usually important.
# Only values smaller than ANNOTATION_THRESHOLD will be printed on the heatmap.
ANNOTATION_THRESHOLD = -0.005
ANNOTATION_FORMAT = ".2f"

# Whether to show a colorbar.
SHOW_COLORBAR = False

# Whether to display the figure window after saving.
SHOW_FIGURE = True


# =============================================================================
# Functions
# =============================================================================

def load_mat_matrix(mat_file_path, dataset_name, transpose=True):
    """
    Load a matrix dataset from a MATLAB v7.3 .mat file.

    Parameters
    ----------
    mat_file_path : str or pathlib.Path
        Path to the .mat file.
    dataset_name : str
        Name of the dataset stored in the .mat file.
    transpose : bool, optional
        If True, transpose the loaded matrix. The default is True.

    Returns
    -------
    numpy.ndarray
        Loaded matrix.
    """
    mat_file_path = Path(mat_file_path)

    if not mat_file_path.exists():
        raise FileNotFoundError(f"The .mat file was not found: {mat_file_path}")

    with h5py.File(mat_file_path, "r") as mat_file:
        if dataset_name not in mat_file:
            available_keys = list(mat_file.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' was not found in {mat_file_path}. "
                f"Available datasets are: {available_keys}"
            )
        matrix = np.array(mat_file[dataset_name])

    if transpose:
        matrix = matrix.T

    return matrix


def print_matrix(matrix):
    """
    Print the matrix values in a tab-separated format.

    This is useful for quickly copying the numerical values to Excel or checking
    whether the loaded matrix is correct.
    """
    string_matrix = matrix.astype(str).tolist()
    for row in string_matrix:
        print("\t".join(row))


def validate_matrix_and_labels(matrix, labels):
    """
    Validate whether the matrix is square and matches the number of labels.
    """
    if matrix.ndim != 2:
        raise ValueError(f"The input data must be a 2D matrix, but got shape {matrix.shape}.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"The matrix must be square, but got shape {matrix.shape}.")

    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"The number of labels ({len(labels)}) must match the matrix size "
            f"({matrix.shape[0]})."
        )


def plot_crosstalk_matrix(
    matrix,
    labels,
    title=None,
    cmap=plt.cm.Greens,
    save_path=None,
    save_dpi=600,
    font_family="Times New Roman",
    font_size=10,
    annotation_threshold=-0.005,
    annotation_format=".2f",
    show_colorbar=False,
    show_figure=True,
):
    """
    Plot a crosstalk matrix as a heatmap.

    Parameters
    ----------
    matrix : numpy.ndarray
        Crosstalk matrix to visualize. For crosstalk in dB, values are often
        negative, where more negative values indicate lower crosstalk.
    labels : list of str
        Axis labels, such as OAM mode indices.
    title : str or None, optional
        Figure title.
    cmap : matplotlib colormap, optional
        Colormap used for the heatmap.
    save_path : str or None, optional
        Path used to save the figure. If None, the figure is not saved.
    save_dpi : int, optional
        Resolution of the saved figure.
    font_family : str, optional
        Font family used in the plot.
    font_size : int, optional
        Font size used in the plot.
    annotation_threshold : float, optional
        Only matrix values smaller than this threshold are annotated.
    annotation_format : str, optional
        Format used for text annotations.
    show_colorbar : bool, optional
        Whether to add a colorbar.
    show_figure : bool, optional
        Whether to display the figure window.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    matplotlib.axes.Axes
        The generated axes object.
    """
    validate_matrix_and_labels(matrix, labels)

    plt.rc("font", family=font_family, size=font_size)

    # This value is kept from the original script as a quick summary metric.
    # For a true confusion matrix, this corresponds to accuracy. For a crosstalk
    # matrix, it should be interpreted only as a diagonal-to-total ratio.
    diagonal_ratio = np.trace(matrix) / float(np.sum(matrix)) if np.sum(matrix) != 0 else np.nan

    fig, ax = plt.subplots()
    image = ax.imshow(matrix, interpolation="nearest", cmap=cmap)

    if show_colorbar:
        fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel=f"Predicted label\ndiagonal ratio = {diagonal_ratio:0.4f}",
    )

    # Draw grid lines to separate individual cells.
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add numerical annotations. This is especially useful for crosstalk values
    # in dB, where selected negative values are highlighted.
    threshold_for_text_color = matrix.max() / 2.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if value < annotation_threshold:
                ax.text(
                    col_idx,
                    row_idx,
                    format(value, annotation_format),
                    ha="center",
                    va="center",
                    color="white" if value > threshold_for_text_color else "black",
                )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=save_dpi, format=save_path.suffix.replace(".", ""))
        print(f"Figure saved to: {save_path.resolve()}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# =============================================================================
# Main program
# =============================================================================

if __name__ == "__main__":
    crosstalk_matrix = load_mat_matrix(
        MAT_FILE_PATH,
        DATASET_NAME,
        transpose=TRANSPOSE_MATRIX,
    )

    print("Loaded matrix:")
    print_matrix(crosstalk_matrix)

    plot_crosstalk_matrix(
        crosstalk_matrix,
        LABELS,
        title=FIGURE_TITLE,
        cmap=COLORMAP,
        save_path=SAVE_FIGURE_PATH,
        save_dpi=SAVE_DPI,
        font_family=FONT_FAMILY,
        font_size=FONT_SIZE,
        annotation_threshold=ANNOTATION_THRESHOLD,
        annotation_format=ANNOTATION_FORMAT,
        show_colorbar=SHOW_COLORBAR,
        show_figure=SHOW_FIGURE,
    )
