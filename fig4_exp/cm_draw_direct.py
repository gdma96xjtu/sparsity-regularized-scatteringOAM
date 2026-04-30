# -*- coding: utf-8 -*-
"""
Plot a confusion/crosstalk matrix from a MATLAB .mat file.

This script is designed for simple direct execution. You only need to modify
several parameters in the USER SETTINGS section below, then run the script.

Example:
    python cm_draw_direct.py

Required packages:
    numpy
    matplotlib
    h5py

Author:
    Please replace this line with your name if needed.
"""

from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt


# =============================================================================
# USER SETTINGS
# Modify the following parameters according to your own data.
# =============================================================================

# Path to the MATLAB .mat file.
# Use r"..." for Windows paths to avoid escape-character problems.
MAT_FILE_PATH = r"E:\文章3\Fig\Fig4\fig4_git/cm.mat"

# Dataset name inside the .mat file.
# In the original file, the crosstalk/confusion matrix is stored as 'cro'.
DATASET_NAME = "cro"

# Whether to transpose the matrix after loading.
# The original script used: cm = np.array(cm_data['cro']).T
TRANSPOSE_MATRIX = True

# Class labels shown on x-axis and y-axis.
LABELS = ["-4", "-3", "-2", "-1", "0", "1", "2", "3"]

# Figure title. Set to None if no title is needed.
FIGURE_TITLE = None

# Colormap used for matrix visualization.
COLORMAP = plt.cm.Greens

# Whether to display the colorbar.
SHOW_COLORBAR = False

# Threshold for displaying numerical values in matrix cells.
# Values smaller than this threshold will not be annotated.
ANNOTATION_THRESHOLD = 5e-5

# Output figure path. Set to None if you only want to display the figure.
SAVE_FIGURE_PATH = "crosstalk_matrix.png"

# Figure resolution when saving.
SAVE_DPI = 600

# Whether to show the figure window after plotting.
SHOW_FIGURE = True


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_matrix_from_mat(mat_file_path, dataset_name, transpose=True):
    """
    Load a matrix from a MATLAB .mat file saved in HDF5 format.

    Parameters
    ----------
    mat_file_path : str or pathlib.Path
        Path to the .mat file.
    dataset_name : str
        Dataset name inside the .mat file.
    transpose : bool, optional
        If True, transpose the loaded matrix. The default is True.

    Returns
    -------
    numpy.ndarray
        Loaded matrix as a NumPy array.
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

    This is useful for copying the matrix into Excel or checking the numerical
    values before plotting.
    """
    str_matrix = matrix.astype(str).tolist()
    for row in str_matrix:
        print("\t".join(row))


def plot_matrix(
    matrix,
    labels,
    title=None,
    cmap=plt.cm.Greens,
    show_colorbar=False,
    annotation_threshold=5e-5,
    save_path=None,
    save_dpi=600,
    show_figure=True,
):
    """
    Plot a confusion/crosstalk matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Two-dimensional matrix to be visualized.
    labels : list of str
        Class labels for x-axis and y-axis.
    title : str or None, optional
        Figure title.
    cmap : matplotlib.colors.Colormap, optional
        Colormap used for imshow.
    show_colorbar : bool, optional
        Whether to show the colorbar.
    annotation_threshold : float, optional
        Only values larger than this threshold will be annotated.
    save_path : str or None, optional
        Path for saving the figure. If None, the figure will not be saved.
    save_dpi : int, optional
        Resolution of the saved figure.
    show_figure : bool, optional
        Whether to display the figure window.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    matplotlib.axes.Axes
        The generated axes object.
    """
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("The input matrix must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The input matrix must be square.")

    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"The number of labels ({len(labels)}) does not match "
            f"the matrix size ({matrix.shape[0]})."
        )

    plt.rc("font", family="Times New Roman", size=10)

    accuracy = np.trace(matrix) / float(np.sum(matrix))

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
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
        xlabel=f"Predicted label\naccuracy = {accuracy:.4f}",
    )

    # Draw grid lines to show the boundary of each cell.
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate matrix values.
    threshold_for_text_color = matrix.max() / 2.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if value > annotation_threshold:
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    color="white" if value > threshold_for_text_color else "black",
                )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    cm = load_matrix_from_mat(
        mat_file_path=MAT_FILE_PATH,
        dataset_name=DATASET_NAME,
        transpose=TRANSPOSE_MATRIX,
    )

    print("Loaded matrix:")
    print_matrix(cm)

    plot_matrix(
        matrix=cm,
        labels=LABELS,
        title=FIGURE_TITLE,
        cmap=COLORMAP,
        show_colorbar=SHOW_COLORBAR,
        annotation_threshold=ANNOTATION_THRESHOLD,
        save_path=SAVE_FIGURE_PATH,
        save_dpi=SAVE_DPI,
        show_figure=SHOW_FIGURE,
    )
