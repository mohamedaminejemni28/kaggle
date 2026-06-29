import os
import numpy as np
import matplotlib.pyplot as plt

def get_histogram_groups(x1, x2, varname, group=('Young', 'Older'), save=False, folder_name=None):
    """
    Plot overlaid histograms for two groups and optionally save the figure.

    Parameters
    ----------
    x1 : array-like
        Data for group 1 (e.g., 'Young').
    x2 : array-like
        Data for group 2 (e.g., 'Older').
    varname : str
        Variable name for the plot title and filename.
    group : tuple of str, optional
        Names of the two groups (default: ('Young', 'Older')).
    save : bool, optional
        Whether to save the figure (default: False).
    folder_name : str, optional
        Folder to save the figure in (default: None).
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    plt.figure(figsize=(6, 4))
    plt.hist(x1, bins=30, alpha=0.7, label=group[0], edgecolor='none')
    plt.hist(x2, bins=30, alpha=0.3, label=group[1], edgecolor='none')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend(group)
    plt.title(varname)
    plt.tight_layout()

    if save:
        if folder_name is not None:
            os.makedirs(folder_name, exist_ok=True)
            save_path = os.path.join(folder_name, f"{varname}_hist.png")
        else:
            save_path = f"{varname}_hist.png"
        plt.savefig(save_path, dpi=100)
        print(f"Histogram saved to {save_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Example usage
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(1, 1, 100)
    get_histogram_groups(x1, x2, 'Example Variable', save=True, folder_name='plots') 