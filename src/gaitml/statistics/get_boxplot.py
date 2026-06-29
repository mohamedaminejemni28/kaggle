import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_boxplot(x1, x2, varname, group=('Young', 'Older'), save=False, folder_name=None):
    """
    Plot a boxplot comparing two groups and optionally save the figure.

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
    x = np.concatenate([x1, x2])
    g = np.array([group[0]] * len(x1) + [group[1]] * len(x2))

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=g, y=x)
    plt.title(varname)
    plt.xlabel('Group')
    plt.ylabel(varname)
    plt.tight_layout()

    if save:
        if folder_name is not None:
            os.makedirs(folder_name, exist_ok=True)
            save_path = os.path.join(folder_name, f"{varname}_stats.png")
        else:
            save_path = f"{varname}_stats.png"
        plt.savefig(save_path, dpi=100)
        print(f"Boxplot saved to {save_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Example usage
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(1, 1, 100)
    get_boxplot(x1, x2, 'Example Variable', save=True, folder_name='plots') 