import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3d projection

def load_mel_features(base_dir):
    """
    Walk through each generator folder in base_dir, load all .npy files
    in the 'mel' subdirectory, flatten them to 1D vectors, and return:
      - X: array of shape (n_samples, n_features)
      - labels: list of generator names, length n_samples
    """
    X = []
    labels = []

    for gen_name in sorted(os.listdir(base_dir)):
        gen_path = os.path.join(base_dir, gen_name)
        mel_dir = os.path.join(gen_path, 'mel')
        if not os.path.isdir(mel_dir):
            continue

        for fname in sorted(os.listdir(mel_dir)):
            if not fname.endswith('.npy'):
                continue
            file_path = os.path.join(mel_dir, fname)
            try:
                mel = np.load(file_path)    # shape: [freq_bins, time_frames]
            except Exception as e:
                print(f"Warning: could not load {file_path}: {e}")
                continue

            vec = mel.reshape(-1)         # flatten
            X.append(vec)
            labels.append(gen_name)

    X = np.stack(X, axis=0)
    return X, labels

def plot_pca_3d_and_save(X, labels, output_path, n_components=3):
    """
    Fit PCA with n_components=3 on X, create a 3D scatter of the first
    three principal components, colour points by label, and save to output_path.
    """
    # fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # prepare figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # assign each generator a distinct colour index
    unique_labels = sorted(set(labels))
    color_map = {lab: idx for idx, lab in enumerate(unique_labels)}

    # scatter each group
    for lab in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(
            X_pca[idxs, 0],
            X_pca[idxs, 1],
            X_pca[idxs, 2],
            label=lab,
            alpha=0.7,
            s=20
        )

    ax.set_title('3D PCA of Mel-Spectrograms by Generator')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()

    # save and close
    plt.savefig(output_path, dpi=300)
    print(f"Saved 3D PCA plot to {output_path}")
    plt.close()
def plot_pca_and_save(X, labels, output_path, n_components=2):
    """
    Fit PCA on X and save a scatter plot of the first two components
    to output_path, coloring points by label.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    unique_labels = sorted(set(labels))
    plt.figure(figsize=(10, 8))

    for lab in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(
            X_pca[idxs, 0],
            X_pca[idxs, 1],
            label=lab,
            alpha=0.6,
            s=20
        )

    plt.title('PCA of Mel-Spectrograms by Generator')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    # replace plt.show() with saving to file
    plt.savefig(output_path, dpi=300)
    print(f"Saved PCA plot to {output_path}")
    plt.close()

if __name__ == '__main__':
    base_directory = 'data/data_32_test'
    print(f"Loading mel features from '{base_directory}' …")
    X, labels = load_mel_features(base_directory)
    print(f"Loaded {X.shape[0]} samples, each of length {X.shape[1]}")

    output_file = os.path.join(base_directory, 'pca_plot.png')
    print("Computing PCA and saving plot …")
    plot_pca_and_save(X, labels, output_file)
