import torch
import os
import numpy as np
from utils.audio import convert_audio
from models.v2_discriminator import Discriminator
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def preprocess_test_data(dataset_dir, condition_window, sample_window, upsample_factor, max_clips_per_class=None):
    """
    Preprocess the test datasets from real and fake folders.
    Limits the total number of audio clips per class to `max_clips_per_class` (if specified).
    Returns preprocessed data and labels.
    """
    data = []
    labels = []

    for label, folder in enumerate([ "fake", "real"]):
        folder_path = os.path.join(dataset_dir, folder)
        audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

        # Limit the number of audio clips if specified
        if max_clips_per_class is not None:
            audio_files = audio_files[:max_clips_per_class]

        print(f"Processing {len(audio_files)} files from {folder}")

        for audio_file in tqdm(audio_files, desc=f"Preprocessing {folder.capitalize()} Data"):
            mel, audio = convert_audio(audio_file)

            # Ensure the audio length is consistent with `sample_window`
            if len(audio) < sample_window:
                audio = np.pad(audio, (0, sample_window - len(audio)), mode='constant')
            else:
                audio = audio[:sample_window]

            # Ensure the mel spectrogram is consistent with `condition_window`
            if len(mel) < condition_window:
                mel = np.pad(mel, ((0, condition_window - len(mel)), (0, 0)), mode='constant')
            else:
                mel = mel[:condition_window]

            data.append((audio, mel))
            labels.append(label)

    print(f"Total preprocessed samples: {len(data)}")  # Debugging
    return data, labels




def evaluate_discriminator(discriminator, test_data, test_labels, device):
    """
    Evaluate the discriminator on the test datasets and print detailed metrics, including TP, TN, FP, FN.
    Additionally, plot the ROC curve and calculate AUC.
    """

    output_dir = "./roc_plots"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    plot_path = os.path.join(output_dir, "roc_curve.png")
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1_score = BinaryF1Score().to(device)

    all_preds = []
    all_probs = []  # For storing probabilities
    all_targets = []

    for (audio, mel), label in tqdm(zip(test_data, test_labels), desc="Evaluating", total=len(test_data)):
        # Convert to PyTorch tensors and move to the device
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, sample_window]
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).permute(0, 2, 1).to(device)  # [1, condition_dim, condition_window]

        # Get discriminator output
        with torch.no_grad():
            output = discriminator(audio_tensor)  # Output shape: [1, 1, feature_dim]

        # Reduce feature_dim (e.g., take the mean over the last dimension)
        prob = output.mean(dim=-1).item()  # Single probability
        pred = (prob > 0.5)  # Binary prediction

        # Append probabilities, predictions, and targets
        all_probs.append(prob)
        all_preds.append(pred)
        all_targets.append(label)

    # Convert lists to numpy arrays for compatibility with sklearn
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate ROC AUC
    auc_score = roc_auc_score(all_targets, all_probs)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    plt.savefig(plot_path)  # Save the plot to the directory
    print(f"ROC curve saved to {plot_path}")

    # Calculate TP, TN, FP, FN
    TP = ((all_preds == 1) & (all_targets == 1)).sum()
    TN = ((all_preds == 0) & (all_targets == 0)).sum()
    FP = ((all_preds == 1) & (all_targets == 0)).sum()
    FN = ((all_preds == 0) & (all_targets == 1)).sum()

    # Calculate metrics
    total_accuracy = accuracy(torch.tensor(all_preds), torch.tensor(all_targets)).item()
    total_precision = precision(torch.tensor(all_preds), torch.tensor(all_targets)).item()
    total_recall = recall(torch.tensor(all_preds), torch.tensor(all_targets)).item()
    total_f1 = f1_score(torch.tensor(all_preds), torch.tensor(all_targets)).item()

    # Display results in a table
    headers = ["Metric", "Value"]
    rows = [
        ["True Positives (TP)", TP],
        ["True Negatives (TN)", TN],
        ["False Positives (FP)", FP],
        ["False Negatives (FN)", FN],
        ["Accuracy", f"{total_accuracy:.4f}"],
        ["Precision", f"{total_precision:.4f}"],
        ["Recall", f"{total_recall:.4f}"],
        ["F1 Score", f"{total_f1:.4f}"],
        ["ROC AUC", f"{auc_score:.4f}"]
    ]
    print("\nEvaluation Results:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    dataset_dir = "../data_eval"  # Root directory containing real/ and fake/ folders
    checkpoint_path = "logdir2/mgan_step_1340000.pth"
    condition_window = 100  # Same as used in training
    upsample_factor = 120
    sample_window = condition_window * upsample_factor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the discriminator
    discriminator = Discriminator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    discriminator.load_state_dict(checkpoint["discriminator"])
    discriminator.eval()

    # Preprocess test data
    test_data, test_labels = preprocess_test_data(dataset_dir, condition_window, sample_window, upsample_factor, 3000)

    # Evaluate the discriminator
    evaluate_discriminator(discriminator, test_data, test_labels, device)

if __name__ == "__main__":
    main()

