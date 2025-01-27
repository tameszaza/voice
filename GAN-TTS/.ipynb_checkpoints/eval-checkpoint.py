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

from sklearn.metrics import precision_recall_curve

plt.rcParams.update({
    'font.size': 12,       # Default font size for all text
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 14,  # Label font size
    'legend.fontsize': 12, # Legend font size
})

def preprocess_test_data(dataset_dir, condition_window, sample_window, upsample_factor, max_clips_per_class=None):
    """
    Preprocess the test datasets from real and fake folders.
    Limits the total number of audio clips per class to `max_clips_per_class` (if specified).
    Returns preprocessed data and labels.
    """
    data = []
    labels = []

    for label, folder in enumerate(["real",  "fake"]):
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
    labels = [1 - label for label in labels]

    print(f"Total preprocessed samples: {len(data)}")  # Debugging
    return data, labels







def plot_metrics_vs_threshold(all_probs, all_targets):
    """
    Plot Precision, Recall, F1, and Accuracy vs. Threshold for Class 0.
    """
    thresholds = np.linspace(0, 1, 100)  # Define thresholds from 0 to 1
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)

        TP = ((preds == 0) & (all_targets == 0)).sum()
        TN = ((preds == 1) & (all_targets == 1)).sum()
        FP = ((preds == 0) & (all_targets == 1)).sum()
        FN = ((preds == 1) & (all_targets == 0)).sum()

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Precision (Class 0)", color="blue")
    plt.plot(thresholds, recalls, label="Recall (Class 0)", color="orange")
    plt.plot(thresholds, f1_scores, label="F1 Score (Class 0)", color="green")
    plt.plot(thresholds, accuracies, label="Accuracy", color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Metrics vs. Threshold (Class 0)")
    plt.legend(loc="lower left")
    plt.grid()
    plt.tight_layout()
    plt.savefig("./roc_plots/metrics_vs_threshold_class_0.png")
    plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def evaluate_discriminator(discriminator, test_data, test_labels, device):
    """
    Evaluate the discriminator on test datasets for Class 0, including ROC Curve.
    """
    output_dir = "./roc_plots"
    os.makedirs(output_dir, exist_ok=True)

    all_probs = []
    all_targets = []

    # Evaluate each test sample
    for (audio, mel), label in tqdm(zip(test_data, test_labels), desc="Evaluating", total=len(test_data)):
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).permute(0, 2, 1).to(device)

        with torch.no_grad():
            output = discriminator(audio_tensor)
        
        prob = output.mean(dim=-1).item()
        all_probs.append(prob)
        all_targets.append(label)

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Treat Class 0 as the positive class
    # If probabilities are inverted, use this logic:
    preds = (all_probs > 0.5).astype(int)  # Class 0 as positive

    # Calculate TP, TN, FP, FN
    TP = ((preds == 0) & (all_targets == 0)).sum()  # True Positives (Class 0 correctly identified)
    TN = ((preds == 1) & (all_targets == 1)).sum()  # True Negatives (Class 1 correctly identified)
    FP = ((preds == 0) & (all_targets == 1)).sum()  # False Positives (Class 1 misclassified as Class 0)
    FN = ((preds == 1) & (all_targets == 0)).sum()  # False Negatives (Class 0 misclassified as Class 1)

    # Debugging print statements
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    # Calculate metrics
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

    # Display results
    rows = [
        ["Accuracy", f"{accuracy:.4f}"],
        ["Precision (Class 0)", f"{precision:.4f}"],
        ["Recall (Class 0)", f"{recall:.4f}"],
        ["F1 Score (Class 0)", f"{f1:.4f}"]
    ]
    print("\nEvaluation Metrics for Class 0:")
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="grid"))

    # Compute ROC Curve for Class 0
    fpr, tpr, thresholds = roc_curve(all_targets, 1 - all_probs, pos_label=0)  # Invert probabilities for Class 0
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Class 0")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_class_0.png"))
    plt.show()

    # Plot metrics vs. threshold
    plot_metrics_vs_threshold(all_probs, all_targets)



def calculate_reverse_roc(all_probs, all_targets):
    """
    Calculate Reverse ROC points by swapping the target classes.
    """
    # Reverse the classes
    reversed_targets = 1 - all_targets  # Swap 0 <-> 1
    fpr, tpr, thresholds = roc_curve(reversed_targets, 1 - all_probs)  # Flip probabilities too
    return fpr, tpr



def main():
    dataset_dir = "../data_test_2"  # Root directory containing real/ and fake/ folders
    # checkpoint_path = "logdir2/mgan_step_930000.pth"
    checkpoint_path = "logdir2/mgan_step_1780000.pth"
    # checkpoint_path = "logdir/model.ckpt-205000.pt"
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

