import torch
import os
import numpy as np
from utils.audio import convert_audio
from models.v2_discriminator import Discriminator
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

plt.rcParams.update({
    'font.size': 12,       # Default font size for all text
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 14,  # Label font size
    'legend.fontsize': 12, # Legend font size
})

def preprocess_test_data(dataset_dir, condition_window, sample_window, upsample_factor, max_clips_per_class=None):
    """
    Recursively processes audio files from all subdirectories under real/ and fake/ folders
    """
    import os
    data = []
    labels = []

    def find_wav_files(directory):
        wav_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    for label, folder in enumerate(["real", "fake"]):
        folder_path = os.path.join(dataset_dir, folder)
        audio_files = find_wav_files(folder_path)

        if max_clips_per_class is not None:
            audio_files = audio_files[:max_clips_per_class]

        print(f"Processing {len(audio_files)} files from {folder} (including subdirectories)")

        for audio_file in tqdm(audio_files, desc=f"Preprocessing {folder.capitalize()} Data"):
            mel, audio = convert_audio(audio_file)

            if len(audio) < sample_window:
                audio = np.pad(audio, (0, sample_window - len(audio)), mode='constant')
            else:
                audio = audio[:sample_window]

            if len(mel) < condition_window:
                mel = np.pad(mel, ((0, condition_window - len(mel)), (0, 0)), mode='constant')
            else:
                mel = mel[:condition_window]

            data.append((audio, mel))
            labels.append(label)
    
    labels = [1 - label for label in labels]
    print(f"Total preprocessed samples: {len(data)}")
    return data, labels


def compute_eer(fpr, tpr):
    """
    Computes EER given FPR and TPR arrays (assuming they're sorted by threshold).
    We want the point where FPR == 1 - TPR => FPR + TPR == 1.
    We'll do a simple linear search for the crossing, then linear interpolation if needed.
    """
    fnr = 1 - tpr
    differences = fpr - fnr
    idx_eq = np.where(differences == 0)[0]
    if len(idx_eq) > 0:
        # direct match
        eer = fpr[idx_eq[0]]
        return eer
    # otherwise we find a crossing
    signs = np.sign(differences)
    idx_change = np.where(np.diff(signs) != 0)[0]
    if len(idx_change) == 0:
        # no crossing found
        idx_min = np.argmin(np.abs(differences))
        eer = (fpr[idx_min] + fnr[idx_min]) / 2
        return eer
    # do a linear interpolation:
    i = idx_change[0]
    x0, x1 = fpr[i], fpr[i+1]
    y0, y1 = fnr[i], fnr[i+1]
    d0, d1 = (x0 - y0), (x1 - y1)
    alpha = abs(d0) / (abs(d1 - d0))
    eer = x0 + alpha * (x1 - x0)
    return eer

def plot_metrics_vs_threshold(all_probs, all_targets, output_dir="./eval_plots"):
    """
    Sweeps thresholds from 0 to 1 (class 0 as the "positive" class) and plots
    Precision, Recall, F1, and Accuracy vs. Threshold. 
    The figure is saved to 'metrics_vs_threshold_class_0.png' in the output directory.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)  # predicted=0 if prob<thresh
        TP = ((preds == 0) & (all_targets == 0)).sum()
        TN = ((preds == 1) & (all_targets == 1)).sum()
        FP = ((preds == 0) & (all_targets == 1)).sum()
        FN = ((preds == 1) & (all_targets == 0)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

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
    plt.savefig(os.path.join(output_dir, "metrics_vs_threshold_class_0.png"))
    plt.close()
    print("Saved threshold vs. metrics plot to:",
          os.path.join(output_dir, "metrics_vs_threshold_class_0.png"))

def evaluate_discriminator(discriminator, test_data, test_labels, device, output_dir="./eval_plots"):
    """
    Evaluate your discriminator on test data, then:
      - find threshold that yields best F1
      - plot confusion matrix at that threshold
      - compute EER%
      - plot/save ROC and PR curves
      - save text summary
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1) Accumulate all probabilities
    all_probs = []
    all_targets = []

    for (audio, mel), label in tqdm(zip(test_data, test_labels), desc="Evaluating", total=len(test_data)):
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).permute(0, 2, 1).to(device)

        with torch.no_grad():
            output = discriminator(audio_tensor)
        
        prob = output.mean(dim=-1).item()
        all_probs.append(prob)
        all_targets.append(label)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # 2) Sweep thresholds for best F1 and best accuracy
    thresholds = np.linspace(0, 1, 101)
    best_f1 = -1
    best_thresh = 0.5
    best_acc = -1
    best_acc_thresh = 0.5

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        TP = ((preds == 0) & (all_targets == 0)).sum()
        TN = ((preds == 1) & (all_targets == 1)).sum()
        FP = ((preds == 0) & (all_targets == 1)).sum()
        FN = ((preds == 1) & (all_targets == 0)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision+recall)>0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
        if accuracy > best_acc:
            best_acc = accuracy
            best_acc_thresh = thresh
            precision_at_best_acc = precision
            recall_at_best_acc = recall
            f1_at_best_acc = f1

    # 3) Confusion matrix + metrics at best threshold
    preds_best = (all_probs >= best_thresh).astype(int)
    cm = confusion_matrix(all_targets, preds_best)
    TP = cm[0,0]
    FN = cm[0,1]
    FP = cm[1,0]
    TN = cm[1,1]

    precision_at_best = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_at_best = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy_at_best = (TP + TN) / (TP + TN + FP + FN)

    # Plot confusion matrix
    import matplotlib
    matplotlib.use("agg")  # or skip if interactive
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.5,4.5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (best F1 threshold={best_thresh:.2f})")
    plt.colorbar()
    class_names = ["Class0(Real)","Class1(Fake)"]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh_val = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh_val else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix_bestF1.png"))
    plt.close()

    # Add confusion matrix at best accuracy
    preds_best_acc = (all_probs >= best_acc_thresh).astype(int)
    cm_acc = confusion_matrix(all_targets, preds_best_acc)
    
    # Plot confusion matrix for best accuracy
    plt.figure(figsize=(5.5,4.5))
    plt.imshow(cm_acc, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (best Accuracy threshold={best_acc_thresh:.2f})")
    plt.colorbar()
    class_names = ["Class0(Real)","Class1(Fake)"]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh_val = cm_acc.max() / 2.
    for i in range(cm_acc.shape[0]):
        for j in range(cm_acc.shape[1]):
            plt.text(j, i, format(cm_acc[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_acc[i, j] > thresh_val else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix_bestAcc.png"))
    plt.close()

    # 4) Compute EER
    # Because we interpret class 0 as "positive", we pass in (1 - prob) to roc_curve
    fpr, tpr, roc_thresh = roc_curve(all_targets, 1 - all_probs, pos_label=0)
    eer = compute_eer(fpr, tpr)
    eer_percent = 100.0 * eer

    # 5) Plot ROC curve
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Class 0 = Real)")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "roc_curve_class0.png"))
    plt.close()

    # 6) Plot Precision-Recall curve
    prec, rec, pr_thresh = precision_recall_curve(all_targets, 1 - all_probs, pos_label=0)
    # AP = area under the precision-recall curve
    ap = auc(rec, prec)
    plt.figure(figsize=(6,6))
    plt.plot(rec, prec, color='green', label=f"PR (AP={ap:.4f})")
    plt.xlabel("Recall (Class 0)")
    plt.ylabel("Precision (Class 0)")
    plt.title("Precision-Recall Curve (Class 0 = Real)")
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "pr_curve_class0.png"))
    plt.close()
    plot_metrics_vs_threshold(all_probs, all_targets, output_dir=output_dir)
    # 7) Print & Save final metrics
    print("====== Evaluation Results ======")
    print("\nMetrics at Best F1:")
    print(f"Best F1 Threshold = {best_thresh:.3f}")
    print(f"F1               = {best_f1:.4f}")
    print(f"Precision        = {precision_at_best:.4f}")
    print(f"Recall           = {recall_at_best:.4f}")
    print(f"Accuracy         = {accuracy_at_best:.4f}")
    
    print("\nMetrics at Best Accuracy:")
    print(f"Best Acc Threshold = {best_acc_thresh:.3f}")
    print(f"Accuracy           = {best_acc:.4f}")
    print(f"F1                 = {f1_at_best_acc:.4f}")
    print(f"Precision          = {precision_at_best_acc:.4f}")
    print(f"Recall             = {recall_at_best_acc:.4f}")
    
    print(f"\nEER%%              = {eer_percent:.4f}")
    print(f"ROC AUC            = {roc_auc:.4f}")
    print(f"PR AUC             = {ap:.4f}")
    print("\nConfusion Matrix at Best F1:\n", cm)
    print("\nConfusion Matrix at Best Accuracy:\n", cm_acc)

    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write("====== Evaluation Results ======\n")
        f.write("\nMetrics at Best F1:\n")
        f.write(f"Best F1 Threshold = {best_thresh:.3f}\n")
        f.write(f"F1               = {best_f1:.4f}\n")
        f.write(f"Precision        = {precision_at_best:.4f}\n")
        f.write(f"Recall           = {recall_at_best:.4f}\n")
        f.write(f"Accuracy         = {accuracy_at_best:.4f}\n")
        
        f.write("\nMetrics at Best Accuracy:\n")
        f.write(f"Best Acc Threshold = {best_acc_thresh:.3f}\n")
        f.write(f"Accuracy           = {best_acc:.4f}\n")
        f.write(f"F1                 = {f1_at_best_acc:.4f}\n")
        f.write(f"Precision          = {precision_at_best_acc:.4f}\n")
        f.write(f"Recall             = {recall_at_best_acc:.4f}\n")
        
        f.write(f"\nEER%%              = {eer_percent:.4f}\n")
        f.write(f"ROC AUC            = {roc_auc:.4f}\n")
        f.write(f"PR AUC             = {ap:.4f}\n")
        f.write("\nConfusion Matrix at Best F1:\n")
        f.write(str(cm) + "\n")
        f.write("\nConfusion Matrix at Best Accuracy:\n")
        f.write(str(cm_acc) + "\n")


def main():
    dataset_dir = "../data_train/eval/"  # Root directory containing real/ and fake/ subfolders
    checkpoint_path = "./logdir_noex_5/mgan_step_240000.pth"
    condition_window = 100
    upsample_factor = 120
    sample_window = condition_window * upsample_factor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load the discriminator
    discriminator = Discriminator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    discriminator.load_state_dict(checkpoint["discriminator"])
    discriminator.eval()

    # 2) Preprocess test data
    test_data, test_labels = preprocess_test_data(dataset_dir,
                                                  condition_window,
                                                  sample_window,
                                                  upsample_factor,
                                                  max_clips_per_class=10000)

    # 3) Evaluate
    evaluate_discriminator(discriminator, test_data, test_labels, device, output_dir="./plots/mgan_noex_5_step_240000")


if __name__ == "__main__":
    main()
