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
    Process audio files from the given directory
    """
    import os
    data = []
    labels = []
    print(f"Looking for data in: {os.path.abspath(dataset_dir)}")

    if not os.path.exists(dataset_dir):
        raise ValueError(f"Directory does not exist: {dataset_dir}")

    def find_wav_files(directory):
        if not os.path.exists(directory):
            print(f"Warning: Directory does not exist: {directory}")
            return []
        wav_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    # Find all .wav files in the directory
    audio_files = find_wav_files(dataset_dir)
    if max_clips_per_class is not None:
        audio_files = audio_files[:max_clips_per_class]

    print(f"Processing {len(audio_files)} files from directory")
    
    # Determine label based on directory name
    label = 1 if os.path.basename(dataset_dir) == "fake" else 0
    
    for audio_file in tqdm(audio_files, desc=f"Preprocessing Data"):
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
        # Real is class 1 (high probability), Fake is class 0 (low probability)
        preds = (all_probs >= threshold).astype(int)
        TP = ((preds == 1) & (all_targets == 1)).sum()
        TN = ((preds == 0) & (all_targets == 0)).sum()
        FP = ((preds == 1) & (all_targets == 0)).sum()
        FN = ((preds == 0) & (all_targets == 1)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Precision (Real)", color="blue")
    plt.plot(thresholds, recalls, label="Recall (Real)", color="orange")
    plt.plot(thresholds, f1_scores, label="F1 Score (Real)", color="green")
    plt.plot(thresholds, accuracies, label="Accuracy", color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Metrics vs. Threshold (Real = Class 1)")
    plt.legend(loc="best")
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
    if len(test_data) == 0:
        raise ValueError("No test data provided for evaluation")
    
    if len(test_labels) == 0:
        raise ValueError("No test labels provided for evaluation")

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
        # Invert probability since discriminator outputs high for fake
        prob = 1 - prob
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
        # Now low prob means fake (0) and high prob means real (1)
        preds = (all_probs >= thresh).astype(int)
        TP = ((preds == 1) & (all_targets == 1)).sum()
        TN = ((preds == 0) & (all_targets == 0)).sum()
        FP = ((preds == 1) & (all_targets == 0)).sum()
        FN = ((preds == 0) & (all_targets == 1)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision+recall)>0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            precision_at_best = precision
            recall_at_best = recall
            
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
    class_names = ["Fake(0)", "Real(1)"]
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
    class_names = ["Fake(0)", "Real(1)"]
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
    # Now we use probabilities directly since high prob = real (class 1)
    fpr, tpr, roc_thresh = roc_curve(all_targets, all_probs)
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
    prec, rec, pr_thresh = precision_recall_curve(all_targets, all_probs)
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

def generate_fake_samples(generators, test_data, device):
    """Generate fake samples using the generators"""
    fake_samples = []
    for audio, mel in tqdm(test_data, desc="Generating fake samples"):
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0).permute(0, 2, 1).to(device)
        z = torch.randn(1, 128, device=device)  # Assuming z_dim=128
        
        # For each generator, generate a fake sample
        for generator in generators:
            with torch.no_grad():
                fake_audio = generator(mel_tensor, z)
            fake_samples.append((fake_audio.squeeze().cpu().numpy(), mel))
    
    return fake_samples

def evaluate_all_scenarios(discriminator, generators, real_eval_data, fake_eval_data, device, output_dir="./eval_plots"):
    """Evaluate both scenarios: eval fakes vs eval reals, and generated fakes vs eval reals"""
    
    # Validate input data
    if len(real_eval_data) == 0:
        raise ValueError("No real evaluation data found. Please check the real data directory path.")
    
    if len(fake_eval_data) == 0:
        print("Warning: No fake evaluation data found. Proceeding with only generated fakes evaluation.")
        
    # First scenario: Only evaluate if we have fake evaluation data
    if len(fake_eval_data) > 0:
        print("\n=== Evaluating Fake (Eval) vs Real (Eval) ===")
        print(f"Number of real samples: {len(real_eval_data)}")
        print(f"Number of fake samples: {len(fake_eval_data)}")
        
        eval_output_dir = os.path.join(output_dir, "eval_fake_vs_real")
        os.makedirs(eval_output_dir, exist_ok=True)
        
        eval_data = real_eval_data + fake_eval_data
        # Real is 1, Fake is 0
        eval_labels = [1] * len(real_eval_data) + [0] * len(fake_eval_data)
        evaluate_discriminator(discriminator, eval_data, eval_labels, device, output_dir=eval_output_dir)
    
    # Second scenario: Generated fakes vs evaluation reals
    print("\n=== Evaluating Fake (Generated) vs Real (Eval) ===")
    gen_output_dir = os.path.join(output_dir, "generated_fake_vs_real")
    os.makedirs(gen_output_dir, exist_ok=True)
    
    # Generate fake samples using our generators
    generated_fakes = generate_fake_samples(generators, real_eval_data, device)
    print(f"Number of real samples: {len(real_eval_data)}")
    print(f"Number of generated fake samples: {len(generated_fakes)}")
    
    gen_data = real_eval_data + generated_fakes
    # Real is 1, Fake is 0
    gen_labels = [1] * len(real_eval_data) + [0] * len(generated_fakes)
    evaluate_discriminator(discriminator, gen_data, gen_labels, device, output_dir=gen_output_dir)

def main():
    # Get the absolute path of the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths
    dataset_dir = os.path.abspath(os.path.join(script_dir, "../data_train/eval"))
    checkpoint_path = os.path.join(script_dir, "logdir_noex3/mgan_step_10000.pth")
    condition_window = 100
    upsample_factor = 120
    sample_window = condition_window * upsample_factor

    # Validate paths
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    print(f"Using dataset directory: {dataset_dir}")
    print(f"Using checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load all models
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load discriminator
    discriminator = Discriminator().to(device)
    discriminator.load_state_dict(checkpoint["discriminator"])
    discriminator.eval()
    
    # Load generators
    from models.generator import Generator
    generators = []
    num_generators = sum(1 for k in checkpoint.keys() if k.startswith("generator_"))
    
    for i in range(num_generators):
        generator = Generator(80, 128).to(device)  # Assuming mel_dim=80, z_dim=128
        generator.load_state_dict(checkpoint[f"generator_{i}"])
        generator.eval()
        generators.append(generator)
    
    print(f"Loaded {len(generators)} generators")

    # 2) Preprocess evaluation data separately for real and fake
    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")
    
    print(f"\nProcessing real directory: {real_dir}")
    real_data, _ = preprocess_test_data(real_dir, condition_window, sample_window, 
                                      upsample_factor, max_clips_per_class=1000)
    
    print(f"\nProcessing fake directory: {fake_dir}")
    fake_data, _ = preprocess_test_data(fake_dir, condition_window, sample_window, 
                                      upsample_factor, max_clips_per_class=1000)

    # 3) Evaluate both scenarios
    evaluate_all_scenarios(
        discriminator=discriminator,
        generators=generators,
        real_eval_data=real_data,
        fake_eval_data=fake_data,
        device=device,
        output_dir="./plots/mgan_noex2_step_10000"
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
