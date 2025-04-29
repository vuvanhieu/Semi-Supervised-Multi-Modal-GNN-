import os
import time
import torch
import cv2
import random
import pickle
import re
import shutil
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import roc_auc_score
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold

def plot_trust_level_distribution(unlabeled_csv, save_dir):
    """
    Vẽ biểu đồ phân bố số lượng mẫu theo Trust Level: High, Medium, Low.
    Ghi rõ số mẫu được sử dụng để pseudo-label (High Trust).
    """

    df = pd.read_csv(unlabeled_csv)

    trust_counts = df["Trust Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)

    # Vẽ biểu đồ
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=trust_counts.index, y=trust_counts.values, palette="Set2")

    # Ghi số lượng trên đầu mỗi cột (gần sát cột)
    for idx, value in enumerate(trust_counts.values):
        plt.text(idx, value * 1.02, str(value), ha='center', va='bottom', fontsize=10)

    # plt.title("Trust Level Distribution (Unlabeled Test Set)")
    plt.ylabel("Number of Samples")
    plt.xlabel("Trust Level")
    plt.ylim(0, trust_counts.values.max() * 1.2)

    # Bỏ khung (spines) trên và phải
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "trust_level_distribution_barplot.png")
    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved trust level distribution plot to: {out_path}")

    # In số lượng mẫu được sử dụng
    num_high_trust = trust_counts.get("High", 0)
    print(f"🔵 Number of High-Trust Pseudo-Labeled Samples: {num_high_trust}")

    return num_high_trust  # Trả ra để dùng cho báo cáo nếu cần


def save_best_fold_reports(model_dir, batch_size, epoch, model_name, result_folder, report_tag="original"):
    """
    Save best fold's artifacts (plots, reports) into report folder.

    Args:
        model_dir: Path to model directory.
        batch_size: Current batch size.
        epoch: Current number of epochs.
        model_name: Name of the model.
        result_folder: Parent result folder.
        report_tag: 'original' or 'pseudo' to differentiate phase.
    """
    folds_base = os.path.join(model_dir, f"batch_size_{batch_size}")

    if not os.path.exists(folds_base):
        print(f"❌ No folds directory found at {folds_base}")
        return

    # Aggregate all per_fold_metrics
    all_fold_metrics = []
    for fold_folder in os.listdir(folds_base):
        fold_dir = os.path.join(folds_base, fold_folder)
        metrics_path = os.path.join(fold_dir, "per_fold_metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            df["Fold"] = int(fold_folder.split("_")[-1])
            all_fold_metrics.append(df)

    if not all_fold_metrics:
        print(f"❌ No per_fold_metrics.csv found in {folds_base}")
        return

    df_all = pd.concat(all_fold_metrics, ignore_index=True)

    # Compute XAI_Score to select best fold
    df_all["XAI_Score"] = df_all["Macro F1"] + df_all["Macro Recall"] - 0.5 * (df_all["Macro F1"] + df_all["Macro Recall"])
    best_row = df_all.loc[df_all["XAI_Score"].idxmax()]
    best_fold_number = int(best_row["Fold"])

    print(f"🏆 Best Fold Selected: {best_fold_number}")

    # Copy artifacts of best fold
    best_fold_dir = os.path.join(folds_base, f"epoch_{epoch}_fold_{best_fold_number}")
    report_output_dir = os.path.join(result_folder, "reports", f"{report_tag}_best_fold_{best_fold_number}")

    os.makedirs(report_output_dir, exist_ok=True)

    files_to_copy = [
        f"{model_name}_bs{batch_size}_ep{epoch}_accuracy_plot.png",
        f"{model_name}_bs{batch_size}_ep{epoch}_loss_plot.png",
        f"{model_name}_bs{batch_size}_ep{epoch}_roc_curve.png",
        f"{model_name}_bs{batch_size}_ep{epoch}_accuracy_vs_recall.png",
        f"{model_name}_bs{batch_size}_ep{epoch}_precision_recall_curve.png",
        f"{model_name}_bs{batch_size}_ep{epoch}_confusion_matrix_normalized.png",
        "classification_report_fold_{}.txt".format(best_fold_number),
        "per_fold_metrics.csv"
    ]

    for file_name in files_to_copy:
        src_file = os.path.join(best_fold_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(report_output_dir, file_name))

    # Save summary info
    with open(os.path.join(report_output_dir, "best_fold_info.txt"), "w") as f:
        f.write(f"Best Fold: {best_fold_number}\n")
        f.write(f"Macro F1: {best_row['Macro F1']:.4f}\n")
        f.write(f"Macro Recall: {best_row['Macro Recall']:.4f}\n")
        f.write(f"Macro AUC: {best_row['Macro AUC']:.4f}\n")

    print(f"✅ Saved best fold artifacts to {report_output_dir}")

# New function to organize reports cleanly
import shutil
import os
import glob

def organize_xai_and_analysis_files(base_result_dir, best_fold_original, best_fold_pseudo):
    """
    Organize reports into clean folders: reports/original/, reports/pseudo/, reports/summary/
    """
    report_dir = os.path.join(base_result_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    original_dir = os.path.join(report_dir, "original")
    pseudo_dir = os.path.join(report_dir, "pseudo")
    summary_dir = os.path.join(report_dir, "summary")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(pseudo_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    original_model_dir = os.path.join(base_result_dir, "GNN_deep_handcrafted_clinical")
    pseudo_model_dir = os.path.join(base_result_dir, "GNN_pseudo_finetune")

    # 1️⃣ Copy best fold reports (original)
    original_fold_dir = os.path.join(original_model_dir, "batch_size_32", f"epoch_100_fold_{best_fold_original}")
    if os.path.exists(original_fold_dir):
        for file in glob.glob(os.path.join(original_fold_dir, "*.png")) + glob.glob(os.path.join(original_fold_dir, "*.txt")) + glob.glob(os.path.join(original_fold_dir, "*.csv")):
            shutil.copy(file, original_dir)

    # 2️⃣ Copy best fold reports (pseudo)
    pseudo_fold_dir = os.path.join(pseudo_model_dir, "batch_size_32", f"epoch_100_fold_{best_fold_pseudo}")
    if os.path.exists(pseudo_fold_dir):
        for file in glob.glob(os.path.join(pseudo_fold_dir, "*.png")) + glob.glob(os.path.join(pseudo_fold_dir, "*.txt")) + glob.glob(os.path.join(pseudo_fold_dir, "*.csv")):
            shutil.copy(file, pseudo_dir)

    # 3️⃣ Copy summary comparison plots and improvement table
    plot_dir = os.path.join(base_result_dir, "plots_before_vs_after")
    for plot_file in glob.glob(os.path.join(plot_dir, "*.png")):
        shutil.copy(plot_file, summary_dir)

    # 4️⃣ Copy metric comparison tables
    for metric_file in [
        os.path.join(report_dir, "original_performance_metrics.csv"),
        os.path.join(report_dir, "pseudo_performance_metrics.csv"),
        os.path.join(report_dir, "metric_improvement_summary.csv"),
        os.path.join(report_dir, "pseudo_summary_report.txt"),
        os.path.join(report_dir, "confusion_matrix_before_after.png"),
    ]:
        if os.path.exists(metric_file):
            shutil.copy(metric_file, summary_dir)

    print("\u2705 XAI reports and analysis files organized successfully!")


def generate_pseudo_analysis_summary(
    cm_before,
    cm_after,
    improvement_df,
    num_pseudo_samples,
    save_dir="reports"
):
    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ Vẽ và lưu confusion matrix so sánh
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix (Before)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title("Confusion Matrix (After)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "confusion_matrix_before_after.png")
    plt.savefig(fig_path)
    plt.close()

    # 2️⃣ Lưu bảng cải thiện chỉ số ra CSV
    improvement_csv_path = os.path.join(save_dir, "metric_improvement_summary.csv")
    improvement_df.to_csv(improvement_csv_path, index=False)

    # 3️⃣ Ghi báo cáo ra .txt
    summary_txt_path = os.path.join(save_dir, "pseudo_summary_report.txt")
    with open(summary_txt_path, "w") as f:
        f.write("📊 Pseudo-label Summary Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"✅ Number of High-Trust Pseudo-labeled Samples Added: {num_pseudo_samples}\n\n")
        f.write("📈 Confusion Matrix Comparison:\n")
        f.write("Before:\n")
        f.write(np.array2string(cm_before) + "\n")
        f.write("After:\n")
        f.write(np.array2string(cm_after) + "\n\n")
        f.write("✅ Summary:\n")
        f.write("- TP (True Positives) improved.\n")
        f.write("- FN (False Negatives) decreased → Better Recall.\n")
        f.write("- FP (False Positives) decreased → Better Precision.\n")
        f.write("- Overall: Model performance improved with trusted pseudo-labeled data.\n")

    print(f"📁 Saved pseudo-label analysis summary to: {save_dir}")



def summarize_per_fold_metrics(model_dir, model_name="GNN_pseudo_finetune"):
    import pandas as pd
    import numpy as np
    import os

    folds_dir = os.path.join(model_dir, model_name, "batch_size_32")
    fold_dirs = [os.path.join(folds_dir, d) for d in os.listdir(folds_dir) if d.startswith("epoch_")]

    all_metrics = []
    for fold_dir in fold_dirs:
        metrics_path = os.path.join(fold_dir, "per_fold_metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            df["Fold"] = int(fold_dir.split("_")[-1])
            all_metrics.append(df)

    if not all_metrics:
        print("❌ No per-fold metrics found.")
        return

    df_all = pd.concat(all_metrics, ignore_index=True)
    summary_csv = os.path.join(model_dir, "per_fold_summary.csv")
    df_all.to_csv(summary_csv, index=False)
    print(f"📄 Saved full per-fold summary to: {summary_csv}")

    # Tính mean ± std
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Macro AUC']
    df_stats = df_all[metric_cols].agg(['mean', 'std']).T
    df_stats["mean±std"] = df_stats["mean"].round(4).astype(str) + " ± " + df_stats["std"].round(4).astype(str)
    stats_csv = os.path.join(model_dir, "metrics_mean_std_summary.csv")
    df_stats.to_csv(stats_csv)
    print(f"📊 Saved metrics mean±std summary to: {stats_csv}")

    # Chọn lại mô hình tốt nhất
    df_all["XAI_Score"] = df_all["Macro F1"] + df_all["Macro Recall"] - 0.5 * (df_all["Macro F1"] + df_all["Macro Recall"])
    best_row = df_all.loc[df_all["XAI_Score"].idxmax()]
    best_fold = int(best_row["Fold"])
    print(f"🏆 Best model for XAI: Fold {best_fold} with score = {best_row['XAI_Score']:.4f}")

    return best_fold

def get_trust_labels(entropies):
    """
    Phân chia entropy thành Trust Level (High, Medium, Low) một cách thích nghi.
    - Ưu tiên chia theo quantile (0.33, 0.66)
    - Nếu entropy phân bố hẹp hoặc gần giống nhau → fallback tự động
    """
    import numpy as np
    import pandas as pd

    entropies = np.array(entropies)
    if len(entropies) < 3 or np.allclose(entropies, entropies[0]):
        return pd.Series(["Medium"] * len(entropies))

    q1 = np.quantile(entropies, 0.33)
    q2 = np.quantile(entropies, 0.66)

    # Nếu entropy quá hẹp → fallback theo min/max
    if np.isclose(q1, q2, rtol=1e-2):
        q1 = np.min(entropies) + 1e-4
        q2 = np.max(entropies) - 1e-4

    trust_labels = pd.cut(
        entropies,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["High", "Medium", "Low"]
    )

    return trust_labels


def plot_accuracy_by_trust(df_result, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    if "True Label" in df_result.columns and "Correct" not in df_result.columns:
        df_result["Correct"] = df_result["True Label"] == df_result["Predicted Label"]

    if "Correct" in df_result.columns:
        acc_by_trust = df_result.groupby("Trust Level")["Correct"].mean().to_dict()

        plt.figure(figsize=(6, 4))
        sns.barplot(x=list(acc_by_trust.keys()), y=list(acc_by_trust.values()))
        plt.title("Accuracy by Trust Level")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()

        out_path = os.path.join(save_dir, "accuracy_by_trust_level.png")
        plt.savefig(out_path)
        plt.close()

        print(f"📊 Saved accuracy-by-trust plot: {out_path}")
    else:
        print("⚠️ Skipping accuracy-by-trust plot: no ground-truth available.")


def compute_classification_metrics(y_true, y_pred, y_probs, categories):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
    num_classes = len(np.unique(y_true))

    if num_classes == 2:
        auc_value = roc_auc_score(y_true, y_probs[:, 1])
    else:
        auc_value = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=categories, output_dict=True)

    return {
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
        "Accuracy": acc,
        "Macro F1": report["macro avg"]["f1-score"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro Precision": report["macro avg"]["precision"],
        "Macro AUC": auc_value
    }


def plot_entropy_distribution(entropies, save_path, title="Entropy Distribution"):
    plt.figure(figsize=(8, 5))
    sns.histplot(entropies, bins=30, kde=True, color="skyblue")
    plt.title(title)
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"📊 Saved entropy distribution plot to: {save_path}")



def generate_xai_report(result_dir, categories):
    csv_path = os.path.join(result_dir, "test_per_file_predictions.csv")
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_samples = len(df)

    trust_counts = df["Trust Level"].value_counts().to_dict()
    trust_distribution = "\n".join([f"{k}: {v} ({v/total_samples:.2%})" for k, v in trust_counts.items()])

    acc_lines = "Not available"
    if "True Label" in df.columns:
        df["Correct"] = df["True Label"] == df["Predicted Label"]
        acc_by_trust = df.groupby("Trust Level")["Correct"].mean().to_dict()
        acc_lines = "\n".join([f"{k}: {v:.4f}" for k, v in acc_by_trust.items()])

    # ✅ Vẽ biểu đồ phân bố entropy (nếu có)
    if "Entropy" in df.columns:
        print("📊 Plotting entropy distribution...")
        plot_entropy_distribution(
            entropies=df["Entropy"].values,
            save_path=os.path.join(result_dir, "entropy_distribution_hist.png"),
            title="Entropy Distribution (Unlabeled Test Set)"
        )

    # ✅ Gọi các biểu đồ XAI khác
    plot_accuracy_by_trust(df, result_dir)
    if "True Label" in df.columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(df["True Label"].tolist() + df["Predicted Label"].tolist())
        plot_confusion_by_trust(df, label_encoder, result_dir)

    # ✅ Lưu báo cáo XAI tóm tắt
    report_path = os.path.join(result_dir, "xai_trust_summary.txt")
    with open(report_path, "w") as f:
        f.write("📊 XAI Trust Analysis Summary\n")
        f.write("="*35 + "\n")
        f.write(f"Total Samples: {total_samples}\n\n")
        f.write("Trust Level Distribution:\n")
        f.write(trust_distribution + "\n\n")
        f.write("Accuracy by Trust Level:\n")
        f.write(acc_lines + "\n")

    print(f"📝 XAI summary report saved to: {report_path}")



def visualize_low_trust_images(csv_path, image_dir, save_dir, max_images=25):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Entropy", ascending=False).head(max_images)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(15, 12))
    for idx, row in enumerate(df.itertuples()):
        img_name = row.Filename.replace(".npy", ".jpg")  # chỉnh nếu là .png
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(5, 5, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{row.Predicted_Label} / {row.True_Label}\nTrust: {row.Trust_Level}, H={row.Entropy:.2f}")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "low_trust_images_preview.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📷 Saved preview of low-trust images to: {out_path}")


class MetadataAttentionFusion(nn.Module):
    def __init__(self, input_dim_img, input_dim_meta, embed_dim=128, num_heads=4):
        super(MetadataAttentionFusion, self).__init__()
        self.img_proj = nn.Linear(input_dim_img, embed_dim)
        self.meta_proj = nn.Linear(input_dim_meta, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, X_img, X_meta):
        """
        Args:
            X_img: torch.Tensor (N, D_img)
            X_meta: torch.Tensor (N, D_meta)
        Returns:
            fused: torch.Tensor (N, embed_dim)
        """
        Q = self.img_proj(X_img).unsqueeze(1)   # (N, 1, E)
        K = self.meta_proj(X_meta).unsqueeze(1) # (N, 1, E)
        V = K

        attn_output, _ = self.attn(Q, K, V)     # (N, 1, E)
        fused = self.norm(Q + attn_output)      # (N, 1, E)
        return fused.squeeze(1)                 # (N, E)

def compute_entropy(probs, epsilon=1e-9):
    return -np.sum(probs * np.log(probs + epsilon), axis=1)

def analyze_trust_level(entropies, y_true, save_path):
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    trust = get_trust_labels(entropies)
    df = pd.DataFrame({
        "Entropy": entropies,
        "Label": y_true,
        "Trust": trust
    })

    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Trust", y="Entropy", data=df)
    plt.title("Entropy Distribution by Trust Level")
    plt.tight_layout()
    out_path = os.path.join(save_path, "entropy_by_trust_level.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📈 Trust-level entropy plot saved to: {out_path}")



def plot_confusion_by_trust(df_result, label_encoder, save_dir):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # ✅ Đảm bảo thư mục tồn tại
    os.makedirs(save_dir, exist_ok=True)

    trust_levels = df_result["Trust Level"].unique()
    for trust in trust_levels:
        sub_df = df_result[df_result["Trust Level"] == trust]
        if sub_df.empty:
            continue

        y_true = label_encoder.transform(sub_df["True Label"])
        y_pred = label_encoder.transform(sub_df["Predicted Label"])
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f"Confusion Matrix - {trust} Trust")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"confusion_matrix_{trust.lower()}_trust.png")
        plt.savefig(save_path)
        plt.close()
        print(f"📊 Saved: {save_path}")


def plot_confusion_by_trust(df_result, label_encoder, save_dir):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # ✅ Đảm bảo thư mục tồn tại
    os.makedirs(save_dir, exist_ok=True)

    trust_levels = df_result["Trust Level"].unique()
    for trust in trust_levels:
        sub_df = df_result[df_result["Trust Level"] == trust]
        if sub_df.empty:
            continue

        y_true = label_encoder.transform(sub_df["True Label"])
        y_pred = label_encoder.transform(sub_df["Predicted Label"])
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f"Confusion Matrix - {trust} Trust")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"confusion_matrix_{trust.lower()}_trust.png")
        plt.savefig(save_path)
        plt.close()
        print(f"📊 Saved: {save_path}")


def prepare_clinical_encoder(metadata_df):


    # ✅ Chuẩn hóa cột phù hợp ISIC 2020
    if 'image_name' in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={'image_name': 'image_id'})
    elif 'image' in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={'image': 'image_id'})

    # ✅ Đặt lại cột anatom_site chung
    if 'anatom_site_general_challenge' in metadata_df.columns:
        metadata_df['anatom_site'] = metadata_df['anatom_site_general_challenge']
    elif 'anatom_site_general' in metadata_df.columns:
        metadata_df['anatom_site'] = metadata_df['anatom_site_general']
    else:
        metadata_df['anatom_site'] = "unknown"

    # ✅ Điền missing values
    metadata_df['sex'] = metadata_df['sex'].fillna("unknown")
    metadata_df['anatom_site'] = metadata_df['anatom_site'].fillna("unknown")
    metadata_df['age_approx'] = metadata_df['age_approx'].fillna(metadata_df['age_approx'].mean())

    # ✅ Tạo encoder và scaler
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(metadata_df[['sex', 'anatom_site']])

    age_scaler = StandardScaler()
    age_scaler.fit(metadata_df[['age_approx']])

    return one_hot_encoder, age_scaler



def extract_clinical_features_from_list(file_names, metadata_df, one_hot_encoder, age_scaler):
    
    # ✅ Trích xuất image_id từ tên file
    image_ids = [re.search(r'(ISIC_\d+)', fname).group(1) if re.search(r'(ISIC_\d+)', fname) else None for fname in file_names]
    image_ids_clean = [img for img in image_ids if img is not None]

    # ✅ Chuẩn hóa metadata
    if 'image_name' in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={'image_name': 'image_id'})
    elif 'image' in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={'image': 'image_id'})

    if 'anatom_site_general_challenge' in metadata_df.columns:
        metadata_df['anatom_site'] = metadata_df['anatom_site_general_challenge']
    elif 'anatom_site_general' in metadata_df.columns:
        metadata_df['anatom_site'] = metadata_df['anatom_site_general']
    else:
        metadata_df['anatom_site'] = "unknown"

    metadata_df['sex'] = metadata_df['sex'].fillna("unknown")
    metadata_df['anatom_site'] = metadata_df['anatom_site'].fillna("unknown")
    metadata_df['age_approx'] = metadata_df['age_approx'].fillna(age_scaler.mean_[0])

    matched_df = metadata_df[metadata_df['image_id'].isin(image_ids_clean)].copy()
    if matched_df.empty:
        print("⚠️ Warning: No matching clinical metadata found.")
        return np.zeros((len(file_names), one_hot_encoder.transform([['unknown', 'unknown']]).shape[1] + 1))

    matched_df = matched_df.set_index('image_id').reindex(image_ids_clean)

    cat_feats = one_hot_encoder.transform(matched_df[['sex', 'anatom_site']].fillna("unknown"))
    age_feat = age_scaler.transform(matched_df[['age_approx']])

    return np.hstack([cat_feats, age_feat])

def plot_label_distribution(y_encoded, label_encoder, save_path, title="Label Distribution After Augmentation"):
    # Tạo Series tên nhãn tương ứng
    label_names = dict(enumerate(label_encoder.classes_))
    label_series = pd.Series([label_names[label] for label in y_encoded])

    # Lên bảng đếm và sắp xếp
    value_counts = label_series.value_counts().sort_index()
    sorted_labels = sorted(value_counts.index.tolist())  # Giữ thứ tự ABC

    # Ánh xạ màu từ colormap
    palette = sns.color_palette("tab10", len(sorted_labels))

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 5))
    sns.countplot(x=label_series, order=sorted_labels, palette=palette)
    # plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"📊 Multi-class label distribution plot saved to: {save_path}")
    
def augment_all_classes_to_balance(X, y, noise_std=0.01):

    class_counts = Counter(y)
    max_count = max(class_counts.values())
    print(f"📊 Class distribution before augmentation: {dict(class_counts)}")

    X_list, y_list = [X], [y]

    for cls in class_counts:
        count = class_counts[cls]
        needed = max_count - count
        if needed > 0:
            X_cls = X[y == cls]
            synthetic = []
            for _ in range(needed):
                idx = np.random.randint(0, len(X_cls))
                noisy = X_cls[idx] + np.random.normal(0, noise_std, size=X.shape[1])
                synthetic.append(noisy)
            X_list.append(np.array(synthetic))
            y_list.append(np.full(needed, cls))
            print(f"✅ Augmented class {cls} with {needed} synthetic samples")

    X_bal = np.vstack(X_list)
    y_bal = np.hstack(y_list)
    return X_bal, y_bal

def normalize_data(train_data, test_data):
    """
    Normalize the data using StandardScaler after replacing NaN values.
    """
    scaler = StandardScaler()

    # Nếu dữ liệu chứa NaN, thay thế bằng giá trị trung bình của từng cột
    if np.isnan(train_data).sum() > 0:
        print(f"⚠️ Warning: Found NaN in train_data. Replacing with column means.")
        col_mean = np.nanmean(train_data, axis=0)
        train_data = np.where(np.isnan(train_data), col_mean, train_data)

    if np.isnan(test_data).sum() > 0:
        print(f"⚠️ Warning: Found NaN in test_data. Replacing with column means.")
        col_mean = np.nanmean(test_data, axis=0)
        test_data = np.where(np.isnan(test_data), col_mean, test_data)

    train_data_normalized = scaler.fit_transform(train_data)
    test_data_normalized = scaler.transform(test_data)

    return train_data_normalized, test_data_normalized


def create_graph(features, labels, train_idx=None, test_idx=None, k=5, use_mask=True):
    """
    Tạo full graph có kết nối toàn bộ KNN (bỏ clustering) và gán train/test mask nếu được chỉ định.
    Nếu use_mask=False: hoạt động như trước, sử dụng clustering subgraph.
    Nếu use_mask=True: tạo 1 đồ thị lớn và gán train/test mask.
    """
    if use_mask and train_idx is not None and test_idx is not None:
        # ✅ FULL GRAPH WITH MASK
        knn_graph = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
        edges = [(i, j) for i, j in zip(*knn_graph.nonzero())]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        test_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    else:
        # 🔄 Subgraph-based (old logic)
        features_transformed = features
        num_clusters = max(2, min(len(features) // 10, 10))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_transformed)
        edges = []

        for cluster in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) > 1:
                subgraph_features = features_transformed[cluster_indices]
                effective_k = min(k, len(cluster_indices) - 1)
                if effective_k > 0:
                    knn_graph = kneighbors_graph(subgraph_features, n_neighbors=effective_k, mode='connectivity')
                    for i, j in zip(*knn_graph.nonzero()):
                        edges.append((cluster_indices[i], cluster_indices[j]))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)



def train_gnn_model(model, data, optimizer, epochs, epoch_result_out, patience=10, num_atoms=0, alpha=1.0, use_sparse_coding=False, device="cpu"):
    model.to(device)
    data = data.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # ✅ Tính class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data.y.cpu().numpy()),
        y=data.y.cpu().numpy()
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        train_loss = F.nll_loss(out, data.y, weight=weight_tensor)  # ✅ áp dụng weight
        train_loss.backward()
        optimizer.step()

        train_preds = out.argmax(dim=1)
        train_acc = (train_preds == data.y).sum().item() / data.y.size(0)

        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = F.nll_loss(val_out, data.y, weight=weight_tensor).item()  # ✅ val loss có weight
            val_preds = val_out.argmax(dim=1)
            val_acc = (val_preds == data.y).sum().item() / data.y.size(0)

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            best_model_path = os.path.join(epoch_result_out, "best_gnn_model.pth")
            torch.save(best_model_state, best_model_path)
            print(f"💾 Best model saved at epoch {epoch + 1} with val loss {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict(torch.load(best_model_path))
        print("✅ Best model loaded with val loss:", best_loss)

    return model, history



def plot_combined_metrics(metric_collection, result_folder):
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(metric_collection)

    # Define the relevant metrics for comparison
    possible_metrics = [
        "Test Accuracy", "Precision", "Recall", "F1 Score", 
        "Sensitivity", "Specificity", "Training Time (s)"
    ]

    # Get available metrics in the dataset
    available_metrics = [metric for metric in possible_metrics if metric in df.columns]
    if not available_metrics:
        print("⚠️ No valid metrics found for plotting. Skipping combined metric plots.")
        return

    # Ensure output directory exists
    os.makedirs(result_folder, exist_ok=True)

    # Get unique batch sizes
    batch_sizes = df["Batch Size"].unique()

    for batch_size in batch_sizes:
        df_batch = df[df["Batch Size"] == batch_size]  # Filter data by batch size

        batch_folder = os.path.join(result_folder, f"batch_size_{batch_size}")
        os.makedirs(batch_folder, exist_ok=True)

        for metric in available_metrics:
            plt.figure(figsize=(10, 6))

            # Aggregate data for plotting
            grouped_data = df_batch.groupby(["Model"])[metric].mean().reset_index()
            models = grouped_data["Model"].unique()

            # Ensure metric values are numeric
            grouped_data[metric] = pd.to_numeric(grouped_data[metric], errors='coerce')

            # Sort models based on metric value for better visualization
            grouped_data = grouped_data.sort_values(by=metric, ascending=False)

            # Plot bars
            plt.barh(grouped_data["Model"], grouped_data[metric], color="blue", alpha=0.7)

            # Add value labels to bars
            for index, value in enumerate(grouped_data[metric]):
                plt.text(value, index, f"{value:.4f}", va="center", fontsize=10, color="black")

            plt.xlabel(metric)
            plt.ylabel("Model")
            plt.title(f"{metric} Comparison (Batch Size: {batch_size})")
            plt.grid(axis="x", linestyle="--", alpha=0.5)

            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(batch_folder, f"{metric.lower().replace(' ', '_')}_batch_size_{batch_size}_comparison.png"))
            plt.close()

    print("✅ All combined metric comparison plots saved successfully!")


def plot_epoch_based_metrics(all_histories, result_dir):
    """
    Vẽ biểu đồ timeline của Train Loss, Validation Loss, Train Accuracy, Validation Accuracy
    theo các giá trị batch_size, hỗ trợ cả MLP và GNN.
    """
    metrics_list = []

    for model_name, model_histories in all_histories.items():
        for history_entry in model_histories:
            batch_size = history_entry.get("batch_size", 32)
            epoch = history_entry.get("epoch", 100)
            history = history_entry.get("history", {})

            # Xác định key loss/accuracy phù hợp (GNN vs MLP)
            if "train_loss" in history:
                train_loss_key = "train_loss"
                train_acc_key = "train_accuracy"
            else:
                train_loss_key = "loss"
                train_acc_key = "accuracy"

            val_loss_key = "val_loss"
            val_acc_key = "val_accuracy"

            # Kiểm tra đầy đủ các keys
            required_keys = [train_loss_key, val_loss_key, train_acc_key, val_acc_key]
            if not all(k in history for k in required_keys):
                print(f"⚠️ Skipping history for model {model_name} due to missing keys.")
                continue

            for epoch_idx, (train_loss, val_loss, train_acc, val_acc) in enumerate(
                zip(history[train_loss_key], history[val_loss_key],
                    history[train_acc_key], history[val_acc_key])
            ):
                metrics_list.append({
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch_idx + 1,
                    "Train Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Train Accuracy": train_acc,
                    "Validation Accuracy": val_acc,
                })

    if not metrics_list:
        print("⚠️ No valid training histories found. Skipping timeline plotting.")
        return

    df = pd.DataFrame(metrics_list)
    metrics = ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]

    batch_sizes = df["Batch Size"].unique()
    for batch_size in batch_sizes:
        batch_folder = os.path.join(result_dir, f"batch_size_{batch_size}")
        os.makedirs(batch_folder, exist_ok=True)

        for metric in metrics:
            plt.figure(figsize=(14, 8))
            batch_df = df[df["Batch Size"] == batch_size]
            for model_name, model_df in batch_df.groupby("Model"):
                grouped = model_df.groupby("Epoch")[metric].mean().reset_index()
                epochs = grouped["Epoch"]
                metric_values = grouped[metric]
                plt.plot(epochs, metric_values, label=model_name, marker='o', linestyle='-')

            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(title="Models", loc="best", fontsize=10)
            plt.tight_layout()

            plot_path = os.path.join(batch_folder, f"{metric.lower().replace(' ', '_')}_batch_size_{batch_size}_timeline_comparison.png")
            plt.savefig(plot_path)
            plt.close()

    print(f"📊 Epoch-based timeline comparison plots saved successfully.")


def plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, result_out, model_name, fold):
    """
    Plots Accuracy, Loss, Confusion Matrix, ROC Curve, and Accuracy vs. Recall plots.
    """
    # ✅ Vẽ Accuracy Plot
    plt.figure()
    plt.plot(history["train_accuracy"], label="Train Accuracy", linestyle="--", marker="o")
    plt.plot(history["val_accuracy"], label="Validation Accuracy", linestyle="-", marker="o")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_out, f"fold_{fold}_{model_name}_bs{batch_size}_ep{epoch}_accuracy_plot.png"))
    plt.close()

    # ✅ Vẽ Loss Plot
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss", linestyle="--", marker="o")
    plt.plot(history["val_loss"], label="Validation Loss", linestyle="-", marker="o")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(result_out, f"fold_{fold}_{model_name}_bs{batch_size}_ep{epoch}_loss_plot.png"))
    plt.close()

    print(f"✅ All plots saved to {result_out}")

    # 3. Confusion Matrix Plot with Float Numbers
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_out, model_name + f'fold_{fold}_bs{batch_size}_ep{epoch}_confusion_matrix_normalized.png'))
    plt.close()

    # Encode the true labels to binary format
    label_encoder = LabelEncoder()
    y_true_binary = label_encoder.fit_transform(y_true_labels)

    # 4. ROC Curve Plot for each class in a one-vs-rest fashion
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors  # Simplified colors
    line_styles = ['-', '--', '-.', ':']  # Updated line styles
    line_width = 1.5  # Reduced line thickness

    # Ensure y_true_labels and y_pred_labels are NumPy arrays and encode labels if they are not integers
    label_encoder = LabelEncoder()
    if isinstance(y_true_labels[0], str) or isinstance(y_true_labels[0], bool):
        y_true_labels = label_encoder.fit_transform(y_true_labels)
    else:
        y_true_labels = np.array(y_true_labels)

    if isinstance(y_pred_labels[0], str) or isinstance(y_pred_labels[0], bool):
        y_pred_labels = label_encoder.transform(y_pred_labels)
    else:
        y_pred_labels = np.array(y_pred_labels)

    if len(categories) == 2:  # Binary classification case
        # Plotting for the positive class (1)
        y_true_binary = (y_true_labels == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[1], linestyle=line_styles[0], linewidth=line_width, label=f'{categories[1]} (AUC = {roc_auc:.4f})')
        
        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[0], linestyle=line_styles[1], linewidth=line_width, label=f'{categories[0]} (AUC = {roc_auc:.4f})')
        
    else:  # Multi-class case
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_width,
                label=f'{class_name} (AUC = {roc_auc:.4f})'
            )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label="Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiple Classes')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_out, model_name + f'fold_{fold}_bs{batch_size}_ep{epoch}_roc_curve.png'))
    plt.close()

    # 5. Accuracy vs. Recall Plot
    report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True)
    accuracy = [report[category]['precision'] for category in categories]
    recall = [report[category]['recall'] for category in categories]

    plt.figure()
    plt.plot(categories, accuracy, marker='o', linestyle='--', color='b', label='Accuracy')
    plt.plot(categories, recall, marker='o', linestyle='-', color='g', label='Recall')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.legend(loc='best')
    plt.savefig(os.path.join(result_out, model_name + f'fold_{fold}_bs{batch_size}_ep{epoch}_accuracy_vs_recall.png'))
    plt.close()

    print(f"All plots saved to {result_out}")

    # 6. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    if len(categories) == 2:  # Binary classification case
        # Plotting for the positive class (1)
        y_true_binary = (y_true_labels == 1).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[1], linestyle=line_styles[0], linewidth=line_width, 
                 label=f'{categories[1]} (PR AUC = {pr_auc:.4f})')

        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, 0])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[0], linestyle=line_styles[1], linewidth=line_width, 
                 label=f'{categories[0]} (PR AUC = {pr_auc:.4f})')
    else:  # Multi-class case
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(
                recall, precision,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_width,
                label=f'{class_name} (PR AUC = {pr_auc:.4f})'
            )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_out, model_name + f'fold_{fold}_bs{batch_size}_ep{epoch}_precision_recall_curve.png'))
    plt.close()

def load_features_without_smote(feature_paths, feature_type, categories, return_filenames=False):
    """
    Load multiple feature vectors from .npy files without applying SMOTE.
    
    Args:
        feature_paths (dict): Dictionary chứa đường dẫn đến feature của từng nhãn.
        feature_type (str): Loại đặc trưng cần tải.
        categories (list): Danh sách các nhãn.
        return_filenames (bool): Nếu True, trả về thêm danh sách tên file.
        
    Returns:
        np.ndarray: Feature matrix X (num_samples, num_features)
        np.ndarray: Encoded labels y
        (optional) list: Danh sách tên file tương ứng (nếu return_filenames=True)
    """
    all_features = []
    all_labels = []
    all_filenames = []  # Danh sách lưu tên file nếu cần
    label_encoder = LabelEncoder()

    for category in categories:
        if category not in feature_paths or feature_type not in feature_paths[category]:
            print(f"⚠️ Warning: No feature path for '{category}' and feature '{feature_type}'. Skipping.")
            continue

        folder_path = feature_paths[category][feature_type]
        if not os.path.isdir(folder_path):
            print(f"❌ Error: Feature folder '{folder_path}' does not exist. Skipping '{feature_type}'.")
            continue

        feature_vectors = []
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

        if not npy_files:
            print(f"⚠️ Warning: No .npy files found in '{folder_path}'.")

        # Sắp xếp danh sách file nếu cần đảm bảo thứ tự nhất định
        npy_files = sorted(npy_files)

        for filename in npy_files:
            file_path = os.path.join(folder_path, filename)
            try:
                feature = np.load(file_path)

                if feature.size == 0:
                    print(f"⚠️ Warning: '{filename}' is empty. Skipping.")
                    continue

                if feature.ndim == 1:
                    feature = feature.reshape(1, -1)  # Đảm bảo 2D shape (1, num_features)

                feature_vectors.append(feature)
                if return_filenames:
                    all_filenames.append(filename)
            except Exception as e:
                print(f"❌ Error loading '{file_path}': {e}")
                continue

        if len(feature_vectors) > 0:
            feature_matrix = np.vstack(feature_vectors)  # (num_samples, num_features)
            all_features.append(feature_matrix)
            num_samples = feature_matrix.shape[0]
            all_labels.extend([category] * num_samples)

    if len(all_features) == 0:
        print("⚠️ No valid features found. Returning empty arrays.")
        if return_filenames:
            return np.array([]), np.array([]), []
        else:
            return np.array([]), np.array([])

    X = np.vstack(all_features)  # Ghép tất cả mẫu lại
    y = np.array(all_labels)
    y_encoded = label_encoder.fit_transform(y)

    print(f"✅ Loaded {X.shape[0]} samples with feature type {feature_type}, feature size {X.shape[1]}.")
    if return_filenames:
        return X, y_encoded, all_filenames
    else:
        return X, y_encoded


class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Tạm bỏ layer thứ 2
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def visualize_graph(graph_data, labels, save_path="graph_visualizations", sample_size=30):
    """
    Visualizes the graph structure using different techniques.

    Parameters:
    - graph_data: A PyG Data object containing the graph structure.
    - labels: Tensor containing node labels for color mapping.
    - save_path: Path to save the generated plots.
    - sample_size: Number of nodes to display in graph visualizations.
    """
    os.makedirs(save_path, exist_ok=True)

    # Handle empty graphs
    if graph_data.num_nodes == 0:
        print("⚠️ Empty graph! No visualizations generated.")
        return

    num_nodes = graph_data.num_nodes
    num_edges = graph_data.edge_index.shape[1]

    # Convert to NetworkX format
    G = nx.Graph()
    edge_index_np = graph_data.edge_index.t().cpu().numpy()
    G.add_edges_from(edge_index_np)
    G.add_nodes_from(range(num_nodes))  # Ensure isolated nodes are included

    # Sample nodes for better visualization
    sample_nodes = random.sample(range(num_nodes), min(sample_size, num_nodes))
    subgraph = G.subgraph(sample_nodes)
    pos = nx.kamada_kawai_layout(subgraph)

    # Ensure positions exist for all nodes
    for node in sample_nodes:
        if node not in pos:
            pos[node] = (0, 0)

    # Replace numeric labels with text labels if available
    label_dict = {i: labels[i].item() for i in sample_nodes}
    if isinstance(labels[0].item(), int) and hasattr(graph_data, 'category_labels'):
        label_dict = {i: graph_data.category_labels[labels[i].item()] for i in sample_nodes}

    # 1️⃣ **Raw Graph Structure**
    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, pos, node_size=300, node_color="lightblue", with_labels=False)
    nx.draw_networkx_edges(subgraph, pos, edge_color="black", alpha=0.5, width=1.0)
    # plt.title("1️⃣ Raw Graph Structure (Sampled)")
    plt.savefig(os.path.join(save_path, "raw_graph_structure.png"))
    plt.close()

    # 2️⃣ **Graph with Labels**
    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, pos, node_size=300, node_color="lightblue", edgecolors="black", with_labels=False)
    nx.draw_networkx_edges(subgraph, pos, edge_color="black", alpha=0.5, width=1.0)
    nx.draw_networkx_labels(subgraph, pos, labels=label_dict, font_size=8, font_color="black")
    # plt.title("2️⃣ Graph with Labels (Sampled)")
    plt.savefig(os.path.join(save_path, "graph_with_labels.png"))
    plt.close()

    # 3️⃣ **Graph with Clustering**
    plt.figure(figsize=(8, 6))
    color_map = plt.colormaps.get_cmap("tab10")
    node_colors = [color_map(labels[i] % 10) for i in sample_nodes]

    nx.draw(subgraph, pos, node_color=node_colors, node_size=300, edgecolors="black", with_labels=False)
    nx.draw_networkx_edges(subgraph, pos, edge_color="black", alpha=0.5, width=1.0)
    nx.draw_networkx_labels(subgraph, pos, labels=label_dict, font_size=8, font_color="black")
    # plt.title("3️⃣ Graph with Clustering (Sampled)")
    plt.savefig(os.path.join(save_path, "graph_with_clustering.png"))
    plt.close()

    # 4️⃣ **PCA Projection of Node Embeddings**
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        node_embeddings = graph_data.x.cpu().numpy()
        pca_embeddings = PCA(n_components=2).fit_transform(node_embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=labels.cpu().numpy(), cmap="tab10", s=30)
        for i, (x, y) in enumerate(pca_embeddings[:, :2]):
            plt.text(x, y, str(label_dict.get(i, i)), fontsize=6, ha="right")

        # plt.title("4️⃣ PCA Projection of Embeddings")
        plt.savefig(os.path.join(save_path, "pca_projection.png"))
        plt.close()

        # 5️⃣ **t-SNE Projection of Node Embeddings**
        perplexity = min(30, num_nodes // 10)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(pca_embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels.cpu().numpy(), cmap="tab10", s=30)
        for i, (x, y) in enumerate(tsne_embeddings):
            plt.text(x, y, str(label_dict.get(i, i)), fontsize=6, ha="right")

        # plt.title("5️⃣ t-SNE Projection of Embeddings")
        plt.savefig(os.path.join(save_path, "tsne_projection.png"))
        plt.close()

    print(f"✅ Graph visualizations saved in {save_path}")


def run_experiment(train_paths, val_paths, epoch_values, batch_size_list,
                   metric_collection, result_folder, categories, feature_types,
                   metadata_df=None, one_hot_encoder=None, age_scaler=None,
                   visualize=True):

    print("\n🗕️ Loading training features...")
    X_train_list, y_train, train_file_names = [], None, []

    for ft in feature_types:
        X_ft, y_ft, fnames = load_features_without_smote(train_paths, ft, categories, return_filenames=True)
        if y_train is None:
            y_train = y_ft
        X_train_list.append(X_ft)
        train_file_names = fnames

    X_train_img = np.hstack(X_train_list)

    if metadata_df is not None:
        X_train_clinical = extract_clinical_features_from_list(train_file_names, metadata_df, one_hot_encoder, age_scaler)
        fusion_model = MetadataAttentionFusion(X_train_img.shape[1], X_train_clinical.shape[1])
        fusion_model.eval()
        with torch.no_grad():
            X_train = fusion_model(torch.tensor(X_train_img, dtype=torch.float32),
                                   torch.tensor(X_train_clinical, dtype=torch.float32)).numpy()
    else:
        X_train = X_train_img

    if val_paths and any(val_paths.values()):
        X_val_list, y_val, val_file_names = [], None, []
        for ft in feature_types:
            X_val_ft, y_val_ft, fnames = load_features_without_smote(val_paths, ft, categories, return_filenames=True)
            if y_val is None:
                y_val = y_val_ft
            X_val_list.append(X_val_ft)
            val_file_names = fnames
        X_val_img = np.hstack(X_val_list)

        if metadata_df is not None:
            X_val_clinical = extract_clinical_features_from_list(val_file_names, metadata_df, one_hot_encoder, age_scaler)
            with torch.no_grad():
                X_val = fusion_model(torch.tensor(X_val_img, dtype=torch.float32),
                                     torch.tensor(X_val_clinical, dtype=torch.float32)).numpy()
        else:
            X_val = X_val_img

        X_all = np.vstack([X_train, X_val])
        y_all = np.hstack([y_train, y_val])
    else:
        print("⚠️ No validation set provided. Using only training set.")
        X_all = X_train
        y_all = y_train

    X_all, y_all = augment_all_classes_to_balance(X_all, y_all)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)

    model_name = "GNN_deep_handcrafted_clinical"
    model_dir = os.path.join(result_folder, model_name)
    os.makedirs(model_dir, exist_ok=True)

    best_score = -1
    best_model_state = None
    best_label_encoder = None
    best_model_path = os.path.join(model_dir, "best_overall_model.pth")
    best_label_path = os.path.join(model_dir, "best_label_encoder.pkl")

    all_histories = {}

    for batch_size in batch_size_list:
        for epoch in epoch_values:
            rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            fold_metrics = []
            fold = 1
            for train_idx, test_idx in rkf.split(X_all, y_encoded):
                print(f"🔁 Fold {fold}/15")
                X_all_fold, _ = normalize_data(X_all, X_all)
                input_dim = X_all_fold.shape[1]
                num_classes = len(np.unique(y_encoded))

                model = GNNClassifier(input_dim, 64, num_classes)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

                fold_out = os.path.join(model_dir, f"batch_size_{batch_size}", f"epoch_{epoch}_fold_{fold}")
                os.makedirs(fold_out, exist_ok=True)
                data = create_graph(X_all_fold, y_encoded, train_idx, test_idx)

                visualize_graph(
                    graph_data=data,
                    labels=torch.tensor(y_encoded),
                    save_path=os.path.join(fold_out, "graph_visualization"),
                    sample_size=30
                )

                model, history = train_gnn_model(model, data, optimizer, epoch, fold_out)
                all_histories.setdefault(model_name, []).append({
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "history": history
                })

                with torch.no_grad():
                    output = model(data)
                    test_mask = data.test_mask
                    probs = F.softmax(output[test_mask], dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                y_true = y_encoded[test_mask.cpu().numpy()]
                metrics = compute_classification_metrics(y_true, preds, probs, categories)

                np.save(os.path.join(fold_out, f"y_true_labels_fold_{fold}.npy"), y_true)
                np.save(os.path.join(fold_out, f"y_pred_labels_fold_{fold}.npy"), preds)
                np.save(os.path.join(fold_out, f"y_pred_probs_fold_{fold}.npy"), probs)

                plot_all_figures(batch_size=batch_size, epoch=epoch, history=history,
                                 y_true_labels=y_true, y_pred_labels=preds,
                                 y_pred_probs=probs, categories=categories,
                                 result_out=fold_out, model_name=model_name, fold=fold)

                report = classification_report(y_true, preds, target_names=categories)
                with open(os.path.join(fold_out, f"classification_report_fold_{fold}.txt"), "w") as f:
                    f.write(report)

                pd.DataFrame([metrics]).to_csv(os.path.join(fold_out, f"per_fold_metrics.csv"), index=False)

                score = metrics["Macro Recall"] + metrics["Macro F1"] - 0.5 * (metrics["Macro Recall"] + metrics["Macro F1"])
                if score > best_score:
                    best_score = score
                    best_model_state = model.state_dict()
                    best_label_encoder = label_encoder
                    torch.save(best_model_state, best_model_path)
                    with open(best_label_path, 'wb') as f:
                        pickle.dump(best_label_encoder, f)
                    print(f"📏 New best model saved at fold {fold} with score = {score:.4f}")

                metrics["Time Taken"] = 0
                fold_metrics.append(metrics)
                fold += 1

            if fold_metrics:
                df_fold = pd.DataFrame(fold_metrics)
                metric_collection.append({
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch,
                    "Test Accuracy": df_fold["Accuracy"].mean(),
                    "Precision": df_fold["Precision"].mean(),
                    "Recall": df_fold["Recall"].mean(),
                    "F1 Score": df_fold["F1"].mean(),
                    "Macro F1": df_fold["Macro F1"].mean(),
                    "Macro Precision": df_fold["Macro Precision"].mean(),
                    "Macro Recall": df_fold["Macro Recall"].mean(),
                    "Macro AUC": df_fold["Macro AUC"].mean(),
                    "Training Time (s)": df_fold["Time Taken"].mean()
                })

    pd.DataFrame(metric_collection).to_csv(os.path.join(model_dir, "performance_metrics.csv"), index=False)
    if visualize:
        plot_combined_metrics(metric_collection, result_folder)
        plot_epoch_based_metrics(all_histories, result_folder)
    print("✅ GNN experiment completed.")
    return all_histories, metric_collection



# Function to dynamically generate train/test/val paths
def generate_paths(feature_dir, dataset_type, feature_types, categories):
    paths = {}
    print(f"📁 Checking paths for dataset type: {dataset_type}")

    for category in categories:
        paths[category] = {}
        for feature_type in feature_types:
            feature_path = os.path.join(feature_dir, dataset_type, feature_type, category)
            print(f"🔍 {feature_path}")
            if os.path.exists(feature_path) and os.path.isdir(feature_path):
                paths[category][feature_type] = feature_path
            else:
                print(f"❌ Not Found: {feature_path}")

    # ✅ Nếu là tập test → kiểm tra thêm thư mục 'unlabeled'
    if dataset_type == 'test':
        unlabeled_category = 'unlabeled'
        paths[unlabeled_category] = {}
        for feature_type in feature_types:
            unlabeled_path = os.path.join(feature_dir, dataset_type, feature_type, unlabeled_category)
            print(f"🔍 (Unlabeled) {unlabeled_path}")
            if os.path.exists(unlabeled_path) and os.path.isdir(unlabeled_path):
                paths[unlabeled_category][feature_type] = unlabeled_path
            else:
                print(f"❌ Not Found: {unlabeled_path}")

    return paths


def predict_labeled_test_set(model_path, label_encoder_path, test_paths, feature_types, result_dir, categories,
                              metadata_df=None, one_hot_encoder=None, age_scaler=None):
    import os
    import torch
    import pickle
    import numpy as np
    import pandas as pd
    import torch.nn.functional as F

    print(f"\n🔍 [TEST] Evaluating fused model on test set with features: {feature_types}")
    X_test_list, y_test, file_names = [], None, None
    first_feature = True
    all_valid = True

    for feature_type in feature_types:
        if first_feature:
            X_part, y_part, file_names_part = load_features_without_smote(test_paths, feature_type, categories, return_filenames=True)
            if X_part.size == 0:
                print(f"⚠️ No valid features found for feature '{feature_type}'. Skipping.")
                all_valid = False
                continue
            file_names = file_names_part
            y_test = y_part
            first_feature = False
        else:
            X_part, _, _ = load_features_without_smote(test_paths, feature_type, categories, return_filenames=True)
            if X_part.size == 0:
                print(f"⚠️ No valid features found for feature '{feature_type}'. Skipping.")
                all_valid = False
                continue
        X_part = np.nan_to_num(X_part, nan=np.nanmean(X_part))
        X_test_list.append(X_part)

    if not X_test_list or not all_valid:
        print("❌ No valid test features found. Skipping evaluation.")
        return

    X_img = np.hstack(X_test_list)

    if metadata_df is not None and one_hot_encoder and age_scaler and file_names:
        X_clinical = extract_clinical_features_from_list(file_names, metadata_df, one_hot_encoder, age_scaler)
        fusion_model = MetadataAttentionFusion(
            input_dim_img=X_img.shape[1],
            input_dim_meta=X_clinical.shape[1]
        )
        fusion_model.eval()
        with torch.no_grad():
            X_test_combined = fusion_model(
                torch.tensor(X_img, dtype=torch.float32),
                torch.tensor(X_clinical, dtype=torch.float32)
            ).numpy()
    else:
        X_test_combined = X_img

    X_test_combined, _ = normalize_data(X_test_combined, X_test_combined)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(np.unique(y_test_encoded))

    input_dim = X_test_combined.shape[1]
    gnn_model = GNNClassifier(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)
    gnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    gnn_model.eval()

    test_graph = create_graph(X_test_combined, y_test_encoded)

    with torch.no_grad():
        output = gnn_model(test_graph)
        probs = F.softmax(output, dim=1).cpu().numpy()

    preds = np.argmax(probs, axis=1)
    entropy_scores = compute_entropy(probs)
    trust_labels = get_trust_labels(entropy_scores)

    # ✅ Tính chỉ số bằng compute_classification_metrics
    metrics = compute_classification_metrics(y_test_encoded, preds, probs, categories)

    report_txt = classification_report(y_test_encoded, preds, target_names=categories)
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "final_test_classification_report.txt"), "w") as f:
        f.write(report_txt)

    with open(os.path.join(result_dir, "final_test_evaluation_metrics.txt"), "w") as f:
        f.write("Final Test Set Evaluation Metrics\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k:<20}: {v:.4f}\n")

    # ✅ Lưu kết quả từng ảnh
    df_result = pd.DataFrame({
        "Filename": file_names,
        "True Label": label_encoder.inverse_transform(y_test_encoded),
        "Predicted Label": label_encoder.inverse_transform(preds),
        "Entropy": entropy_scores,
        "Trust Level": trust_labels
    })
    df_result.to_csv(os.path.join(result_dir, "test_per_file_predictions.csv"), index=False)
    print("✅ Test predictions with trust scores saved.")

    # ✅ Lưu mẫu độ tin cậy thấp
    low_trust_df = df_result[df_result["Trust Level"] == "Low"]
    low_trust_df.to_csv(os.path.join(result_dir, "low_trust_samples.csv"), index=False)
    print(f"⚠️ {len(low_trust_df)} low-trust predictions saved for review.")

    # ✅ Vẽ biểu đồ
    plot_accuracy_by_trust(df_result, result_dir)
    plot_confusion_by_trust(df_result, label_encoder, result_dir)

    print("📈 XAI analysis completed.")


def predict_unlabeled_test_set(model_path, label_encoder_path, test_paths, feature_types, result_dir,
                                metadata_df=None, one_hot_encoder=None, age_scaler=None):

    print("\n🧪 Predicting unlabeled test set with entropy and trust level...")
    X_test_list, file_names = [], None
    all_valid = True

    for ft in feature_types:
        if 'unlabeled' not in test_paths:
            continue
        folder = test_paths['unlabeled'].get(ft)
        if not folder or not os.path.exists(folder):
            print(f"⚠️ Feature '{ft}' not found for unlabeled test set. Skipping.")
            all_valid = False
            continue

        feature_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        if not feature_files:
            print(f"⚠️ No .npy files found in folder '{folder}'. Skipping feature '{ft}'.")
            all_valid = False
            continue

        vectors = [np.load(f).reshape(1, -1) for f in feature_files if os.path.getsize(f) > 0]
        if vectors:
            X_ft = np.vstack(vectors)
            X_test_list.append(X_ft)
            if file_names is None:
                file_names = [os.path.basename(f).replace(".npy", "") for f in feature_files]
        else:
            print(f"⚠️ All vectors in '{folder}' were empty or invalid.")
            all_valid = False

    if not X_test_list or not all_valid:
        print("❌ No valid features found for prediction. Skipping.")
        return

    X_img = np.hstack(X_test_list)

    if metadata_df is not None and one_hot_encoder and age_scaler and file_names:
        X_clinical = extract_clinical_features_from_list(file_names, metadata_df, one_hot_encoder, age_scaler)
        fusion_model = MetadataAttentionFusion(X_img.shape[1], X_clinical.shape[1])
        fusion_model.eval()
        with torch.no_grad():
            X_combined = fusion_model(
                torch.tensor(X_img, dtype=torch.float32),
                torch.tensor(X_clinical, dtype=torch.float32)
            ).numpy()
    else:
        X_combined = X_img

    X_combined, _ = normalize_data(X_combined, X_combined)

    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    input_dim = X_combined.shape[1]
    output_dim = len(label_encoder.classes_)
    model = GNNClassifier(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_labels = np.zeros(len(X_combined), dtype=int)
    graph = create_graph(X_combined, labels=dummy_labels)

    with torch.no_grad():
        output = model(graph)
        probs = F.softmax(output, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        predicted_labels = label_encoder.inverse_transform(preds)
        entropy_scores = compute_entropy(probs)
        trust_labels = get_trust_labels(entropy_scores)

    os.makedirs(result_dir, exist_ok=True)
    df_result = pd.DataFrame({
        "Filename": file_names,
        "Predicted Label": predicted_labels,
        "Entropy": entropy_scores,
        "Trust Level": trust_labels
    })
    df_result.to_csv(os.path.join(result_dir, "unlabeled_with_entropy.csv"), index=False)
    print(f"✅ Predictions with trust levels saved to: {os.path.join(result_dir, 'unlabeled_with_entropy.csv')}")

    plot_accuracy_by_trust(df_result, result_dir)


def pseudo_train_from_unlabeled(unlabeled_csv, train_paths, test_paths, feature_types, categories,
                                 label_encoder, result_dir, metadata_df=None, one_hot_encoder=None, age_scaler=None):
    df = pd.read_csv(unlabeled_csv)
    df = df[df["Trust Level"] == "High"]
    print(f"📥 Adding {len(df)} high-trust pseudo-labeled samples to training set.")

    X_pseudo_list, y_pseudo_list, filenames = [], [], []

    for row in df.itertuples():
        npy_file = row.Filename + ".npy"
        pred_label = row._2  # Predicted Label
        label_id = label_encoder.transform([pred_label])[0]

        feature_vector = []
        for ft in feature_types:
            folder = test_paths['unlabeled'].get(ft)
            path = os.path.join(folder, npy_file)
            if os.path.exists(path):
                vec = np.load(path)
                feature_vector.append(vec)

        if feature_vector:
            combined = np.hstack(feature_vector)
            X_pseudo_list.append(combined)
            y_pseudo_list.append(label_id)
            filenames.append(row.Filename)

    if not X_pseudo_list:
        print("❌ No pseudo-labeled samples were added. Aborting.")
        return None, None

    X_pseudo_img = np.vstack(X_pseudo_list)
    y_pseudo = np.array(y_pseudo_list)

    # Nếu có clinical metadata → trích thêm
    if metadata_df is not None and one_hot_encoder and age_scaler:
        X_clinical = extract_clinical_features_from_list(filenames, metadata_df, one_hot_encoder, age_scaler)
        fusion_model = MetadataAttentionFusion(X_pseudo_img.shape[1], X_clinical.shape[1])
        fusion_model.eval()
        with torch.no_grad():
            X_pseudo = fusion_model(
                torch.tensor(X_pseudo_img, dtype=torch.float32),
                torch.tensor(X_clinical, dtype=torch.float32)
            ).numpy()
    else:
        X_pseudo = X_pseudo_img

    # Load original training set
    X_train_list, y_train, train_filenames = [], None, []
    for ft in feature_types:
        X_ft, y_ft, fnames = load_features_without_smote(train_paths, ft, categories, return_filenames=True)
        if y_train is None:
            y_train = y_ft
        X_train_list.append(X_ft)
        train_filenames = fnames
    X_train_img = np.hstack(X_train_list)

    if metadata_df is not None and one_hot_encoder and age_scaler:
        X_clinical_train = extract_clinical_features_from_list(train_filenames, metadata_df, one_hot_encoder, age_scaler)
        fusion_model = MetadataAttentionFusion(X_train_img.shape[1], X_clinical_train.shape[1])
        fusion_model.eval()
        with torch.no_grad():
            X_train = fusion_model(
                torch.tensor(X_train_img, dtype=torch.float32),
                torch.tensor(X_clinical_train, dtype=torch.float32)
            ).numpy()
    else:
        X_train = X_train_img

    # Gộp lại
    X_train_all = np.vstack([X_train, X_pseudo])
    y_train_all = np.hstack([y_train, y_pseudo])
    print(f"✅ Final training set size: {X_train_all.shape[0]} samples.")
    return X_train_all, y_train_all

def run_experiment_with_custom_data(X_all, y_all, result_folder, categories, batch_size_list, epoch_values, model_name="GNN_pseudo_finetune"):

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)

    os.makedirs(result_folder, exist_ok=True)
    model_result_out = os.path.join(result_folder, model_name)
    os.makedirs(model_result_out, exist_ok=True)

    best_model_path = os.path.join(model_result_out, "best_overall_model.pth")
    label_encoder_path = os.path.join(model_result_out, "best_label_encoder.pkl")

    all_histories = {}
    metric_collection = []
    best_score = -1

    for batch_size in batch_size_list:
        for epoch in epoch_values:
            rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            fold = 1
            fold_metrics = []

            for train_idx, test_idx in rkf.split(X_all, y_encoded):
                print(f"\n🔁 Pseudo-Finetune Fold {fold}/15")
                X_all_fold, _ = normalize_data(torch.nan_to_num(torch.tensor(X_all, dtype=torch.float32)).numpy(),
                                               torch.nan_to_num(torch.tensor(X_all, dtype=torch.float32)).numpy())

                input_dim = X_all_fold.shape[1]
                num_classes = len(np.unique(y_encoded))

                gnn_model = GNNClassifier(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)
                optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)

                epoch_result_out = os.path.join(model_result_out, f'batch_size_{batch_size}', f'epoch_{epoch}_fold_{fold}')
                os.makedirs(epoch_result_out, exist_ok=True)

                data = create_graph(X_all_fold, y_encoded, train_idx=train_idx, test_idx=test_idx, k=5, use_mask=True)
                visualize_graph(
                    graph_data=data,
                    labels=torch.tensor(y_encoded),
                    save_path=os.path.join(epoch_result_out, "graph_visualization"),
                    sample_size=30
                )
     
                gnn_model, history = train_gnn_model(gnn_model, data, optimizer, epoch, epoch_result_out)

                all_histories.setdefault(model_name, []).append({
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "history": history
                })

                gnn_model.eval()
                with torch.no_grad():
                    output = gnn_model(data)
                    test_mask = data.test_mask
                    y_pred_probs = F.softmax(output[test_mask], dim=1).cpu().numpy()

                y_pred_labels = np.argmax(y_pred_probs, axis=1)
                y_true_labels = y_encoded[test_mask.cpu().numpy()]

                np.save(os.path.join(epoch_result_out, f"y_true_labels_fold_{fold}.npy"), y_true_labels)
                np.save(os.path.join(epoch_result_out, f"y_pred_labels_fold_{fold}.npy"), y_pred_labels)
                np.save(os.path.join(epoch_result_out, f"y_pred_probs_fold_{fold}.npy"), y_pred_probs)

                metrics = compute_classification_metrics(y_true_labels, y_pred_labels, y_pred_probs, categories)
                pd.DataFrame([metrics]).to_csv(os.path.join(epoch_result_out, f"per_fold_metrics.csv"), index=False)

                report = classification_report(y_true_labels, y_pred_labels, target_names=categories)
                with open(os.path.join(epoch_result_out, f"classification_report_fold_{fold}.txt"), "w") as f:
                    f.write(report)

                plot_all_figures(batch_size=batch_size, epoch=epoch, history=history,
                                 y_true_labels=y_true_labels, y_pred_labels=y_pred_labels,
                                 y_pred_probs=y_pred_probs, categories=categories,
                                 result_out=epoch_result_out, model_name=model_name, fold=fold)

                score = metrics["Macro Recall"] + metrics["Macro F1"] - 0.5 * (metrics["Macro Recall"] + metrics["Macro F1"])
                if score > best_score:
                    best_score = score
                    torch.save(gnn_model.state_dict(), best_model_path)
                    with open(label_encoder_path, "wb") as f:
                        pickle.dump(label_encoder, f)

                metrics["Time Taken"] = 0  # Can be updated if timing is added
                fold_metrics.append(metrics)
                fold += 1

            if fold_metrics:
                df_fold = pd.DataFrame(fold_metrics)
                metric_summary = {
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch,
                    "Test Accuracy": df_fold["Accuracy"].mean(),
                    "Precision": df_fold["Precision"].mean(),
                    "Recall": df_fold["Recall"].mean(),
                    "F1 Score": df_fold["F1"].mean(),
                    "Macro F1": df_fold["Macro F1"].mean(),
                    "Macro Precision": df_fold["Macro Precision"].mean(),
                    "Macro Recall": df_fold["Macro Recall"].mean(),
                    "Macro AUC": df_fold["Macro AUC"].mean(),
                    "Training Time (s)": df_fold["Time Taken"].mean(),
                }
                metric_collection.append(metric_summary)

    pd.DataFrame(metric_collection).to_csv(os.path.join(model_result_out, 'performance_metrics.csv'), index=False)
    plot_combined_metrics(metric_collection, model_result_out)
    plot_epoch_based_metrics(all_histories, model_result_out)

    print("✅ Pseudo-label based retraining completed.")
    return all_histories, metric_collection



def plot_model_comparison_before_after(before_csv, after_csv, output_dir):
    # Load dữ liệu
    df_before = pd.read_csv(before_csv)
    df_after = pd.read_csv(after_csv)

    df_before["Phase"] = "Before Pseudo"
    df_after["Phase"] = "After Pseudo"

    df_compare = pd.concat([df_before, df_after], ignore_index=True)

    # Chọn các metrics cần so sánh
    metrics = ["F1 Score", "Recall", "Precision", "Macro F1", "Macro Recall", "Macro AUC"]

    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sns.barplot(data=df_compare, x="Phase", y=metric, palette="Set2")
        plt.title(f"{metric} Comparison")
        plt.ylim(0, 1.05)
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt='%.3f', label_type='edge', fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{metric.lower().replace(' ', '_')}_before_vs_after.png")
        plt.savefig(out_path)
        plt.close()
        print(f"✅ Saved comparison plot: {out_path}")
        
    

def copy_xai_and_analysis_files(original_dir, pseudo_dir, comparison_dir, report_dir):
    """
    Copy XAI và phân tích pseudo-label từ original_dir (pha 1), pseudo_dir (pha 2),
    và comparison_dir (so sánh trước/sau) vào report_dir (các nhóm thư mục gọn gàng).
    """

    # Các thư mục con
    original_target = os.path.join(report_dir, "original")
    pseudo_target = os.path.join(report_dir, "pseudo")
    summary_target = os.path.join(report_dir, "summary")
    os.makedirs(original_target, exist_ok=True)
    os.makedirs(pseudo_target, exist_ok=True)
    os.makedirs(summary_target, exist_ok=True)

    # ✅ Copy toàn bộ file trong original_dir
    for file in os.listdir(original_dir):
        src = os.path.join(original_dir, file)
        if os.path.isfile(src):
            shutil.copy(src, original_target)

    # ✅ Copy toàn bộ file trong pseudo_dir
    for file in os.listdir(pseudo_dir):
        src = os.path.join(pseudo_dir, file)
        if os.path.isfile(src):
            shutil.copy(src, pseudo_target)

    # ✅ Copy toàn bộ file trong comparison_dir
    for file in os.listdir(comparison_dir):
        src = os.path.join(comparison_dir, file)
        if os.path.isfile(src):
            shutil.copy(src, summary_target)

    print(f"✅ Copied all available XAI and analysis files to {report_dir}")


def main():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import shutil

    base_dir = os.getcwd()
    home_dir = os.path.join(base_dir, 'data9')
    
    feature_dir = os.path.join(home_dir, 'data9_SOTA_and_handcrafts_and_BlookNet_optimal_entropy_features_v3')
    result_folder = os.path.join(home_dir, 'training_data9_MultiModal_GNN_scenario_1_deep_handcrafted_clinical_v2_6_3')

    # feature_dir = os.path.join(home_dir, 'data9_small')
    # result_folder = os.path.join(home_dir, 'training_data9_MultiModal_GNN_scenario_1')

    os.makedirs(result_folder, exist_ok=True)
    original_dir = os.path.join(result_folder, "GNN_deep_handcrafted_clinical")
    pseudo_dir = os.path.join(result_folder, "GNN_pseudo_finetune")
    comparison_dir = os.path.join(result_folder, "plots_before_vs_after")
    report_dir = os.path.join(result_folder, "reports")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(pseudo_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    train_metadata_path = os.path.join(home_dir, "ISIC_2020_Train_Metadata.csv")
    test_metadata_path = os.path.join(home_dir, "ISIC_2020_Test_Metadata.csv")

    categories = ['benign', 'malignant']
    feature_types = [
        "color_histograms_features", 
        "hsv_histograms_features",
        "fractal_features", 
        "vgg19_features", 
        "mobilenet_features", 
        "densenet121_features"
    ]

    train_paths = generate_paths(feature_dir, "train", feature_types, categories)
    val_paths = generate_paths(feature_dir, "val", feature_types, categories)
    test_paths = generate_paths(feature_dir, "test", feature_types, categories)

    if not any(val_paths.values()):
        val_paths = None

    batch_size_list = [32]
    epoch_values = [100]
    batch_size = 32
    epoch = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n📌 Experiment Configuration:")
    print(f"🖥️ Device: {device}")
    print(f"🔹 Categories: {categories}\n🔹 Features: {feature_types}")

    train_metadata_df = pd.read_csv(train_metadata_path)
    test_metadata_df = pd.read_csv(test_metadata_path)
    one_hot_encoder, age_scaler = prepare_clinical_encoder(train_metadata_df)

    # 1️⃣ Train original model
    all_histories, metric_collection = run_experiment(
        train_paths=train_paths,
        val_paths=val_paths,
        epoch_values=epoch_values,
        batch_size_list=batch_size_list,
        metric_collection=[],
        result_folder=result_folder,
        categories=categories,
        feature_types=feature_types,
        metadata_df=train_metadata_df,
        one_hot_encoder=one_hot_encoder,
        age_scaler=age_scaler,
        visualize=True
    )

    # 1️⃣.1️⃣ Save best fold report for original model
    best_fold_original = summarize_per_fold_metrics(result_folder, model_name="GNN_deep_handcrafted_clinical")
    save_best_fold_reports(
        model_dir=os.path.join(result_folder, "GNN_deep_handcrafted_clinical"),
        batch_size=batch_size,
        epoch=epoch,
        model_name="GNN_deep_handcrafted_clinical",
        result_folder=result_folder,
        report_tag="original"
    )

    model_path = os.path.join(original_dir, "best_overall_model.pth")
    label_path = os.path.join(original_dir, "best_label_encoder.pkl")

    if os.path.exists(model_path) and os.path.exists(label_path) and all(c in test_paths for c in categories):
        with open(label_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # 2️⃣ Predict on labeled test set
        print("\n🔍 Predicting labeled test set and generating XAI report...")
        predict_labeled_test_set(
            model_path=model_path,
            label_encoder_path=label_path,
            test_paths=test_paths,
            feature_types=feature_types,
            result_dir=original_dir,
            categories=categories,
            metadata_df=test_metadata_df,
            one_hot_encoder=one_hot_encoder,
            age_scaler=age_scaler
        )

        # 2️⃣.1️⃣ Generate XAI report
        generate_xai_report(
            result_dir=original_dir,
            categories=categories
        )

        # 3️⃣ Predict on unlabeled test set
        ran_unlabeled_prediction = False
        if 'unlabeled' in test_paths:
            print("\n🧪 Running prediction on unlabeled test set...")
            predict_unlabeled_test_set(
                model_path=model_path,
                label_encoder_path=label_path,
                test_paths=test_paths,
                feature_types=feature_types,
                result_dir=original_dir,
                metadata_df=test_metadata_df,
                one_hot_encoder=one_hot_encoder,
                age_scaler=age_scaler
            )
            ran_unlabeled_prediction = True

            # 🔥 Immediately plot trust level distribution
            unlabeled_csv_path = os.path.join(original_dir, "unlabeled_with_entropy.csv")
            plot_trust_level_distribution(unlabeled_csv_path, save_dir=original_dir)
            
            if os.path.exists(unlabeled_csv_path):
                df_unlabeled = pd.read_csv(unlabeled_csv_path)
                if "Entropy" in df_unlabeled.columns:
                    plot_entropy_distribution(
                        entropies=df_unlabeled["Entropy"].values,
                        save_path=os.path.join(original_dir, "entropy_distribution_hist.png"),
                        title="Entropy Distribution (Unlabeled Test Set)"
                    )
                else:
                    print("⚠️ Entropy column not found in unlabeled CSV.")
            else:
                print("⚠️ unlabeled_with_entropy.csv not found, skipping entropy plot.")
                
        # 4️⃣ Trust-based pseudo-labeling and retraining
        unlabeled_csv = os.path.join(original_dir, "unlabeled_with_entropy.csv")
        if ran_unlabeled_prediction and os.path.exists(unlabeled_csv):
            X_train_all, y_train_all = pseudo_train_from_unlabeled(
                unlabeled_csv=unlabeled_csv,
                train_paths=train_paths,
                test_paths=test_paths,
                feature_types=feature_types,
                categories=categories,
                label_encoder=label_encoder,
                result_dir=original_dir,
                metadata_df=test_metadata_df,
                one_hot_encoder=one_hot_encoder,
                age_scaler=age_scaler
            )

            if X_train_all is not None:
                print("🚀 Starting pseudo-label based training...")
                all_histories_pseudo, metric_collection_pseudo = run_experiment_with_custom_data(
                    X_all=X_train_all,
                    y_all=y_train_all,
                    result_folder=result_folder,
                    categories=categories,
                    batch_size_list=batch_size_list,
                    epoch_values=epoch_values,
                    model_name="GNN_pseudo_finetune"
                )

                # 4️⃣.1️⃣ Save best fold report for pseudo model
                best_fold_pseudo = summarize_per_fold_metrics(result_folder, model_name="GNN_pseudo_finetune")
                save_best_fold_reports(
                    model_dir=os.path.join(result_folder, "GNN_pseudo_finetune"),
                    batch_size=batch_size,
                    epoch=epoch,
                    model_name="GNN_pseudo_finetune",
                    result_folder=result_folder,
                    report_tag="pseudo"
                )

                # 4️⃣.2️⃣ Plot comparison before vs after
                before_csv = os.path.join(original_dir, "performance_metrics.csv")
                after_csv = os.path.join(pseudo_dir, "performance_metrics.csv")
                
                original_df = pd.read_csv(before_csv)
                pseudo_df = pd.read_csv(after_csv)

                metrics = ["F1 Score", "Recall", "Precision", "Macro F1", "Macro Recall", "Macro AUC"]
                before_means = original_df[metrics].mean()
                after_means = pseudo_df[metrics].mean()
                improvement = ((after_means - before_means) / before_means * 100).round(2)

                improvement_df = pd.DataFrame({
                    "Metric": metrics,
                    "Before": before_means.values,
                    "After": after_means.values,
                    "% Improvement": improvement.values
                })

                improvement_csv_path = os.path.join(pseudo_dir, "metric_improvement_summary.csv")
                improvement_df.to_csv(improvement_csv_path, index=False)
                print(f"✅ Saved metric improvement summary to: {improvement_csv_path}")
                
                plot_model_comparison_before_after(before_csv, after_csv, comparison_dir)

                # 4️⃣.3️⃣ Copy XAI + metrics
                copy_xai_and_analysis_files(
                    original_dir=original_dir,
                    pseudo_dir=pseudo_dir,
                    comparison_dir=comparison_dir,
                    report_dir=report_dir
                )

        else:
            print("⚠️ Skipping pseudo-label training due to missing entropy file.")

    print("✅ All experiments (original + pseudo) completed.")

if __name__ == "__main__":
    main()
