# ==============================================
# 📊 Training Log Visualization
# ==============================================
import pandas as pd
import matplotlib.pyplot as plt
from utils import RESULTS_DIR, PLOTS_DIR

GRAPHS_DIR = PLOTS_DIR / "gnn_plots"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load the saved training log ---
df_logs = pd.read_csv(RESULTS_DIR/"training_log.csv")

# --- Compute train-test accuracy gap ---
df_logs["Acc_Gap"] = df_logs["Train_Acc"] - df_logs["Test_Acc"]

# --- Average across folds per epoch ---
df_summary = df_logs.groupby("Epoch")[["Train_Loss","Test_Loss", "Train_Acc", "Test_Acc", "Acc_Gap"]].mean().reset_index()

# ==============================================
# 🔹 Plot 1: Train vs Test Accuracy
# ==============================================
plt.figure(figsize=(10, 6))
plt.plot(df_summary["Epoch"], df_summary["Train_Acc"], label="Train Accuracy", linewidth=2)
plt.plot(df_summary["Epoch"], df_summary["Test_Acc"], label="Test Accuracy", linewidth=2, color="orange")
plt.fill_between(df_summary["Epoch"], df_summary["Test_Acc"], df_summary["Train_Acc"],
                 color='lightcoral', alpha=0.3, label="Accuracy Gap")
plt.title("Train vs Test Accuracy per Epoch", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "train_test_acc.png", dpi=300)
plt.close()

# ==============================================
# 🔹 Plot 2: Accuracy Gap
# ==============================================
plt.figure(figsize=(10, 6))
plt.plot(df_summary["Epoch"], df_summary["Acc_Gap"], color='red', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Train-Test Accuracy Difference (Gap)", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Train - Test Accuracy")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "accuracy_gap.png", dpi=300)
plt.close()

# ==============================================
# 🔹 Plot 3: Loss Curve
# ==============================================
plt.figure(figsize=(10,6))
plt.plot(df_summary["Epoch"], df_summary["Train_Loss"], label="Train Loss", linewidth=2)
plt.plot(df_summary["Epoch"], df_summary["Test_Loss"], label="Test Loss", linewidth=2, color="orange")
plt.fill_between(df_summary["Epoch"], df_summary["Test_Loss"], df_summary["Train_Loss"],
                 color="lightcoral", alpha=0.3, label="Loss Gap")
plt.title("Train vs Test Loss per Epoch", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(GRAPHS_DIR / "train_test_loss.png", dpi=300)
plt.close()

print("✅ Visualization complete. All comparison plots generated successfully.")
