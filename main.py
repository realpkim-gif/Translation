import json

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

with open("transformer1_history.json") as f:
    history_1 = json.load(f)

with open("transformer2_history.json") as f:
    history_2 = json.load(f)

with open("transformer3_history.json") as f:
    history_3 = json.load(f)

epochs_1 = range(1, len(history_1["train_loss"]) + 1)
epochs_2 = range(1, len(history_2["train_loss"]) + 1)
epochs_3 = range(1, len(history_3["train_loss"]) + 1)


fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

ax_loss.plot(epochs_1, history_1["train_loss"], marker="o", label="Transformer I (full data)")
ax_loss.plot(epochs_2, history_2["train_loss"], marker="o", label="Transformer II (85/10/5 split)")
ax_loss.plot(epochs_3, history_3["train_loss"], marker="o", label="Transformer III (85/10/5 split) 100 epoch")
ax_loss.set_title("Train Loss per Epoch")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.legend()

ax_acc.plot(epochs_1, history_1["train_accuracy"], marker="o", label="Transformer I (full data)")
ax_acc.plot(epochs_2, history_2["train_accuracy"], marker="o", label="Transformer II (85/10/5 split)")
ax_acc.plot(epochs_3, history_3["train_accuracy"], marker="o", label="Transformer III (85/10/5 split) 100 epoch")
ax_acc.set_title("Train Accuracy per Epoch")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.legend()


fig.suptitle("Transformer I vs Transformer II vs Transformer III — Training Performance")
fig.tight_layout()
fig.savefig("model_comparison.png", dpi=150)
plt.show()


def epochs_run(history):
    stopped = history.get("stopped_epoch", 0)
    return f"{stopped} (early-stopped)" if stopped else f"{len(history['train_loss'])}"


def pct(value):
    return "—" if value is None else f"{value * 100:.2f}%"


rows = ["Epochs run", "Train accuracy", "Val accuracy", "Test accuracy"]
columns = ["Transformer I", "Transformer II", "Transformer III"]
cell_data = [
    ["10 (fixed)", epochs_run(history_2), epochs_run(history_3)],
    [pct(history_1["train_accuracy"][-1]), pct(history_2["train_accuracy"][-1]), pct(history_3["train_accuracy"][-1])],
    [pct(None), pct(history_2["val_accuracy"][-1]), pct(history_3["val_accuracy"][-1])],
    [pct(history_1["test-accuracy"]), pct(history_2["test_accuracy"]), pct(history_3["test_accuracy"])],
]

fig_table, ax_table = plt.subplots(figsize=(9, 2.5))
ax_table.axis("off")
table = ax_table.table(
    cellText=cell_data,
    rowLabels=rows,
    colLabels=columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
for (row, col), cell in table.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#eeeeee")

fig_table.suptitle("Transformer I vs II vs III — Summary")
fig_table.tight_layout()
fig_table.savefig("model_comparison_table.png", dpi=150, bbox_inches="tight")
plt.show()
