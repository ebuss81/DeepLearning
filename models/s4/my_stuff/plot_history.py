import matplotlib.pyplot as plt
import torch
import matplotlib
matplotlib.use('TkAgg')
def plot_history(history, savepath=None):
    epochs = [h["epoch"] for h in history]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axs[0,0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axs[0,0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axs[0,0].plot(epochs, [h["test_loss"] for h in history], label="test")
    axs[0,0].set_title("Loss")
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Cross-Entropy Loss")
    axs[0,0].legend()

    # Accuracy
    axs[0,1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axs[0,1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axs[0,1].plot(epochs, [h["test_acc"] for h in history], label="test")
    axs[0,1].set_title("Accuracy")
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("Accuracy (%)")
    axs[0,1].legend()

    # Learning rate
    axs[1,0].plot(epochs, [h["lr"] for h in history], label="lr", color="tab:orange")
    axs[1,0].set_title("Learning Rate")
    axs[1,0].set_xlabel("Epoch")
    axs[1,0].set_ylabel("LR")

    # Empty or custom plot (e.g. train vs val gap)
    gap = [h["train_acc"] - h["val_acc"] for h in history]
    axs[1,1].plot(epochs, gap, label="train - val acc gap", color="tab:red")
    axs[1,1].set_title("Generalization Gap")
    axs[1,1].set_xlabel("Epoch")
    axs[1,1].set_ylabel("Δ Accuracy (%)")
    axs[1,1].legend()

    fig.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

# Example usage:
#history = torch.load("/home/wp/Documents/GitHub/DataProcessing/Classifier/s4/my_stuff/history.pt")
history = torch.load("/home/wp/Documents/GitHub/DataProcessing/DeepLearning/my_stuff/history.pt")
plot_history(history, savepath="history_plots.png")
