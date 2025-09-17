import matplotlib.pyplot as plt
import os

def plot_training_curves(metrics_dict, output_path, title="Training Curves"):
    """
    Plot training loss and accuracy curves over communication rounds.
    
    Args:
      metrics_dict: Dictionary with keys 'loss' and 'accuracy', each a list over rounds.
      output_path: File path to save the generated plot.
      title: Title of the plot.
    """
    rounds = list(range(1, len(metrics_dict['loss']) + 1))
    fig, ax1 = plt.subplots()
    color_loss = 'tab:red'
    color_acc = 'tab:blue'
    
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.plot(rounds, metrics_dict['loss'], color=color_loss, label="Loss")
    ax1.tick_params(axis='y', labelcolor=color_loss)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color=color_acc)
    ax2.plot(rounds, metrics_dict['accuracy'], color=color_acc, label="Accuracy")
    ax2.tick_params(axis='y', labelcolor=color_acc)
    
    plt.title(title)
    fig.tight_layout()
    plt.legend()
    plt.savefig(output_path)
    plt.close(fig)

def plot_group_accuracies(group_accuracy_dict, output_path, title="Group Accuracies"):
    """
    Plot a bar chart for per-group accuracies.
    
    Args:
      group_accuracy_dict: Dictionary mapping group keys to accuracy.
      output_path: File path to save the bar chart.
      title: Title of the plot.
    """
    groups = list(group_accuracy_dict.keys())
    accuracies = [group_accuracy_dict[g] for g in groups]
    
    plt.figure()
    plt.bar(groups, accuracies)
    plt.xlabel("Group")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1)
    plt.savefig(output_path)
    plt.close()
