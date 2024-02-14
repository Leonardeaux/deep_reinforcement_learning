import matplotlib.pyplot as plt
import numpy as np


def create_plots(episode_numbers, all_win_rates, all_loss_rates):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(episode_numbers[:-1], all_win_rates[:-1], c='green')
    plt.xlabel("Episode number")
    plt.ylabel("Win Rate")
    plt.title("Test: Winning Percentage Over Time")

    plt.subplot(1, 2, 2)
    plt.plot(episode_numbers[:-1], all_loss_rates[:-1], c='red')
    plt.xlabel("Episode number")
    plt.ylabel("Loss Rate")
    plt.title("Test: Losing Percentage Over Time")

    plt.show()


def create_score_plot(label: str, all_scores, moving_average):
    plt.figure(figsize=(16, 6))
    plt.plot(all_scores, c='darkblue', alpha=0.3)
    plt.plot(moving_average, c='darkblue')
    plt.xlabel("Episode number")
    plt.ylabel("Score")
    plt.title(label + ": Scores per Episode")

    plt.show()

