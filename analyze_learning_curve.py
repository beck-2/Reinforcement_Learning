"""
Analyze and plot learning curves for the successor representation agent.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d

def plot_learning_curve(csv_path, output_path=None, show_plot=True):
    """
    Load training log and plot learning curve.
    
    Args:
        csv_path: Path to training_log.csv
        output_path: Where to save the plot (default: same dir as input, .png)
        show_plot: Whether to display the plot
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Successor Representation Agent - Learning Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean Return vs Steps (with smoothing)
    ax = axes[0, 0]
    window = min(20, len(df) // 10)  # Adaptive window size
    smoothed = uniform_filter1d(df['mean_return'].values, size=window, mode='nearest')
    ax.plot(df['steps'], df['mean_return'], linewidth=1, color='#2E86AB', alpha=0.3, label='Raw')
    ax.plot(df['steps'], smoothed, linewidth=2.5, color='#E63946', alpha=0.9, label='Smoothed')
    ax.fill_between(df['steps'], smoothed, alpha=0.15, color='#E63946')
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Mean Return', fontsize=11)
    ax.set_title('Reward vs Training Steps', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    ax.ticklabel_format(style='plain', axis='x')
    
    # Plot 2: Mean Return vs Rollout (with smoothing)
    ax = axes[0, 1]
    smoothed_rollout = uniform_filter1d(df['mean_return'].values, size=window, mode='nearest')
    ax.plot(df['rollout'], df['mean_return'], linewidth=1, color='#A23B72', alpha=0.3, label='Raw')
    ax.plot(df['rollout'], smoothed_rollout, linewidth=2.5, color='#F18F01', alpha=0.9, label='Smoothed')
    ax.fill_between(df['rollout'], smoothed_rollout, alpha=0.15, color='#F18F01')
    ax.set_xlabel('Rollout Number', fontsize=11)
    ax.set_ylabel('Mean Return', fontsize=11)
    ax.set_title('Reward vs Rollout Number', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    # Plot 3: Loss Components (separate scales)
    ax = axes[1, 0]
    ax.plot(df['steps'], df['value_loss'], label='Value Loss', linewidth=2, alpha=0.8, color='#E63946')
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Value Loss', fontsize=11, color='#E63946')
    ax.set_title('Loss Components vs Steps', fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#E63946')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='x')
    
    # Create secondary y-axis for policy and SR loss
    ax2 = ax.twinx()
    ax2.plot(df['steps'], df['policy_loss'], label='Policy Loss', linewidth=1.5, alpha=0.7, color='#457B9D', linestyle='--')
    ax2.plot(df['steps'], df['sr_loss'], label='SR Loss', linewidth=1.5, alpha=0.7, color='#F1FAEE', linestyle=':')
    ax2.set_ylabel('Policy / SR Loss', fontsize=11)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    
    # Plot 4: Mean Episode Length vs Steps
    ax = axes[1, 1]
    ax.plot(df['steps'], df['mean_episode_length'], linewidth=2, color='#F18F01', alpha=0.8)
    ax.fill_between(df['steps'], df['mean_episode_length'], alpha=0.2, color='#F18F01')
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Mean Episode Length', fontsize=11)
    ax.set_title('Episode Length vs Steps', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        csv_dir = Path(csv_path).parent
        output_path = csv_dir / "learning_curve.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LEARNING CURVE SUMMARY")
    print("="*60)
    print(f"Total training steps:      {df['steps'].iloc[-1]:,}")
    print(f"Total rollouts:            {df['rollout'].iloc[-1]:,}")
    print(f"\nFinal mean return:         {df['mean_return'].iloc[-1]:.2f}")
    print(f"Best mean return:          {df['mean_return'].max():.2f}")
    print(f"Initial mean return:       {df['mean_return'].iloc[0]:.2f}")
    print(f"\nFinal mean episode length: {df['mean_episode_length'].iloc[-1]:.0f}")
    print(f"Initial mean episode length: {df['mean_episode_length'].iloc[0]:.0f}")
    print("\nFirst 5 steps:")
    print(df[['steps', 'rollout', 'mean_return', 'mean_episode_length']].head(5).to_string(index=False))
    print("\nLast 5 steps:")
    print(df[['steps', 'rollout', 'mean_return', 'mean_episode_length']].tail(5).to_string(index=False))
    print("="*60)
    
    if show_plot:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from training log")
    parser.add_argument(
        "--log",
        type=str,
        default="results/ssr_rnn/training_log.csv",
        help="Path to training_log.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: same dir as log, named learning_curve.png)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plot interactively"
    )
    
    args = parser.parse_args()
    plot_learning_curve(args.log, args.output, not args.no_show)
