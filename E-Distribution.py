import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_interaction_energy_histograms(raw_file, hp_file):
    # 1. Load the Data
    # Assuming whitespace delimiter for .dat and .txt
    df_raw = pd.read_csv(raw_file, sep='\s+', names=['Time', 'Energy'], engine='python')
    df_hp = pd.read_csv(hp_file, sep='\s+', comment='#', names=['Time', 'HP_Energy'], engine='python')

    # 2. Synchronize Time Windows
    # Get the time range of the HP data (which is already trimmed 5% at both ends)
    hp_start = df_hp['Time'].min()
    hp_end = df_hp['Time'].max()
    
    # Trim the Raw data to match the HP start/end times exactly
    # This ensures we are comparing the exact same set of snapshots
    df_raw_trimmed = df_raw[(df_raw['Time'] >= hp_start) & (df_raw['Time'] <= hp_end)]
    
    print(f"HP Data Range: {hp_start:.2f} to {hp_end:.2f} ns")
    print(f"Raw Data Trimmed to: {df_raw_trimmed['Time'].min():.2f} to {df_raw_trimmed['Time'].max():.2f} ns")

    # 3. Setup Publication-Quality Figure
    # Using a clean style and professional fonts
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    # --- Panel A: Raw Interaction Energy ---
    # Using a density plot (stat='density') for comparison
    sns.histplot(
        data=df_raw_trimmed, x='Energy', 
        kde=True, ax=axes[0], 
        color='#004c6d',  # Professional dark blue
        stat='density', bins='auto', alpha=0.6, edgecolor=None
    )
    axes[0].set_title('Raw Interaction Energy\n(Multi-modal / Skewed)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Interaction Energy (kcal/mol)', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    
    # Add text label for the shoulder/skew if visible (optional customization)
    # axes[0].text(-55, 0.05, 'Shoulder\n(Sub-state)', ha='center', fontsize=10, color='red')

    # --- Panel B: High-Pass Component ---
    # Plot histogram
    sns.histplot(
        data=df_hp, x='HP_Energy', 
        kde=False, ax=axes[1], 
        color='#a72608',  # Professional dark red
        stat='density', bins='auto', alpha=0.6, edgecolor=None,
        label='HP Distribution'
    )
    
    # Overlay Gaussian Fit to prove normality
    mu, std = norm.fit(df_hp['HP_Energy'])
    xmin, xmax = axes[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    axes[1].plot(x, p, 'k--', linewidth=2, label=f'Gaussian Fit\n$\mu={mu:.2f}, \sigma={std:.2f}$')
    axes[1].set_title('High-Frequency Component\n(Gaussian / Entropic)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Energy Fluctuation (kcal/mol)', fontsize=12)
    axes[1].set_ylabel('') # Redundant label
    
    # Clean up legend
    axes[1].legend(loc='upper right', frameon=False, fontsize=10)

    # 4. Save and Show
    plt.savefig('Figure4_Histograms.png', dpi=600, bbox_inches='tight')
    plt.show()

# === Run from command line ===
if len(sys.argv) != 2:
    print("Usage: python E-Distribution.py filename.dat")
    sys.exit(1)

raw_file = sys.argv[1]

# Automatically assume the HP file name
hp_file = "data_highfreq_autocorr.txt"

plot_interaction_energy_histograms(raw_file, hp_file)