import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

# ========================== Atmospheric Constants ===========================
density_at_surface = 1.225e-3
scale_height = 6.5e5
SIMULATION_GROUND_DEPTH = density_at_surface * scale_height  # Should be ~796.25 g/cm^2


# =========================== Plotting Functions ===========================

def plot_avg_longitudinal_profile(df, num_showers, filename):
    """Plots the average longitudinal profile with a clearly marked Xmax."""
    print("  - Analyzing longitudinal profile...")
    max_depth = df['atmospheric_depth'].max()
    bins = np.linspace(0, max_depth * 1.05, 60)  
    bin_centers = (bins[:-1] + bins[1:]) / 2

    em_df = df[df['type'].isin(['electron', 'positron', 'photon'])]
    mu_df = df[df['type'].isin(['muon_plus', 'muon_minus'])]

    total_counts, _ = np.histogram(df['atmospheric_depth'], bins=bins)
    em_counts, _ = np.histogram(em_df['atmospheric_depth'], bins=bins)
    mu_counts, _ = np.histogram(mu_df['atmospheric_depth'], bins=bins)

    avg_em_counts = em_counts / num_showers
    avg_mu_counts = mu_counts / num_showers

    # Find and comment the Shower Maximum on the graph (Xmax)
    # We find the peak of the EM component as it defines the shower max
    if len(avg_em_counts) > 0 and np.max(avg_em_counts) > 0:
        xmax_index = np.argmax(avg_em_counts)
        xmax_val = bin_centers[xmax_index]
        max_particles = avg_em_counts[xmax_index]
    else:
        xmax_val, max_particles = None, None

    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    #  plotting histogram
    plt.step(bin_centers, avg_em_counts + avg_mu_counts, where='mid', color='black', label=f'All Particles',
             linewidth=2.5)
    plt.step(bin_centers, avg_em_counts, where='mid', color='blue', linestyle='--', label='EM Component (e±, γ)')
    plt.step(bin_centers, avg_mu_counts, where='mid', color='red', linestyle=':', label='Muonic Component (μ±)',
             linewidth=2)

    plt.axvline(SIMULATION_GROUND_DEPTH, color='gray', linestyle='-.',
                label=f'Sim Ground Depth ({SIMULATION_GROUND_DEPTH:.1f} g/cm²)')

    # --- Add annotation for Xmax ---
    if xmax_val is not None:
        plt.axvline(xmax_val, color='green', linestyle=':', linewidth=2, label=f'Peak (Xmax) = {xmax_val:.1f} g/cm²')
        plt.annotate('Shower Maximum',
                     xy=(xmax_val, max_particles),
                     xytext=(xmax_val + 50, max_particles * 0.8),  # Position the text
                     arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="ivory", ec="gray", lw=1, alpha=0.8))

    plt.yscale('log')
    plt.xlabel('Atmospheric Depth (g/cm²)', fontsize=14)
    plt.ylabel('Average Number of Particles per Shower per Bin', fontsize=14)
    plt.title(f'Average Longitudinal Shower Profile ({num_showers} Showers)', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlim(left=0)
    plt.ylim(bottom=1e-2)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(filename, dpi=150)  
    plt.close()
    print(f"  - Saved {filename}")


def plot_xmax_distribution(df, filename):
    """Calculates and plots a high-quality Xmax distribution."""
    print("  - Analyzing Xmax distribution...")
    xmax_values = []
    for shower_id, shower_df in df.groupby('shower_id'):
        em_charged_df = shower_df[shower_df['type'].isin(['electron', 'positron'])]
        if not em_charged_df.empty:
            counts, bin_edges = np.histogram(em_charged_df['atmospheric_depth'], bins=50)  
            if np.max(counts) > 0:
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                xmax = bin_centers[np.argmax(counts)]
                if xmax < SIMULATION_GROUND_DEPTH * 0.98:
                    xmax_values.append(xmax)

    if not xmax_values:
        print("  - Warning: No valid Xmax values found.");
        return

    mean_xmax = np.mean(xmax_values)
    median_xmax = np.median(xmax_values)
    std_xmax = np.std(xmax_values)

    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.hist(xmax_values, bins=35, color='teal', alpha=0.75, edgecolor='black', label=f'{len(xmax_values)} Showers')
    plt.axvline(mean_xmax, color='red', linestyle='--', linewidth=2, label=f'Mean Xmax = {mean_xmax:.1f} g/cm²')
    plt.axvline(median_xmax, color='darkviolet', linestyle=':', linewidth=2,
                label=f'Median Xmax = {median_xmax:.1f} g/cm²')

    stats_text = f'Std Dev = {std_xmax:.1f} g/cm²\nShowers Analyzed = {len(xmax_values)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.4))

    plt.xlabel('Xmax (Atmospheric Depth, g/cm²)', fontsize=14)
    plt.ylabel('Number of Showers', fontsize=14)
    plt.title(f'Distribution of Shower Maximum (Xmax) for {len(xmax_values)} Showers', fontsize=16)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  - Saved {filename}")


def plot_ground_particle_distribution(df, num_showers, filename):
    """Plots a clearer histogram of the number of particles reaching ground level."""
    print("  - Analyzing ground particle distribution...")
    ground_df = df[df['atmospheric_depth'] >= SIMULATION_GROUND_DEPTH * 0.99]
    ground_counts = ground_df.groupby('shower_id').size()

    # Fill in zeros for showers that had no ground particles
    all_shower_counts = ground_counts.reindex(range(1, num_showers + 1), fill_value=0)

    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    if all_shower_counts.sum() == 0:
        plt.text(0.5, 0.5, 'No particles reached ground level in any shower.', horizontalalignment='center',
                 verticalalignment='center', fontsize=12)
    else:
        max_count = all_shower_counts.max()
        bins = np.arange(-0.5, max_count + 1.5, 1)
        plt.hist(all_shower_counts, bins=bins, color='darkorange', alpha=0.8, edgecolor='black', rwidth=0.8)

        mean_particles = all_shower_counts.mean()
        plt.axvline(mean_particles, color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {mean_particles:.2f} particles/shower')

        showers_with_hits = (all_shower_counts > 0).sum()
        stats_text = (f'Showers with ≥1 particle: {showers_with_hits} ({showers_with_hits / num_showers:.1%})\n'
                      f'Max particles in one shower: {max_count}')
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.4))

        plt.legend(fontsize=12)

    plt.xlabel('Number of Particles at Ground per Shower', fontsize=14)
    plt.ylabel('Number of Showers', fontsize=14)
    plt.title(f'Distribution of Particle Count at Ground Level (≥ {SIMULATION_GROUND_DEPTH:.1f} g/cm²)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  - Saved {filename}")


def plot_ground_energy_spectrum(df, filename):
   
    print("  - Analyzing ground energy spectrum...")
    ground_df = df[df['atmospheric_depth'] >= SIMULATION_GROUND_DEPTH * 0.99].copy()

    em_ground = ground_df[ground_df['type'].isin(['electron', 'positron', 'photon'])]
    mu_ground = ground_df[ground_df['type'].isin(['muon_plus', 'muon_minus'])]

    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    if not ground_df.empty:
        bins = np.logspace(np.log10(max(1e-4, ground_df['total_energy'].min())),
                           np.log10(max(1.0, ground_df['total_energy'].max())), 50)

        # Calculate mean energies for legend in the graph
        mean_em_energy = em_ground['total_energy'].mean() if not em_ground.empty else 0
        mean_mu_energy = mu_ground['total_energy'].mean() if not mu_ground.empty else 0

        plt.hist(em_ground['total_energy'], bins=bins, alpha=0.6,
                 label=f'EM Component ({len(em_ground)} particles, Mean={mean_em_energy:.3f} GeV)', color='blue')
        plt.hist(mu_ground['total_energy'], bins=bins, alpha=0.7,
                 label=f'Muonic Component ({len(mu_ground)} particles, Mean={mean_mu_energy:.3f} GeV)', color='red')

        if len(em_ground) > 0: plt.axvline(mean_em_energy, color='blue', linestyle=':', linewidth=2)
        if len(mu_ground) > 0: plt.axvline(mean_mu_energy, color='red', linestyle=':', linewidth=2)

        plt.legend(fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No particles reached ground level.', horizontalalignment='center',
                 verticalalignment='center', fontsize=12)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Particle Energy at Ground (GeV)', fontsize=14)
    plt.ylabel('Number of Particles', fontsize=14)
    plt.title(f'Energy Spectrum of Particles at Ground Level (≥ {SIMULATION_GROUND_DEPTH:.1f} g/cm²)', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  - Saved {filename}")


def plot_muon_em_ratio(df, num_showers, filename):
    """Plots the ratio of muonic to electromagnetic particles vs. depth."""
    print("  - Analyzing muon-to-EM ratio...")
    max_depth = df['atmospheric_depth'].max()
    bins = np.linspace(0, max_depth * 1.05, 40)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    em_df = df[df['type'].isin(['electron', 'positron', 'photon'])]
    mu_df = df[df['type'].isin(['muon_plus', 'muon_minus'])]

    em_counts, _ = np.histogram(em_df['atmospheric_depth'], bins=bins)
    mu_counts, _ = np.histogram(mu_df['atmospheric_depth'], bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(em_counts > 0, mu_counts / em_counts, np.nan)

    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.plot(bin_centers, ratio, marker='o', linestyle='-', color='purple', label='Ratio (μ± / EM)')
    plt.axvline(SIMULATION_GROUND_DEPTH, color='gray', linestyle='-.',
                label=f'Sim Ground Depth ({SIMULATION_GROUND_DEPTH:.1f} g/cm²)')

    #  Adding annotation where there is a spike 

    valid_ratio_points = ratio[bin_centers < SIMULATION_GROUND_DEPTH * 0.98]
    if len(valid_ratio_points) > 0:
        last_valid_idx = len(valid_ratio_points) - 1
        last_bin_center = bin_centers[last_valid_idx]

        # point the spike the spike
        spike_idx = np.nanargmax(ratio)
        if spike_idx > last_valid_idx:
            plt.annotate('Ratio spikes as\nEM component vanishes',
                         xy=(bin_centers[spike_idx], ratio[spike_idx]),
                         xytext=(bin_centers[spike_idx] - 300, ratio[spike_idx] * 0.5),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="ivory", lw=1, alpha=0.8))

    plt.yscale('log')
    plt.xlabel('Atmospheric Depth (g/cm²)', fontsize=14)
    plt.ylabel('Ratio (Number of Muons / Number of EM Particles)', fontsize=14)
    plt.title(f'Muon-to-Electromagnetic Ratio vs. Depth ({num_showers} Showers)', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  - Saved {filename}")


# =========================== Main analysis function ===========================

def analyze_shower_data(filename):
    """Reads a shower data CSV and generates all plots and summaries."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return

    if df.empty:
        print("The data file is empty. No analysis to perform.");
        return

    num_showers = df['shower_id'].nunique()
    print("\n--- Overall Summary ---")
    print(f"Analyzed {num_showers} showers from '{filename}'")
    print(f"Total particles from all showers: {len(df)}")
    print(f"Average particles per shower: {len(df) / num_showers:.1f}")
    print("\nTotal Particle Counts (All Showers):")
    counts = df['type'].value_counts().sort_index()
    for ptype, count in counts.items(): print(f"  {ptype}: {count}")

    print("\nGenerating plots...")
    plot_avg_longitudinal_profile(df, num_showers, 'avg_longitudinal_profile.png')
    plot_xmax_distribution(df, 'xmax_distribution.png')
    plot_ground_particle_distribution(df, num_showers, 'ground_particle_distribution.png')
    plot_ground_energy_spectrum(df, 'ground_energy_spectrum.png')
    plot_muon_em_ratio(df, num_showers, 'muon_em_ratio.png')

    print("\nAnalysis complete. Plots have been saved as high-quality PNG files.")


# =========================== Exececuting the programn ===========================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
        analyze_shower_data(csv_filename)
    else:
        print("Usage: python analyze_shower_data.py <filename.csv>")
        print("Example: python analyze_shower_data.py shower_data_100GeV_1000runs.csv")



