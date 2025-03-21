import os
import json
import polars as pl
import matplotlib.pyplot as plt
import logging
import sys
from scipy.stats import ttest_ind
import argparse
import numpy as np
import matplotlib
import seaborn as sns
from scipy import stats

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["font.family"] = "Iosevka"
matplotlib.rcParams["font.size"] = 14

handler = logging.StreamHandler(sys.stdout)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def custom_sort_key(folder_name):
    order = [
        "baseline",
        "thp",
        "mimalloc",
        "mimalloc-thp",
        "jemalloc",
        "jemalloc-thp",
        "tcmalloc",
        "tcmalloc-thp",
    ]
    folder_name = os.path.basename(folder_name).lower()
    
    # Exact match
    if folder_name in order:
        return order.index(folder_name)
    
    # Check for compound names
    for i, name in enumerate(order):
        if name.endswith('-thp') and folder_name == name:
            return i
    
    # Check for partial matches, ensuring 'thp' doesn't match others
    for i, name in enumerate(order):
        if name in folder_name and (name != 'thp' or folder_name == 'thp'):
            return i
    
    return len(order)  # Put any unmatched folders at the end



def read_json_data(folder_path):
    """Reads execution time data from all JSON files in a folder."""
    execution_times = []
    for file in os.listdir(folder_path):
        if file.startswith("run_") and file.endswith(".json"):
            with open(os.path.join(folder_path, file), "r") as f:
                data = json.load(f)
                execution_times.append(data["total_time_s"])
    return execution_times


def print_baseline_stats(baseline_execution_time):
    """Prints statistics for the baseline execution time data."""
    baseline_mean = pl.Series(baseline_execution_time).mean()
    baseline_std = pl.Series(baseline_execution_time).std()
    baseline_min = min(baseline_execution_time)
    baseline_max = max(baseline_execution_time)

    logging.info("Baseline Statistics:")
    logging.info(f"Mean Execution Time: {baseline_mean:.15f}")
    logging.info(f"Standard Deviation: {baseline_std:.15f}")
    logging.info(f"Minimum Execution Time: {baseline_min:.15f}")
    logging.info(f"Maximum Execution Time: {baseline_max:.15f}")
    print("=" * 100)


def compare_execution_times(
    baseline_execution_time, optimised_execution_time, sample_size, random_state
):
    """Compares execution times between baseline and optimised datasets."""
    # Sample the specified number of executions from both datasets
    baseline_sample = (
        pl.Series(baseline_execution_time)
        .sample(n=sample_size, seed=random_state)
        .to_numpy()
    )
    optimised_sample = (
        pl.Series(optimised_execution_time)
        .sample(n=sample_size, seed=random_state)
        .to_numpy()
    )

    # Perform an independent t-test
    t_stat, p_value = ttest_ind(baseline_sample, optimised_sample)

    # Calculate means
    baseline_mean = baseline_sample.mean()
    baseline_std = baseline_sample.std()
    optimised_mean = optimised_sample.mean()
    optimised_std = optimised_sample.std()
    optimised_min = min(optimised_sample)
    optimised_max = max(optimised_sample)
    percentage_diff = calculate_percentage_difference(baseline_mean, optimised_mean)

    # Calculate Coefficient of Variation (CoV)
    baseline_cov = (baseline_std / baseline_mean) * 100
    optimised_cov = (optimised_std / optimised_mean) * 100

    logging.info(f"Baseline Mean Execution Time (Sample): {baseline_mean:.15f}")
    logging.info(f"Baseline Coefficient of Variation: {baseline_cov:.2f}%")
    logging.info(f"Optimised Mean Execution Time (Sample): {optimised_mean:.15f}")
    logging.info(f"Optimised Standard Deviation: {optimised_std:.15f}")
    logging.info(f"Optimised Coefficient of Variation: {optimised_cov:.2f}%")
    logging.info(f"T-statistic: {t_stat:.15f}")
    logging.info(f"P-value: {p_value:.15f}")
    logging.info(f"Minimum Execution Time: {optimised_min:.15f}")
    logging.info(f"Maximum Execution Time: {optimised_max:.15f}")
    logging.info(
        f"Percentage difference (negative means faster): {percentage_diff:.15f}%"
    )

    # Check if the p-value is below 0.05 for significance
    if p_value < 0.05:
        if optimised_mean < baseline_mean:
            logging.info(
                "The optimised dataset has a \033[92m\033[1mstatistically significant\033[0m improvement (lower execution time)"
            )
        else:
            logging.info(
                "The optimised dataset has a statistically \033[91m\033[1msignificant negative impact\033[0m (higher execution time)"
            )
    else:
        logging.info(
            "The difference in execution times is \033[91m\033[1mnot statistically significant\033[0m"
        )

    return baseline_sample, optimised_sample, optimised_mean


def plot_boxplots(baseline_execution_time, optimised_files_data, output_folder):
    """Creates box plots comparing baseline and optimised execution times."""
    plt.figure(figsize=(12, 8))

    # Prepare data for box plots
    data = [baseline_execution_time]
    tick_labels = ["baseline"]

    # Add optimised data
    for execution_time, folder_path in optimised_files_data:
        data.append(execution_time)
        tick_labels.append(os.path.basename(folder_path))

    # Create box plot
    plt.boxplot(data, tick_labels=tick_labels)

    # Customize the plot
    plt.title("Execution Times Distribution Comparison", weight="bold")
    plt.ylabel("Execution Time")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, "execution_times_boxplot.svg")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Box plot saved to {output_path}")


def plot_combined_execution_times(
    baseline_sample, optimised_samples, optimised_folders, output_folder
):
    """Plots combined execution times from baseline and optimised samples."""
    plt.figure(figsize=(14, 8))
    run_numbers = range(1, len(baseline_sample) + 1)

    # Plot baseline sample
    plt.scatter(run_numbers, baseline_sample, color="blue", alpha=0.7, label="baseline")

    # Add mean line for baseline
    baseline_mean = baseline_sample.mean()
    plt.axhline(baseline_mean, color="blue", linestyle="--", label="baseline Mean")

    # Plot each set of optimised samples
    colors = plt.cm.viridis(np.linspace(0, 1, len(optimised_samples)))
    for idx, (optimised_sample, optimised_folder) in enumerate(
        zip(optimised_samples, optimised_folders)
    ):
        plt.scatter(
            run_numbers,
            optimised_sample,
            color=colors[idx],
            alpha=0.7,
            label=f"{os.path.basename(optimised_folder)}",
        )
        # Add mean line for each optimised sample
        optimised_mean = optimised_sample.mean()
        plt.axhline(
            optimised_mean,
            color=colors[idx],
            linestyle="--",
            label=f"{os.path.basename(optimised_folder)} Mean",
        )

    # Add titles and labels
    plt.title("Execution Times Comparison with Mean", weight="bold")
    plt.xlabel("Run Number")
    plt.ylabel("Execution Time (Lower = Better)")
    plt.grid(True)
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "scatter_plot.svg")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Combined plot with mean lines saved to {output_path}")


def find_benchmark_folders(root_folder):
    """Finds all benchmark folders in the given folder structure."""
    benchmark_folders = []
    for item in os.listdir(root_folder):
        if item == "stats" or item == "splitlock":
            continue
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path) and any(
            file.startswith("run_") and file.endswith(".json")
            for file in os.listdir(item_path)
        ):
            benchmark_folders.append(item_path)
    return benchmark_folders


def calculate_percentage_difference(baseline_mean, best_mean):
    percentage_diff = ((best_mean - baseline_mean) / baseline_mean) * 100
    return percentage_diff


def plot_percentage_improvement(
    baseline_mean, optimized_means, folders, output_folder, baseline_folder
):
    improvements = [
        (om - baseline_mean) / baseline_mean * 100 for om in optimized_means
    ]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(improvements)), improvements)
    plt.title("Percentage Improvement Over Baseline", weight="bold")
    plt.ylabel("Percentage Improvement (Lower = Better)")
    plt.xlabel("Optimisation")
    plt.xticks(
        range(len(improvements)),
        [os.path.basename(f) for f in folders if f != baseline_folder],
        rotation=45,
        ha="right",
    )
    plt.axhline(y=0, color="r", linestyle="-")
    plt.tight_layout()
    output_path = os.path.join(output_folder, "percentage_improvement.svg")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Percentage improvement chart saved to {output_path}")

def create_pairwise_comparison_heatmap(benchmark_folders, output_folder):
    """
    Creates a heatmap of pairwise comparisons between all experiments,
    showing both percentage difference and p-value.
    """
    # Sort folders to ensure baseline is first
    benchmark_folders = sorted(benchmark_folders, key=custom_sort_key)

    # Calculate mean execution times and store all execution times for each experiment
    mean_times = []
    all_times = []
    labels = []

    for folder in benchmark_folders:
        execution_times = read_json_data(folder)
        mean_times.append(np.mean(execution_times))
        all_times.append(execution_times)
        labels.append(os.path.basename(folder))

    # Calculate percentage differences and p-values
    n = len(mean_times)
    diff_matrix = np.zeros((n, n), dtype=object)
    color_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):  # Only calculate for upper triangle
            perc_diff = (mean_times[j] - mean_times[i]) / mean_times[i] * 100
            _, p_value = stats.ttest_ind(all_times[i], all_times[j])
            
            # Store values for (i,j)
            diff_matrix[i, j] = f"{perc_diff:.2f}%\np={p_value:.3f}"
            color_matrix[i, j] = perc_diff
            
            # Mirror and flip sign for (j,i)
            diff_matrix[j, i] = f"{-perc_diff:.2f}%\np={p_value:.3f}"
            color_matrix[j, i] = -perc_diff

    # Set diagonal elements
    for i in range(n):
        diff_matrix[i, i] = "0%\np=1.000"
        color_matrix[i, i] = 0

    # Create heatmap
    plt.figure(figsize=(14, 12))

    # Create the heatmap with annotations disabled
    ax = sns.heatmap(
        color_matrix,
        annot=False,  # Ensure no built-in annotations
        fmt="",
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        center=0,
        cbar_kws={"label": "Percentage Difference"},
    )
    
    # Get the colormap and normalization function for brightness calculations
    cmap_used = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=color_matrix.min(), vmax=color_matrix.max())

    # Increase the vertical offset for better separation between lines
    offset = 0.15  # Adjust this value as needed
    
    for i in range(n):
        for j in range(n):
            # Split the annotation text into percentage difference and p-value parts
            perc_diff, p_value_str = diff_matrix[i, j].split("\n")
            p_value = float(p_value_str.split("=")[1])  # Extract numeric p-value
    
            # Get the cell's background color from the colormap for brightness calculation
            value = color_matrix[i, j]
            cell_color = cmap_used(norm(value))
            brightness = (cell_color[0] * 0.299 + cell_color[1] * 0.587 + cell_color[2] * 0.114)
            default_text_color = "white" if brightness < 0.5 else "black"
    
            # Determine the p-value text color: use dark red if p > 0.05, otherwise use default
            p_value_color = "darkred" if p_value > 0.05 else default_text_color
    
            # Place the p-value at the top of the cell (swapped position)

            if i != j:
                ax.text(
                    j + 0.5,
                    i + 0.5 + offset,  # upward offset for p-value
                    f"p={p_value:.3f}",
                    ha="center", va="center",
                    color=p_value_color,
                    fontsize=14,
                    weight="bold"
                )
                
                ax.text(
                    j + 0.5,
                    i + 0.5 - offset,  # downward offset for percentage difference
                    perc_diff,
                    ha="center", va="center",
                    color=default_text_color,
                    fontsize=14
                )

    plt.title("Pairwise Comparison: % Difference and p-value", weight="bold")
    plt.xlabel("Comparison Experiment")
    plt.ylabel("Base Experiment")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save the heatmap
    output_path = os.path.join(output_folder, "pairwise_comparison_heatmap.svg")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Pairwise comparison heatmap saved to {output_path}")

def is_statistically_significant(baseline_times, optimized_times, alpha=0.05):
    t_stat, p_value = ttest_ind(baseline_times, optimized_times)
    return p_value < alpha

def save_results_to_json(results, output_folder):
    """Saves the analysis results to a JSON file."""
    output_path = os.path.join(output_folder, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {output_path}")

def generate_typst_table(results, output_folder):
    """Generates a Typst table from the results dictionary."""
    
    # Find the best performing experiment based on percentage difference
    best_experiment = min(
        results["comparisons"],
        key=lambda x: x["percentage_difference"] if x["is_significant"] else float('inf')
    )
    
    # Start building the table string
    table_str = """#figure(
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      inset: 6pt,
      align: horizon,
      table.header(
        [*Experiment*],
        [*Mean ($mu$)*],
        [*$C_V$*],
        [*% Diff ($Delta$)*],
        [*Min*],
        [*Max*],
        [*P-value*],
      ),\n"""

    # Add baseline row
    baseline = results["baseline"]
    baseline_cov = (baseline["std"] / baseline["mean"]) * 100
    table_str += f"""
      [baseline], [{baseline["mean"]:.3f}], [{baseline_cov:.2f}], [-], [{baseline["min"]:.3f}], [{baseline["max"]:.3f}], [-],"""

    # Add comparison rows
    for comp in results["comparisons"]:
        # Calculate CoV
        cov = (comp["std"] / comp["mean"]) * 100
        
        # Determine if this is the best experiment
        is_best = (comp["folder"] == best_experiment["folder"])
        
        # Format p-value with red text if > 0.05
        p_value_str = f'[#text("{comp["p_value"]:.6f}", fill: red)]' if comp["p_value"] > 0.05 else f'[{comp["p_value"]:.6f}]'
        
        # Format the row, wrapping in bold if it's the best
        row_values = [
            comp["folder"],
            f"{comp['mean']:.3f}",
            f"{cov:.2f}",
            f"{'+' if comp['percentage_difference'] > 0 else ''}{comp['percentage_difference']:.2f}",
            f"{comp['min']:.3f}",
            f"{comp['max']:.3f}",
            p_value_str
        ]
        
        if is_best:
            row_values = [f"*{val}*" if not val.startswith('[#') else val for val in row_values]
        
        row_str = f"""
      [{row_values[0]}], [{row_values[1]}], [{row_values[2]}], [{row_values[3]}], [{row_values[4]}], [{row_values[5]}], {row_values[6]},"""
        
        table_str += row_str

    # Add closing tags
    table_str += """
    )
  ],
  caption: [Statistical analysis of Random Forest training time and different optimisation techniques],
)
"""

    # Write to file
    output_path = os.path.join(output_folder, "typst_table.txt")
    with open(output_path, "w") as f:
        f.write(table_str)
    logging.info(f"Typst table saved to {output_path}")

def main(root_folder, sample_size, random_state):
    output_folder = os.path.join(root_folder, "stats")
    
    # Initialize results dictionary
    results = {
        "baseline": {},
        "comparisons": [],
        "best_optimization": {},
        "metadata": {
            "sample_size": sample_size,
            "random_state": random_state
        }
    }

    # Find all benchmark folders
    benchmark_folders = sorted(find_benchmark_folders(root_folder), key=custom_sort_key)
    
    logging.info(benchmark_folders)

    # Identify baseline folder
    baseline_folder = next((f for f in benchmark_folders if "baseline" in f.lower()), None)
    if not baseline_folder:
        logging.info("Error: No baseline folder found.")
        return

    # Read baseline execution time data
    baseline_execution_time = read_json_data(baseline_folder)

    # Store baseline statistics
    baseline_mean = pl.Series(baseline_execution_time).mean()
    baseline_std = pl.Series(baseline_execution_time).std()
    results["baseline"] = {
        "mean": baseline_mean,
        "std": baseline_std,
        "min": min(baseline_execution_time),
        "max": max(baseline_execution_time),
        "cov": (baseline_std / baseline_mean) * 100
    }

    # Initialize variables for tracking best optimization
    best_optimised_folder = None
    best_optimised_mean = float("inf")
    best_p_value = 1.0

    # Store samples for combined plotting
    baseline_sample = None
    optimised_samples = []

    # Store all execution times for box plots
    optimised_files_data = []

    for optimised_folder in benchmark_folders:
        if optimised_folder == baseline_folder:
            continue

        optimised_execution_time = read_json_data(optimised_folder)
        
        # Store execution time data for box plots
        optimised_files_data.append((optimised_execution_time, optimised_folder))

        current_sample_size = min(
            sample_size, len(baseline_execution_time), len(optimised_execution_time)
        )

        baseline_sample, optimised_sample, optimised_mean = compare_execution_times(
            baseline_execution_time,
            optimised_execution_time,
            current_sample_size,
            random_state,
        )

        # Store samples for plotting
        optimised_samples.append(optimised_sample)

        # Calculate statistics
        t_stat, p_value = ttest_ind(baseline_sample, optimised_sample)
        percentage_diff = calculate_percentage_difference(baseline_mean, optimised_mean)
        is_significant = is_statistically_significant(baseline_sample, optimised_sample)

        # Store comparison results
        comparison_result = {
            "folder": os.path.basename(optimised_folder),
            "mean": float(optimised_mean),
            "std": float(optimised_sample.std()),
            "min": float(optimised_sample.min()),
            "max": float(optimised_sample.max()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "percentage_difference": float(percentage_diff),
            "is_significant": bool(is_significant)
        }
        results["comparisons"].append(comparison_result)

        # Track best optimization
        if is_significant and (optimised_mean < best_optimised_mean or not best_optimised_folder):
            best_optimised_mean = optimised_mean
            best_optimised_folder = optimised_folder
            best_p_value = p_value

    # Store best optimization results
    if best_optimised_folder:
        results["best_optimization"] = {
            "folder": os.path.basename(best_optimised_folder),
            "mean": float(best_optimised_mean),
            "p_value": float(best_p_value),
            "improvement": float(calculate_percentage_difference(baseline_mean, best_optimised_mean))
        }

    # Generate plots
    plot_combined_execution_times(baseline_sample, optimised_samples, benchmark_folders[1:], output_folder)
    plot_boxplots(baseline_execution_time, optimised_files_data, output_folder)
    create_pairwise_comparison_heatmap(benchmark_folders, output_folder)
    
    baseline_mean = sum(baseline_execution_time) / len(baseline_execution_time)
    optimized_folders = [f for f in benchmark_folders if f != baseline_folder]
    optimized_means = [sum(read_json_data(f)) / len(read_json_data(f)) for f in optimized_folders]
    
    plot_percentage_improvement(
        baseline_mean,
        optimized_means,
        optimized_folders,
        output_folder,
        baseline_folder,
    )

    # Save results to JSON file and tables
    save_results_to_json(results, output_folder)
    generate_typst_table(results, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare execution times for baseline and optimised datasets."
    )
    parser.add_argument(
        "root_folder",
        type=str,
        help="Path to the root folder containing baseline and optimised folders",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=30,
        help="Sample size for each dataset (default: 20)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    main(args.root_folder, args.sample_size, args.random_state)