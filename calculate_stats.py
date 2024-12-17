import os
import json
import polars as pl
import matplotlib.pyplot as plt
import logging
from scipy.stats import ttest_ind
import argparse

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


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
    print("=" * 50)


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
    optimised_mean = optimised_sample.mean()

    logging.info(f"Baseline Mean Execution Time (Sample): {baseline_mean:.15f}")
    logging.info(f"Optimised Mean Execution Time (Sample): {optimised_mean:.15f}")
    logging.info(f"T-statistic: {t_stat:.15f}")
    logging.info(f"P-value: {p_value:.15f}")

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
    tick_labels = ["Baseline"]

    # Add optimised data
    for execution_time, folder_path in optimised_files_data:
        data.append(execution_time)
        tick_labels.append(os.path.basename(folder_path))

    # Create box plot
    plt.boxplot(data, labels=tick_labels)

    # Customize the plot
    plt.title("Execution Times Distribution Comparison")
    plt.ylabel("Execution Time")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, "execution_times_boxplot.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Box plot saved to {output_path}")


def plot_combined_execution_times(
    baseline_sample, optimised_samples, optimised_folders, output_folder
):
    """Plots combined execution times from baseline and optimised samples."""
    plt.figure(figsize=(12, 8))
    run_numbers = range(1, len(baseline_sample) + 1)

    # Plot baseline sample
    plt.scatter(run_numbers, baseline_sample, color="blue", label="Baseline Sample")

    # Plot each set of optimised samples
    for optimised_sample, optimised_folder in zip(optimised_samples, optimised_folders):
        plt.scatter(
            run_numbers,
            optimised_sample,
            label=f"{os.path.basename(optimised_folder)} Sample",
        )

    # Add titles and labels
    plt.title("Execution Times Comparison")
    plt.xlabel("Run Number")
    plt.ylabel("Execution Time")
    plt.grid(True)
    plt.legend()

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "combined_execution_time_plot.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Combined plot saved to {output_path}")


def find_benchmark_folders(root_folder):
    """Finds all benchmark folders in the given folder structure."""
    benchmark_folders = []
    for item in os.listdir(root_folder):
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


def main(root_folder, sample_size, random_state):
    output_folder = os.path.join(root_folder, "stats")

    # Find all benchmark folders
    benchmark_folders = find_benchmark_folders(root_folder)

    # Identify baseline folder (assuming it's named 'baseline')
    baseline_folder = next(
        (f for f in benchmark_folders if "baseline" in f.lower()), None
    )
    if not baseline_folder:
        print("[!] Error: No baseline folder found.")
        return

    # Read baseline execution time data
    baseline_execution_time = read_json_data(baseline_folder)

    # Print baseline statistics
    print_baseline_stats(baseline_execution_time)

    # Initialize variables to track the best optimization
    best_optimised_folder = None
    best_optimised_mean = float("inf")

    # Store samples for combined plotting
    baseline_sample = None
    optimised_samples = []

    # Store all execution times for box plots
    optimised_files_data = []

    # Iterate over each optimised folder
    for optimised_folder in benchmark_folders:
        if optimised_folder == baseline_folder:
            continue

        logging.info(f"Optimised Folder: {optimised_folder}")

        # Read optimised execution time data
        optimised_execution_time = read_json_data(optimised_folder)

        # Store execution time data for box plots
        optimised_files_data.append((optimised_execution_time, optimised_folder))

        # Ensure the sample size does not exceed the size of either dataset
        current_sample_size = min(
            sample_size, len(baseline_execution_time), len(optimised_execution_time)
        )

        # Compare execution times and get the samples and mean of the optimised sample
        baseline_sample, optimised_sample, optimised_mean = compare_execution_times(
            baseline_execution_time,
            optimised_execution_time,
            current_sample_size,
            random_state,
        )

        # Track the best optimization based on mean execution time
        if optimised_mean < best_optimised_mean:
            best_optimised_mean = optimised_mean
            best_optimised_folder = optimised_folder

        # Store samples for plotting
        optimised_samples.append(optimised_sample)

        print("=" * 50)

    # Plot combined execution times
    plot_combined_execution_times(
        baseline_sample, optimised_samples, benchmark_folders[1:], output_folder
    )

    # Create box plots
    plot_boxplots(baseline_execution_time, optimised_files_data, output_folder)

    # Display the best optimization
    if best_optimised_folder:
        baseline_mean = sum(baseline_execution_time) / len(baseline_execution_time)
        percentage_diff = calculate_percentage_difference(
            baseline_mean, best_optimised_mean
        )
        logging.info(
            f"Best Optimization: {os.path.basename(best_optimised_folder)} with mean execution time: {best_optimised_mean:.15f}"
        )

        if percentage_diff < 0:
            logging.info(
                f"The best optimization improved performance by {abs(percentage_diff):.2f}%"
            )
        elif percentage_diff > 0:
            logging.info(
                f"The best optimization degraded performance by {percentage_diff:.2f}%"
            )
        else:
            logging.info("The best optimization had no effect on performance")
    else:
        logging.info("No optimization folders provided")


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
        default=20,
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
