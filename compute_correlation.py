import polars as pl
import json
import os
import argparse
from scipy import stats
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def perform_pca_analysis(df):
    """Perform PCA analysis on the performance metrics."""
    # Define metrics to include in PCA
    metrics = [
        'dtlb_loads',
        'dtlb_miss_rate',
        'page_faults',
        'mmap_calls',
        'brk_calls',
        'munmap_calls',
        'total_calls'
    ]
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    # Prepare data for PCA
    X = df.select(available_metrics).to_numpy()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Get component loadings (correlations between PCs and original features)
    loadings = pca.components_
    
    # Prepare results dictionary
    pca_results = {
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
        'n_components': len(explained_variance_ratio),
        'metrics': available_metrics,
        'loadings': {
            f'PC{i+1}': {
                metric: float(loading) 
                for metric, loading in zip(available_metrics, loadings[i])
            }
            for i in range(len(loadings))
        },
        'scores': {
            'experiments': df['experiment'].to_list(),
            'values': {
                f'PC{i+1}': X_pca[:, i].tolist()
                for i in range(len(loadings))
            }
        }
    }
    
    return pca_results

def generate_typst_pca_table(pca_results, output_folder):
    """Generates a Typst table for PCA results."""
    
    # Create mapping for metric names
    metric_names = {
        'dtlb_loads': 'dTLB Loads',
        'dtlb_miss_rate': 'dTLB Miss Rate (%)',
        'page_faults': 'Page Faults',
        'mmap_calls': 'mmap() Calls',
        'brk_calls': 'brk() Calls',
        'munmap_calls': 'munmap() Calls',
        'total_calls': 'Total System Calls'
    }

    # Generate main components table
    table_str = """#figure(
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto, auto),
      inset: 6pt,
      align: horizon,
      table.header(
        [*Principal Component*],
        [*Variance Explained (%)*],
        [*Cumulative Variance (%)*],
        [*Most Influential Metrics*],
      ),\n"""

    # Add rows for each principal component
    for i in range(pca_results['n_components']):
        # Get the loadings for this component
        pc_loadings = pca_results['loadings'][f'PC{i+1}']
        
        # Sort metrics by absolute loading value
        sorted_metrics = sorted(
            pc_loadings.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top 2 influential metrics
        top_metrics = []
        for metric, loading in sorted_metrics[:2]:
            sign = '+' if loading > 0 else '-'
            metric_name = metric_names.get(metric, metric)
            top_metrics.append(f"{sign}{metric_name}")
        
        top_metrics_str = ", ".join(top_metrics)
        
        # Add row to table
        table_str += f"""      [PC{i+1}],
      [{pca_results['explained_variance_ratio'][i]*100:.1f}],
      [{pca_results['cumulative_variance_ratio'][i]*100:.1f}],
      [{top_metrics_str}],\n"""

    # Add closing tags and caption
    table_str += """    )
  ],
  caption: [Principal Component Analysis of Performance Metrics],
)
"""

    # Generate loadings table
    loadings_table = """#figure(
  text(size: 9pt)[
    #table(
      columns: (auto, """ + ", ".join(["auto"] * len(pca_results['metrics'])) + """),
      inset: 6pt,
      align: horizon,
      table.header(
        [*Component*],
""" + ",\n".join(f"        [{metric_names.get(metric, metric)}]" for metric in pca_results['metrics']) + """
      ),\n"""

    # Add rows for each principal component
    for i in range(pca_results['n_components']):
        loadings_table += f"""      [PC{i+1}],\n"""
        loadings_table += ",\n".join(
            f"      [{pca_results['loadings'][f'PC{i+1}'][metric]:.3f}]"
            for metric in pca_results['metrics']
        ) + ",\n"

    # Add closing tags and caption
    loadings_table += """    )
  ],
  caption: [PCA Loading Factors for Performance Metrics],
)
"""

    # Write tables to files
    with open(os.path.join(output_folder, "pca_summary_table.txt"), "w") as f:
        f.write(table_str)
    
    with open(os.path.join(output_folder, "pca_loadings_table.txt"), "w") as f:
        f.write(loadings_table)
    
    logging.info("PCA tables saved to output folder")

def save_pca_results(pca_results, output_folder):
    """Save PCA results to JSON file."""
    output_path = os.path.join(output_folder, "pca_results.json")
    with open(output_path, "w") as f:
        json.dump(pca_results, f, indent=4)
    logging.info(f"PCA results saved to {output_path}")

def read_stats_results(stats_folder):
    """Read the results.json from the stats folder."""
    with open(os.path.join(stats_folder, "results.json"), "r") as f:
        stats_data = json.load(f)
    return stats_data

def read_perf_strace_data(folder_path):
    """Read perf and strace data from a folder."""
    perf_data = None
    strace_data = None
    
    for file in os.listdir(folder_path):
        if file.startswith("run_") and file.endswith(".perf.json"):
            with open(os.path.join(folder_path, file), "r") as f:
                perf_data = json.load(f)
        elif file.startswith("run_") and file.endswith(".strace.json"):
            with open(os.path.join(folder_path, file), "r") as f:
                strace_data = json.load(f)
    
    return perf_data, strace_data

def extract_metrics(perf_data, strace_data):
    """Extract the relevant metrics from perf and strace data."""
    metrics = {}
    
    if perf_data:
        metrics.update({
            'dtlb_loads': perf_data['dTLB-loads']['value'],
            'dtlb_miss_rate': perf_data['dTLB-load-misses']['percentage'],
            'page_faults': perf_data['page-faults']['value'],
        })
    
    if strace_data:
        metrics.update({
            'mmap_calls': strace_data['mmap']['calls'],
            'brk_calls': strace_data['brk']['calls'],
            'munmap_calls': strace_data['munmap']['calls'],
            'total_calls': strace_data['summary']['calls'],
        })
    
    return metrics

def calculate_correlations(stats_folder, perf_strace_folder):
    """Calculate correlations between execution times and performance metrics."""
    # Read stats results
    stats_data = read_stats_results(os.path.join(stats_folder))
    
    # Prepare data for correlation analysis
    data = []
    
    # Add baseline data
    baseline_folder = os.path.join(perf_strace_folder, "baseline")
    if os.path.exists(baseline_folder):
        perf_data, strace_data = read_perf_strace_data(baseline_folder)
        metrics = extract_metrics(perf_data, strace_data)
        metrics['mean_time'] = stats_data['baseline']['mean']
        metrics['experiment'] = 'baseline'
        data.append(metrics)
    
    # Add optimization data
    for comp in stats_data['comparisons']:
        folder_name = comp['folder']
        folder_path = os.path.join(perf_strace_folder, folder_name)
        
        if os.path.exists(folder_path):
            perf_data, strace_data = read_perf_strace_data(folder_path)
            metrics = extract_metrics(perf_data, strace_data)
            metrics['mean_time'] = comp['mean']
            metrics['experiment'] = folder_name
            data.append(metrics)
    
    # Convert to Polars DataFrame
    df = pl.DataFrame(data)
    
    # Calculate correlations
    metrics_to_correlate = [
        'dtlb_loads',
        'dtlb_miss_rate',
        'page_faults',
        'mmap_calls',
        'brk_calls',
        'munmap_calls',
        'total_calls'
    ]
    
    correlations = {}
    for metric in metrics_to_correlate:
        if metric in df.columns:
            correlation, p_value = stats.pearsonr(
                df['mean_time'].to_numpy(),
                df[metric].to_numpy()
            )
            correlations[metric] = {
                'correlation': float(correlation),
                'p_value': float(p_value),
                'interpretation': {
                    'strength': 'weak' if abs(correlation) < 0.3 else 
                               'moderate' if abs(correlation) < 0.7 else 
                               'strong',
                    'direction': 'positive' if correlation > 0 else 'negative',
                    'significance': 'highly significant' if p_value < 0.01 else
                                  'significant' if p_value < 0.05 else
                                  'not significant',
                    'practical_significance': (
                        f"{'Stronger' if abs(correlation) > 0.5 else 'Weaker'} "
                        f"relationship with execution time"
                    )
                }
            }
    
    return correlations, df

def save_results(correlations, df, output_folder):
    """Save correlation results and data to JSON files."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Save correlations
    correlation_path = os.path.join(output_folder, "correlations.json")
    with open(correlation_path, "w") as f:
        json.dump(correlations, f, indent=4)
    logging.info(f"Correlations saved to {correlation_path}")
    
    # Save raw data
    data_path = os.path.join(output_folder, "correlation_data.json")
    with open(data_path, "w") as f:
        json.dump(df.to_dict(as_series=False), f, indent=4)
    logging.info(f"Raw data saved to {data_path}")

def generate_typst_correlation_table(correlations, output_folder):
    """Generates a Typst table from correlation results."""
    
    # Create mapping for metric names to make them more readable
    metric_names = {
        'dtlb_loads': 'dTLB Loads',
        'dtlb_miss_rate': 'dTLB Miss Rate (%)',
        'page_faults': 'Page Faults',
        'mmap_calls': '`mmap()` Calls',
        'brk_calls': '`brk()` Calls',
        'munmap_calls': '`munmap()` Calls',
        'total_calls': 'Total System Calls'
    }

    table_str = """#figure(
  text(size: 9pt)[
    #table(
      columns: (auto, auto, auto, auto, auto),
      inset: 6pt,
      align: horizon,
      table.header(
        [*Metric*],
        [*Correlation*],
        [*P-value*],
        [*Significance*],
        [*Relationship*],
      ),\n"""

    # Sort metrics by absolute correlation value to show strongest correlations first
    # sorted_metrics = sorted(
    #     correlations.items(),
    #     key=lambda x: abs(x[1]['correlation']),
    #     reverse=True
    # )

    for metric, values in correlations.items():
        # Format correlation value with sign
        corr = values['correlation']
        corr_str = f"{'+' if corr > 0 else ''}{corr:.3f}"
        
        # Format p-value with red text if not significant
        p_value = values['p_value']
        p_value_str = f'[#text("{p_value:.3f}", fill: red)]' if p_value >= 0.05 else f"[{p_value:.3f}]"
        
        # Determine significance level
        if p_value < 0.001:
            significance = "p < 0.001"
        elif p_value < 0.01:
            significance = "p < 0.01"
        elif p_value < 0.05:
            significance = "p < 0.05"
        else:
            significance = "Not significant"
        
        # Determine relationship strength and direction
        corr_abs = abs(corr)
        if corr_abs < 0.3:
            strength = "Weak"
        elif corr_abs < 0.7:
            strength = "Moderate"
        else:
            strength = "Strong"
        
        direction = "Positive" if corr > 0 else "Negative"
        relationship = f"{strength} {direction}"

        # Add row to table
        table_str += f"""      [{metric_names.get(metric, metric)}],
      [{corr_str}],
      {p_value_str},
      [{significance}],
      [{relationship}],\n"""

    # Add closing tags and caption
    table_str += """    )
  ],
  caption: [Pearson correlation analysis between execution time and performance metrics],
)
"""

    # Write to file
    output_path = os.path.join(output_folder, "correlation_table.txt")
    with open(output_path, "w") as f:
        f.write(table_str)
    logging.info(f"Correlation table saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate correlations between execution times and performance metrics.")
    parser.add_argument("stats_folder", help="Folder containing the stats results")
    parser.add_argument("perf_strace_folder", help="Folder containing perf and strace results")
    
    args = parser.parse_args()
    
    # Calculate correlations
    correlations, df = calculate_correlations(args.stats_folder, args.perf_strace_folder)
    
    # Create output folder in the stats directory
    output_folder = os.path.join(args.stats_folder, "correlations")
    
    # Save results
    save_results(correlations, df, output_folder)

    # Generate Typst table
    generate_typst_correlation_table(correlations, output_folder)

    # Perform PCA analysis
    pca_results = perform_pca_analysis(df)

    # Save PCA results
    save_pca_results(pca_results, output_folder)
    generate_typst_pca_table(pca_results, output_folder)
    
    # Print correlations
    logging.info("\nCorrelations with execution time:")
    for metric, values in correlations.items():
        correlation = values['correlation']
        p_value = values['p_value']
        significance = "significant" if p_value < 0.05 else "not significant"
        logging.info(f"{metric}:")
        logging.info(f"  Correlation: {correlation:.3f}")
        logging.info(f"  P-value: {p_value:.3f} ({significance})")
        logging.info("-" * 50)

    # Print PCA summary
    logging.info("\nPCA Analysis Summary:")
    for i in range(pca_results['n_components']):
        logging.info(f"PC{i+1}:")
        logging.info(f"  Variance Explained: {pca_results['explained_variance_ratio'][i]*100:.1f}%")
        logging.info(f"  Cumulative Variance: {pca_results['cumulative_variance_ratio'][i]*100:.1f}%")
        logging.info("-" * 50)

if __name__ == "__main__":
    main()