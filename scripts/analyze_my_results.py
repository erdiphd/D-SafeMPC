import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
import re
from collections import defaultdict

class ResultsAnalyzer:
    """
    Analyzer for experimental results with automatic variant/seed parsing and publication-quality outputs.
    
    Designed for results dictionaries with structure:
    {
        'variant_name_seed': {
            'n_success': array,
            'n_success_and_constraints': array, 
            'n_steps': array,
            'n_violations': array,
            'total_violations': array
        }
    }
    """
    
    def __init__(self, results_dict: Dict[str, Dict[str, np.ndarray]]):
        """
        Initialize analyzer with results dictionary.
        
        Args:
            results_dict: Dictionary with keys like 'dpcc-t-tightened_1', 'diffmpc_2', etc.
        """
        self.results_dict = results_dict
        self.variants, self.seeds_by_variant = self._parse_variants_and_seeds()
        self.aggregated_results = self._aggregate_results()
        
        # Publication-ready styling
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'lines.linewidth': 2.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
    def _parse_variants_and_seeds(self) -> Tuple[List[str], Dict[str, List[int]]]:
        """Parse variant names and seeds from result keys."""
        variants = set()
        seeds_by_variant = defaultdict(list)
        
        for key in self.results_dict.keys():
            # Extract variant name and seed
            # Assumes format: variant_name_seed (e.g., 'dpcc-t-tightened_1', 'diffmpc_2')
            parts = key.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                seed = int(parts[-1])
                variant = '_'.join(parts[:-1])
            else:
                # Fallback: assume everything is variant name, seed = 0
                variant = key
                seed = 0
                
            variants.add(variant)
            seeds_by_variant[variant].append(seed)
        
        # Sort variants and seeds
        variants = sorted(list(variants))
        for variant in seeds_by_variant:
            seeds_by_variant[variant] = sorted(seeds_by_variant[variant])
            
        print(f"📊 Detected variants: {variants}")
        print(f"📊 Seeds per variant: {dict(seeds_by_variant)}")
        
        return variants, dict(seeds_by_variant)
    
    def _aggregate_results(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Aggregate results across seeds for each variant."""
        aggregated = {}
        
        for variant in self.variants:
            variant_data = defaultdict(list)
            
            # Collect data across all seeds for this variant
            for seed in self.seeds_by_variant[variant]:
                key = f"{variant}_{seed}"
                if key in self.results_dict:
                    for metric, values in self.results_dict[key].items():
                        if isinstance(values, np.ndarray):
                            # Calculate summary statistics
                            variant_data[f"{metric}_mean"].append(np.mean(values))
                            variant_data[f"{metric}_success_rate"].append(np.mean(values > 0) if 'success' in metric else np.mean(values))
                            variant_data[f"{metric}_total"].append(np.sum(values))
                            variant_data[f"{metric}_std"].append(np.std(values))
            
            # Calculate mean and std across seeds
            aggregated[variant] = {}
            for metric, values in variant_data.items():
                if values:  # Only if we have data
                    mean_val = np.mean(values)
                    std_val = np.std(values) if len(values) > 1 else 0.0
                    aggregated[variant][metric] = (mean_val, std_val)
        
        return aggregated
    
    def create_publication_bar_charts(self, save_dir: str = "results_analysis", 
                                     metrics: List[str] = None) -> None:
        """Create publication-quality bar charts."""
        if metrics is None:
            metrics = ['n_success_success_rate', 'n_success_and_constraints_success_rate', 
                      'n_violations_mean', 'total_violations_mean', 'n_steps_mean']
        
        Path(save_dir).mkdir(exist_ok=True)
        
        # Publication-friendly colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for metric in metrics:
            if not any(metric in variant_data for variant_data in self.aggregated_results.values()):
                print(f"⚠️  Metric {metric} not found in aggregated results")
                continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            variants = []
            means = []
            stds = []
            
            for variant in self.variants:
                if metric in self.aggregated_results[variant]:
                    variants.append(variant.replace('_', '-').replace('-', ' ').title())
                    mean_val, std_val = self.aggregated_results[variant][metric]
                    means.append(mean_val)
                    stds.append(std_val)
            
            if not means:
                print(f"⚠️  No data for metric {metric}")
                continue
            
            # Create bar chart
            x_pos = np.arange(len(variants))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                         color=colors[:len(variants)], alpha=0.8, 
                         edgecolor='black', linewidth=1)
            
            # Styling
            ax.set_xlabel('Method', fontweight='bold')
            ax.set_ylabel(self._get_metric_label(metric), fontweight='bold')
            ax.set_title(f'{self._get_metric_title(metric)}', fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(variants, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, mean_val, std_val in zip(bars, means, stds):
                height = bar.get_height()
                if 'rate' in metric:
                    label = f'{mean_val:.2f}±{std_val:.2f}'
                else:
                    label = f'{mean_val:.1f}±{std_val:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01*max(means),
                       label, ha='center', va='bottom', fontweight='bold')
            
            # Grid and layout
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            plt.tight_layout()
            
            # Save in multiple formats
            metric_clean = metric.replace('_', '-')
            fig.savefig(f'{save_dir}/bar_chart_{metric_clean}.png', dpi=300, bbox_inches='tight')
            fig.savefig(f'{save_dir}/bar_chart_{metric_clean}.pdf', bbox_inches='tight')
            plt.close(fig)
            
        print(f"✅ Bar charts saved to {save_dir}/")
    
    def create_latex_table(self, save_dir: str = "results_analysis") -> str:
        """Create LaTeX table with all results."""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Define metrics to include in table
        key_metrics = [
            ('n_success_success_rate', 'Success Rate'),
            ('n_success_and_constraints_success_rate', 'Safe Success Rate'),
            ('n_violations_mean', 'Avg. Violations'),
            ('total_violations_mean', 'Total Violation Magnitude'),
            ('n_steps_mean', 'Avg. Steps')
        ]
        
        # Start LaTeX table
        latex = r"\begin{table}[htbp]" + "\n"
        latex += r"\centering" + "\n"
        latex += r"\caption{Experimental Results Summary}" + "\n"
        latex += r"\label{tab:results}" + "\n"
        latex += r"\begin{tabular}{l" + "c" * len(key_metrics) + "}\n"
        latex += r"\toprule" + "\n"
        
        # Header
        header = "Method"
        for _, label in key_metrics:
            header += f" & {label}"
        header += r" \\" + "\n"
        latex += header
        latex += r"\midrule" + "\n"
        
        # Data rows
        for variant in self.variants:
            row = variant.replace('_', '-').replace('-', ' ').title()
            
            for metric_key, _ in key_metrics:
                if metric_key in self.aggregated_results[variant]:
                    mean_val, std_val = self.aggregated_results[variant][metric_key]
                    if 'rate' in metric_key:
                        formatted = f"{mean_val:.3f} \\pm {std_val:.3f}"
                    else:
                        formatted = f"{mean_val:.1f} \\pm {std_val:.1f}"
                    row += f" & ${formatted}$"
                else:
                    row += " & --"
            
            row += r" \\" + "\n"
            latex += row
        
        # Footer
        latex += r"\bottomrule" + "\n"
        latex += r"\end{tabular}" + "\n"
        latex += r"\end{table}" + "\n"
        
        # Save to file
        latex_file = f'{save_dir}/results_table.tex'
        with open(latex_file, 'w') as f:
            f.write(latex)
        
        print(f"✅ LaTeX table saved to {latex_file}")
        return latex
    
    def create_summary_report(self, save_dir: str = "results_analysis") -> None:
        """Create comprehensive summary report."""
        Path(save_dir).mkdir(exist_ok=True)
        
        report = []
        report.append("# Experimental Results Analysis Report\n")
        report.append(f"**Number of variants:** {len(self.variants)}\n")
        report.append(f"**Variants:** {', '.join(self.variants)}\n")
        report.append(f"**Total configurations:** {len(self.results_dict)}\n\n")
        
        report.append("## Summary Statistics\n\n")
        
        for variant in self.variants:
            report.append(f"### {variant.replace('_', '-').title()}\n")
            
            # Key metrics
            if 'n_success_success_rate' in self.aggregated_results[variant]:
                mean_val, std_val = self.aggregated_results[variant]['n_success_success_rate']
                report.append(f"- **Success Rate:** {mean_val:.3f} ± {std_val:.3f}\n")
            
            if 'n_success_and_constraints_success_rate' in self.aggregated_results[variant]:
                mean_val, std_val = self.aggregated_results[variant]['n_success_and_constraints_success_rate']
                report.append(f"- **Safe Success Rate:** {mean_val:.3f} ± {std_val:.3f}\n")
            
            if 'n_violations_mean' in self.aggregated_results[variant]:
                mean_val, std_val = self.aggregated_results[variant]['n_violations_mean']
                report.append(f"- **Average Violations:** {mean_val:.2f} ± {std_val:.2f}\n")
            
            if 'n_steps_mean' in self.aggregated_results[variant]:
                mean_val, std_val = self.aggregated_results[variant]['n_steps_mean']
                report.append(f"- **Average Steps:** {mean_val:.1f} ± {std_val:.1f}\n")
                
            report.append(f"- **Number of seeds:** {len(self.seeds_by_variant[variant])}\n\n")
        
        # Save report
        report_text = ''.join(report)
        with open(f'{save_dir}/analysis_report.md', 'w') as f:
            f.write(report_text)
            
        print(f"✅ Analysis report saved to {save_dir}/analysis_report.md")
        return report_text
    
    def _get_metric_label(self, metric: str) -> str:
        """Get formatted label for metric."""
        labels = {
            'n_success_success_rate': 'Success Rate',
            'n_success_and_constraints_success_rate': 'Safe Success Rate', 
            'n_violations_mean': 'Average Violations',
            'total_violations_mean': 'Total Violation Magnitude',
            'n_steps_mean': 'Average Steps'
        }
        return labels.get(metric, metric.replace('_', ' ').title())
    
    def _get_metric_title(self, metric: str) -> str:
        """Get formatted title for metric."""
        titles = {
            'n_success_success_rate': 'Task Success Rate by Method',
            'n_success_and_constraints_success_rate': 'Safe Success Rate by Method',
            'n_violations_mean': 'Average Constraint Violations by Method', 
            'total_violations_mean': 'Total Violation Magnitude by Method',
            'n_steps_mean': 'Average Steps to Completion by Method'
        }
        return titles.get(metric, self._get_metric_label(metric))
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get summary results as pandas DataFrame."""
        rows = []
        
        for variant in self.variants:
            row = {'Variant': variant}
            
            for metric, (mean_val, std_val) in self.aggregated_results[variant].items():
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "="*60)
        print("📊 EXPERIMENTAL RESULTS SUMMARY")
        print("="*60)
        
        for variant in self.variants:
            print(f"\n🔹 {variant.replace('_', '-').upper()}:")
            
            # Success rates
            if 'n_success_success_rate' in self.aggregated_results[variant]:
                mean_val, std_val = self.aggregated_results[variant]['n_success_success_rate']
                print(f"   Success Rate: {mean_val:.3f} ± {std_val:.3f}")
                
            if 'n_success_and_constraints_success_rate' in self.aggregated_results[variant]:
                mean_val, std_val = self.aggregated_results[variant]['n_success_and_constraints_success_rate']
                print(f"   Safe Success: {mean_val:.3f} ± {std_val:.3f}")
            
            # Performance metrics
            if 'n_violations_mean' in self.aggregated_results[variant]:
                mean_val, std_val = self.aggregated_results[variant]['n_violations_mean']
                print(f"   Avg Violations: {mean_val:.2f} ± {std_val:.2f}")
                
            if 'n_steps_mean' in self.aggregated_results[variant]:
                mean_val, std_val = self.aggregated_results[variant]['n_steps_mean']
                print(f"   Avg Steps: {mean_val:.1f} ± {std_val:.1f}")
                
            print(f"   Seeds: {len(self.seeds_by_variant[variant])}")
        
        print("\n" + "="*60)


def load_and_analyze_results(file_path: str, save_dir: str = "results_analysis") -> ResultsAnalyzer:
    """
    Load results from file and create analyzer.
    
    Args:
        file_path: Path to results file (.pkl, .json, or .npz)
        save_dir: Directory to save analysis outputs
        
    Returns:
        ResultsAnalyzer instance
    """
    file_path = Path(file_path)
    
    print(f"📂 Loading results from: {file_path}")
    
    # Load based on file extension
    if file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            results_dict = pickle.load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        # Convert lists back to numpy arrays
        results_dict = {}
        for key, value in json_data.items():
            results_dict[key] = {k: np.array(v) if isinstance(v, list) else v 
                               for k, v in value.items()}
    elif file_path.suffix == '.npz':
        data = np.load(file_path, allow_pickle=True)
        results_dict = {}
        for key in data.files:
            results_dict[key] = data[key].item()  # .item() to get the dictionary
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"✅ Loaded {len(results_dict)} result configurations")
    
    # Create analyzer
    analyzer = ResultsAnalyzer(results_dict)
    
    # Run full analysis
    print(f"\n🔄 Running comprehensive analysis...")
    analyzer.print_summary()
    analyzer.create_publication_bar_charts(save_dir)
    analyzer.create_latex_table(save_dir)
    analyzer.create_summary_report(save_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {save_dir}")
    
    return analyzer


# Example usage
if __name__ == "__main__":
    # Example with sample data structure matching your format
    sample_results = {
        'dpcc-t-tightened_1': {
            'n_success': np.array([1, 0, 1, 1, 0]),
            'n_success_and_constraints': np.array([1, 0, 0, 1, 0]),
            'n_steps': np.array([45, 67, 23, 34, 89]),
            'n_violations': np.array([0, 2, 1, 0, 3]),
            'total_violations': np.array([0.0, 0.15, 0.05, 0.0, 0.23])
        },
        'diffmpc_1': {
            'n_success': np.array([0, 0, 1, 0, 1]),
            'n_success_and_constraints': np.array([0, 0, 0, 0, 1]),
            'n_steps': np.array([78, 82, 45, 91, 56]),
            'n_violations': np.array([3, 4, 1, 5, 2]),
            'total_violations': np.array([0.34, 0.45, 0.12, 0.67, 0.23])
        }
    }
    
    # Create analyzer and run analysis
    analyzer = ResultsAnalyzer(sample_results)
    analyzer.print_summary()
    analyzer.create_publication_bar_charts("sample_analysis")
    analyzer.create_latex_table("sample_analysis")
