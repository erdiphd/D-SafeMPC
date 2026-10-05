#!/usr/bin/env python3
"""
Helper script to load saved all_results and run publication analysis

This script shows how to:
1. Load saved all_results from logs folder
2. Use the results analysis system for publication plots and tables
"""

import os
import pickle
import numpy as np
import glob
from datetime import datetime

# Project path (defaults to repository root)
path_str = os.path.dirname(os.path.abspath(__file__))

def find_latest_results_file(logs_dir, experiment_pattern="*", format_type="pkl"):
    """Find the most recent results file matching the pattern"""
    pattern = f"{logs_dir}/all_results_{experiment_pattern}*.{format_type}"
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, return most recent
    return max(files, key=os.path.getmtime)

def load_results_from_logs(experiment_name=None, halfspace_variant=None, 
                          format_type="pkl", logs_dir=None):
    """
    Load saved results from logs folder
    
    Args:
        experiment_name: e.g., "avoiding_d3il" (optional)
        halfspace_variant: e.g., "top_left_hard" (optional) 
        format_type: "pkl", "json", or "npz"
        logs_dir: custom logs directory (optional)
    """
    if logs_dir is None:
        logs_dir = f'{path_str}/logs'
    
    # Build search pattern
    pattern = "all_results"
    if experiment_name:
        pattern += f"_{experiment_name}"
    if halfspace_variant:
        pattern += f"_{halfspace_variant}"
    
    # Find matching files
    search_pattern = f"{logs_dir}/{pattern}*.{format_type}"
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"❌ No results files found matching: {search_pattern}")
        return None
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getmtime)
    print(f"📂 Loading results from: {latest_file}")
    
    # Load based on format
    if format_type == "pkl":
        with open(latest_file, 'rb') as f:
            results = pickle.load(f)
    elif format_type == "json":
        import json
        with open(latest_file, 'r') as f:
            json_results = json.load(f)
        # Convert lists back to numpy arrays
        results = {}
        for key, value in json_results.items():
            results[key] = {k: np.array(v) if isinstance(v, list) else v 
                           for k, v in value.items()}
    elif format_type == "npz":
        npz_data = np.load(latest_file, allow_pickle=True)
        results = {}
        for key in npz_data.files:
            results[key] = npz_data[key].item()
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    print(f"✅ Loaded {len(results)} result configurations")
    return results

def run_analysis_on_saved_results(results_file=None, output_dir="publication_analysis"):
    """
    Run the publication analysis on saved results
    """
    # Load results
    if results_file:
        print(f"📂 Loading specific file: {results_file}")
        with open(results_file, 'rb') as f:
            all_results = pickle.load(f)
    else:
        # Find latest results
        all_results = load_results_from_logs()
        if all_results is None:
            return
    
    # Import and use the analysis system
    import sys
    sys.path.append('scripts')
    from analyze_my_results import ResultsAnalyzer
    
    # Create analyzer and run analysis
    analyzer = ResultsAnalyzer(all_results)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run comprehensive analysis
    print("\n🔄 Running publication analysis...")
    analyzer.create_publication_bar_charts(save_dir=output_dir)
    
    # Generate tables
    analyzer.create_latex_table(save_dir=output_dir)
    analyzer.create_summary_report(save_dir=output_dir)
    
    print(f"✅ Analysis complete! Results saved in: {output_dir}")
    return analyzer

def list_available_results():
    """List all available results files in logs"""
    logs_dir = f'{path_str}/logs'
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory not found: {logs_dir}")
        return
    
    print(f"📁 Available results files in {logs_dir}:")
    print("=" * 50)
    
    for format_type in ['pkl', 'json', 'npz']:
        files = glob.glob(f"{logs_dir}/all_results*.{format_type}")
        if files:
            print(f"\n{format_type.upper()} files:")
            for file in sorted(files, key=os.path.getmtime, reverse=True):
                size = os.path.getsize(file)
                mtime = datetime.fromtimestamp(os.path.getmtime(file))
                print(f"  📄 {os.path.basename(file)}")
                print(f"     Size: {size:,} bytes, Modified: {mtime}")

def quick_analysis_example():
    """Quick example of loading and analyzing saved results"""
    print("🚀 Quick Analysis Example")
    print("=" * 40)
    
    # Load the most recent results
    all_results = load_results_from_logs()
    if all_results is None:
        print("❌ No results found. Run eval.py first to generate results.")
        return
    
    # Quick summary
    print(f"\n📊 Quick Summary:")
    print(f"   Total configurations: {len(all_results)}")
    print(f"   Configuration keys: {list(all_results.keys())[:5]}{'...' if len(all_results) > 5 else ''}")
    
    # Show sample data structure
    if all_results:
        sample_key = next(iter(all_results))
        sample_data = all_results[sample_key]
        print(f"   Data keys per config: {list(sample_data.keys())}")
        print(f"   Sample sizes: {[(k, len(v) if hasattr(v, '__len__') else type(v).__name__) for k, v in sample_data.items()]}")
    
    # Run analysis
    analyzer = run_analysis_on_saved_results(output_dir="quick_analysis_output")
    
    if analyzer:
        # Show quick stats
        summary = analyzer.get_summary_statistics()
        print("\n📈 Summary Statistics:")
        for variant, stats in summary.items():
            print(f"   {variant}:")
            print(f"     Success Rate: {stats['success_rate']['mean']:.3f} ± {stats['success_rate']['std']:.3f}")
            print(f"     Safety Rate: {stats['safety_success_rate']['mean']:.3f} ± {stats['safety_success_rate']['std']:.3f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and analyze saved evaluation results")
    parser.add_argument("--list", action="store_true", help="List available results files")
    parser.add_argument("--quick", action="store_true", help="Run quick analysis on latest results")
    parser.add_argument("--file", type=str, help="Specific results file to analyze")
    parser.add_argument("--experiment", type=str, help="Filter by experiment name")
    parser.add_argument("--halfspace", type=str, help="Filter by halfspace variant")
    parser.add_argument("--output", type=str, default="publication_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_results()
    elif args.quick:
        quick_analysis_example()
    elif args.file:
        run_analysis_on_saved_results(args.file, args.output)
    else:
        # Default: load filtered results and analyze
        all_results = load_results_from_logs(args.experiment, args.halfspace)
        if all_results:
            run_analysis_on_saved_results(output_dir=args.output)
        else:
            print("❌ No matching results found") 