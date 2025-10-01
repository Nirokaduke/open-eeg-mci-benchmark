"""
Evidence Synthesis Map Generator for EEG-MCI-Bench

Creates interactive visualizations synthesizing results across multiple models,
features, and studies for systematic comparison and meta-analysis.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("Set2")


class EvidenceMapGenerator:
    """Generate evidence synthesis visualizations and meta-analysis."""

    def __init__(self,
                 results_dir: str,
                 output_dir: str = "reports",
                 literature_data: Optional[str] = None):
        """
        Initialize evidence map generator.

        Parameters
        ----------
        results_dir : str
            Directory containing model results
        output_dir : str
            Output directory for reports
        literature_data : str, optional
            Path to literature review data files
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.literature_data = Path(literature_data) if literature_data else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self) -> pd.DataFrame:
        """Load model results into DataFrame."""
        results_list = []

        for result_file in self.results_dir.glob("*_results.json"):
            model_name = result_file.stem.replace("_results", "")
            with open(result_file, 'r') as f:
                result = json.load(f)

            metrics = result.get('test_metrics', {})
            row = {
                'model': model_name,
                'f1': metrics.get('f1', 0),
                'f1_ci_lower': metrics.get('f1_ci_lower', 0),
                'f1_ci_upper': metrics.get('f1_ci_upper', 0),
                'auc': metrics.get('auc', 0),
                'auc_ci_lower': metrics.get('auc_ci_lower', 0),
                'auc_ci_upper': metrics.get('auc_ci_upper', 0),
                'mcc': metrics.get('mcc', 0),
                'sensitivity': metrics.get('sensitivity', 0),
                'specificity': metrics.get('specificity', 0),
                'accuracy': metrics.get('accuracy', 0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0)
            }
            results_list.append(row)

        return pd.DataFrame(results_list)

    def load_literature_data(self) -> Optional[pd.DataFrame]:
        """Load literature review data if available."""
        if not self.literature_data or not self.literature_data.exists():
            return None

        try:
            # Try reading as TSV first
            df = pd.read_csv(self.literature_data, sep="\t", engine="python")
        except Exception:
            try:
                # Try CSV
                df = pd.read_csv(self.literature_data, sep=",", engine="python")
            except Exception as e:
                print(f"Warning: Could not load literature data: {e}")
                return None

        return df

    def generate_all_visualizations(self):
        """Generate complete evidence synthesis package."""
        print("\nGenerating Evidence Synthesis Visualizations...")

        results_df = self.load_results()

        # 1. Model comparison chart
        self._generate_model_comparison(results_df)

        # 2. Forest plots for metrics
        self._generate_forest_plots(results_df)

        # 3. Feature importance synthesis
        self._generate_feature_importance_synthesis()

        # 4. Performance heatmap
        self._generate_performance_heatmap(results_df)

        # 5. Interactive dashboard
        self._generate_interactive_dashboard(results_df)

        # 6. Literature comparison (if available)
        lit_df = self.load_literature_data()
        if lit_df is not None:
            self._generate_literature_comparison(results_df, lit_df)

        # 7. Summary table
        self._generate_summary_table(results_df)

        print(f"\n✓ Evidence synthesis complete! Files saved to: {self.output_dir}")

    def _generate_model_comparison(self, df: pd.DataFrame):
        """Generate model comparison bar chart."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ('f1', 'F1 Score', axes[0, 0]),
            ('auc', 'AUC', axes[0, 1]),
            ('mcc', 'MCC', axes[1, 0]),
            ('balanced_accuracy', 'Balanced Accuracy', axes[1, 1])
        ]

        for metric, title, ax in metrics:
            if metric in df.columns:
                # Sort by metric value
                sorted_df = df.sort_values(metric, ascending=False)

                x = np.arange(len(sorted_df))
                y = sorted_df[metric].values

                # Get CI if available
                ci_lower_col = f'{metric}_ci_lower'
                ci_upper_col = f'{metric}_ci_upper'

                if ci_lower_col in sorted_df.columns:
                    yerr_lower = y - sorted_df[ci_lower_col].values
                    yerr_upper = sorted_df[ci_upper_col].values - y
                    yerr = [yerr_lower, yerr_upper]
                else:
                    yerr = None

                bars = ax.bar(x, y, color='steelblue', alpha=0.8, edgecolor='black')
                if yerr is not None:
                    ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black',
                               capsize=5, capthick=2)

                # Highlight best model
                best_idx = y.argmax()
                bars[best_idx].set_color('darkgreen')
                bars[best_idx].set_alpha(1.0)

                ax.set_xticks(x)
                ax.set_xticklabels(sorted_df['model'].values, rotation=45, ha='right')
                ax.set_ylabel(title, fontsize=12, fontweight='bold')
                ax.set_title(f'{title} by Model', fontsize=13, fontweight='bold')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_ylim([0, 1])

                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, y)):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        fig_path = self.output_dir / 'model_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

    def _generate_forest_plots(self, df: pd.DataFrame):
        """Generate forest plots for key metrics."""
        metrics_to_plot = [
            ('f1', 'F1 Score'),
            ('auc', 'Area Under ROC Curve'),
            ('sensitivity', 'Sensitivity'),
            ('specificity', 'Specificity')
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx]

            if metric not in df.columns:
                continue

            # Sort by metric value
            sorted_df = df.sort_values(metric, ascending=True)

            y_pos = np.arange(len(sorted_df))
            values = sorted_df[metric].values

            # Get CI if available
            ci_lower_col = f'{metric}_ci_lower'
            ci_upper_col = f'{metric}_ci_upper'

            if ci_lower_col in sorted_df.columns:
                ci_lower = sorted_df[ci_lower_col].values
                ci_upper = sorted_df[ci_upper_col].values
                has_ci = True
            else:
                # Estimate CI as +/- 0.05 for visualization
                ci_lower = values - 0.05
                ci_upper = values + 0.05
                has_ci = False

            # Plot points and CIs
            ax.plot(values, y_pos, 'o', markersize=10, color='darkblue',
                   markeredgewidth=2, markeredgecolor='black', zorder=3)

            for i, (val, low, high) in enumerate(zip(values, ci_lower, ci_upper)):
                ax.plot([low, high], [i, i], 'k-', linewidth=2, zorder=2)
                ax.plot([low, low], [i-0.15, i+0.15], 'k-', linewidth=2, zorder=2)
                ax.plot([high, high], [i-0.15, i+0.15], 'k-', linewidth=2, zorder=2)

            # Add reference line at overall mean
            mean_val = values.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      alpha=0.5, label=f'Mean: {mean_val:.3f}', zorder=1)

            # Formatting
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_df['model'].values)
            ax.set_xlabel(f'{title} (95% CI)', fontsize=11, fontweight='bold')
            ax.set_title(f'Forest Plot: {title}', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle=':')
            ax.set_xlim([0, 1])
            ax.legend(loc='lower right', fontsize=9)

            # Add value labels
            for i, val in enumerate(values):
                ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

        plt.suptitle('Forest Plots - Performance Metrics with 95% CI',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        fig_path = self.output_dir / 'forest_plots.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

    def _generate_feature_importance_synthesis(self):
        """Synthesize feature importance across models."""
        # Load feature importance from all models
        feature_scores = {}

        for result_file in self.results_dir.glob("*_results.json"):
            model_name = result_file.stem.replace("_results", "")
            with open(result_file, 'r') as f:
                result = json.load(f)

            importance = result.get('feature_importance', {})
            if not importance:
                continue

            for feature, score in importance.items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append((model_name, score))

        if not feature_scores:
            print("  No feature importance data found")
            return

        # Calculate aggregated importance
        aggregated = {}
        for feature, scores in feature_scores.items():
            values = [s[1] for s in scores]
            aggregated[feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }

        # Sort by mean importance
        sorted_features = sorted(aggregated.items(),
                                key=lambda x: x[1]['mean'], reverse=True)[:25]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        features = [f[0] for f in sorted_features]
        means = [f[1]['mean'] for f in sorted_features]
        stds = [f[1]['std'] for f in sorted_features]

        y_pos = np.arange(len(features))

        ax.barh(y_pos, means, xerr=stds, color='coral', alpha=0.7,
               edgecolor='black', error_kw={'ecolor': 'black', 'capsize': 3})

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score (Mean ± SD)', fontsize=12, fontweight='bold')
        ax.set_title('Top 25 Features - Aggregated Importance Across Models',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        fig_path = self.output_dir / 'feature_importance_synthesis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

    def _generate_performance_heatmap(self, df: pd.DataFrame):
        """Generate heatmap of all performance metrics."""
        # Select metrics for heatmap
        metric_cols = ['f1', 'auc', 'mcc', 'sensitivity', 'specificity',
                      'accuracy', 'balanced_accuracy']
        available_cols = [col for col in metric_cols if col in df.columns]

        if not available_cols:
            return

        # Create matrix
        matrix = df[['model'] + available_cols].set_index('model')

        # Normalize to 0-100 scale
        matrix_normalized = matrix * 100

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))

        sns.heatmap(matrix_normalized, annot=True, fmt='.1f', cmap='RdYlGn',
                   vmin=0, vmax=100, cbar_kws={'label': 'Score (%)'},
                   linewidths=0.5, linecolor='gray', ax=ax)

        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Models', fontsize=12, fontweight='bold')
        ax.set_title('Performance Heatmap - All Metrics (%)',
                    fontsize=14, fontweight='bold')

        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        fig_path = self.output_dir / 'performance_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

    def _generate_interactive_dashboard(self, df: pd.DataFrame):
        """Generate interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1 Score Comparison', 'AUC Comparison',
                          'Sensitivity vs Specificity', 'All Metrics Radar'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'polar'}]]
        )

        # 1. F1 Score bar chart
        fig.add_trace(
            go.Bar(x=df['model'], y=df['f1'], name='F1 Score',
                  marker_color='steelblue',
                  error_y=dict(
                      type='data',
                      symmetric=False,
                      array=df['f1_ci_upper'] - df['f1'],
                      arrayminus=df['f1'] - df['f1_ci_lower']
                  ) if 'f1_ci_lower' in df.columns else None),
            row=1, col=1
        )

        # 2. AUC bar chart
        fig.add_trace(
            go.Bar(x=df['model'], y=df['auc'], name='AUC',
                  marker_color='coral',
                  error_y=dict(
                      type='data',
                      symmetric=False,
                      array=df['auc_ci_upper'] - df['auc'],
                      arrayminus=df['auc'] - df['auc_ci_lower']
                  ) if 'auc_ci_lower' in df.columns else None),
            row=1, col=2
        )

        # 3. Sensitivity vs Specificity scatter
        fig.add_trace(
            go.Scatter(x=df['sensitivity'], y=df['specificity'],
                      mode='markers+text', name='Models',
                      marker=dict(size=12, color=df['f1'],
                                 colorscale='Viridis', showscale=True,
                                 colorbar=dict(title='F1 Score', x=0.46)),
                      text=df['model'],
                      textposition='top center'),
            row=2, col=1
        )

        # Add diagonal reference line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(dash='dash', color='gray'),
                      name='Equal Sens/Spec', showlegend=False),
            row=2, col=1
        )

        # 4. Radar chart for best model
        best_model_idx = df['f1'].idxmax()
        best_model = df.loc[best_model_idx]

        categories = ['F1', 'AUC', 'MCC', 'Sensitivity', 'Specificity', 'Accuracy']
        values = [
            best_model['f1'],
            best_model['auc'],
            best_model['mcc'],
            best_model['sensitivity'],
            best_model['specificity'],
            best_model['accuracy']
        ]

        fig.add_trace(
            go.Scatterpolar(r=values, theta=categories, fill='toself',
                           name=best_model['model']),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_xaxes(title_text="Sensitivity", range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text="F1 Score", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="AUC", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="Specificity", range=[0, 1], row=2, col=1)

        fig.update_layout(
            title_text="EEG-MCI Classification - Interactive Evidence Dashboard",
            title_font_size=20,
            height=800,
            showlegend=True
        )

        # Save interactive HTML
        fig_path = self.output_dir / 'evidence_dashboard.html'
        pio.write_html(fig, fig_path, include_plotlyjs='cdn', full_html=True)
        print(f"  Saved: {fig_path}")

    def _generate_literature_comparison(self, df: pd.DataFrame, lit_df: pd.DataFrame):
        """Compare current results with literature."""
        # Try to extract accuracy/performance from literature data
        # This is a smart heuristic that handles various column names

        # Find accuracy column
        acc_col = self._smart_pick_column(lit_df, ['accuracy', 'acc', 'performance', 'f1'])
        n_col = self._smart_pick_column(lit_df, ['n', 'n_subjects', 'sample_size', 'subjects'])

        if acc_col is None:
            print("  Warning: Could not identify accuracy column in literature data")
            return

        # Clean and convert literature data
        lit_df['_acc'] = pd.to_numeric(lit_df[acc_col].astype(str).str.replace('%', ''),
                                       errors='coerce') / 100
        lit_df = lit_df.dropna(subset=['_acc'])

        if n_col:
            lit_df['_n'] = pd.to_numeric(lit_df[n_col], errors='coerce')
        else:
            lit_df['_n'] = 50  # Default placeholder

        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot literature as scatter
        ax.scatter(lit_df['_n'], lit_df['_acc'], s=50, alpha=0.5,
                  color='gray', label='Literature Studies', marker='o')

        # Plot current study as larger points
        for idx, row in df.iterrows():
            ax.scatter(50, row['f1'], s=200, alpha=0.9,
                      marker='*', edgecolors='black', linewidths=2,
                      label=f"This Study: {row['model']}")

        ax.set_xlabel('Sample Size (n subjects)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Classification Performance (Accuracy/F1)', fontsize=12, fontweight='bold')
        ax.set_title('EEG-MCI Classification: Current Study vs Literature',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])

        plt.tight_layout()
        fig_path = self.output_dir / 'literature_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

        # Also create interactive version
        self._create_interactive_literature_map(lit_df, df)

    def _create_interactive_literature_map(self, lit_df: pd.DataFrame, df: pd.DataFrame):
        """Create interactive evidence map similar to original implementation."""
        fig = go.Figure()

        # Add literature data
        fig.add_trace(go.Scatter(
            x=lit_df['_n'],
            y=lit_df['_acc'] * 100,
            mode='markers',
            name='Literature',
            marker=dict(size=8, color='lightblue', opacity=0.6,
                       line=dict(width=1, color='darkblue')),
            text=[f"Literature Study<br>n={n}<br>Acc={a:.1%}"
                  for n, a in zip(lit_df['_n'], lit_df['_acc'])],
            hoverinfo='text'
        ))

        # Add current study
        for idx, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[50],
                y=[row['f1'] * 100],
                mode='markers',
                name=f"This Study: {row['model']}",
                marker=dict(size=15, symbol='star',
                           line=dict(width=2, color='black')),
                text=f"{row['model']}<br>F1={row['f1']:.3f}<br>AUC={row['auc']:.3f}",
                hoverinfo='text'
            ))

        fig.update_layout(
            title='EEG-MCI Classification Evidence Map: Literature vs Current Study',
            xaxis_title='Sample Size (n subjects)',
            yaxis_title='Classification Accuracy/Performance (%)',
            hovermode='closest',
            height=600
        )

        fig_path = self.output_dir / 'evidence_map.html'
        pio.write_html(fig, fig_path, include_plotlyjs='cdn', full_html=True)
        print(f"  Saved: {fig_path}")

    def _generate_summary_table(self, df: pd.DataFrame):
        """Generate comprehensive summary table."""
        # Sort by F1 score
        df_sorted = df.sort_values('f1', ascending=False).reset_index(drop=True)

        # Create summary table
        summary = []
        for idx, row in df_sorted.iterrows():
            rank = idx + 1
            summary.append({
                'Rank': rank,
                'Model': row['model'],
                'F1 Score': f"{row['f1']:.3f} ({row['f1_ci_lower']:.3f}-{row['f1_ci_upper']:.3f})",
                'AUC': f"{row['auc']:.3f} ({row['auc_ci_lower']:.3f}-{row['auc_ci_upper']:.3f})",
                'MCC': f"{row['mcc']:.3f}",
                'Sensitivity': f"{row['sensitivity']:.3f}",
                'Specificity': f"{row['specificity']:.3f}",
                'Balanced Acc': f"{row['balanced_accuracy']:.3f}"
            })

        summary_df = pd.DataFrame(summary)

        # Save as CSV
        csv_path = self.output_dir / 'performance_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # Generate formatted table visualization
        fig, ax = plt.subplots(figsize=(14, max(6, len(summary_df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(len(summary_df.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')

        # Style best model row
        for i in range(len(summary_df.columns)):
            cell = table[(1, i)]
            cell.set_facecolor('#E8F5E9')

        plt.title('Model Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        fig_path = self.output_dir / 'summary_table.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

    @staticmethod
    def _smart_pick_column(df: pd.DataFrame, candidate_names: List[str]) -> Optional[str]:
        """Smart column name matching with fuzzy logic."""
        cols = {c.lower(): c for c in df.columns}

        for candidate in candidate_names:
            # Exact match
            if candidate in cols:
                return cols[candidate]

            # Partial match
            for lc, orig in cols.items():
                if candidate in lc or lc in candidate:
                    return orig

        return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate evidence synthesis visualizations')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing model results')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for visualizations')
    parser.add_argument('--literature-data', type=str, default=None,
                       help='Path to literature review data file')

    args = parser.parse_args()

    # Generate evidence maps
    generator = EvidenceMapGenerator(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        literature_data=args.literature_data
    )

    generator.generate_all_visualizations()

    print("\n" + "="*60)
    print("Evidence Synthesis Complete!")
    print("="*60)
    print(f"\nGenerated files in: {args.output_dir}/")
    print("  - model_comparison.png")
    print("  - forest_plots.png")
    print("  - feature_importance_synthesis.png")
    print("  - performance_heatmap.png")
    print("  - evidence_dashboard.html (interactive)")
    print("  - performance_summary.csv")
    print("  - summary_table.png")
    if args.literature_data:
        print("  - literature_comparison.png")
        print("  - evidence_map.html (interactive)")


if __name__ == "__main__":
    main()
