"""
Model comparison page for Streamlit app
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json

st.set_page_config(
    page_title="Model Comparison - Wildfire DSS",
    page_icon="C",
    layout="wide"
)

st.title("Model Comparison")
st.markdown("Compare performance of different trained models")
st.markdown("---")


def load_evaluation_metrics(metrics_path: str):
    """Load evaluation metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def main():
    # Look for evaluation results
    results_dir = Path("results/metrics")

    if not results_dir.exists():
        st.error("No evaluation results found")
        st.info("Run evaluation first: `python scripts/evaluate_model.py`")
        return

    # Find all evaluation metric files
    metric_files = list(results_dir.glob("evaluation_metrics_*.json"))

    if not metric_files:
        st.warning("No evaluation metrics found")
        return

    # Load metrics
    all_metrics = {}
    for metric_file in metric_files:
        model_name = metric_file.stem
        all_metrics[model_name] = load_evaluation_metrics(metric_file)

    # Create comparison dataframe
    comparison_data = []

    for model_name, metrics in all_metrics.items():
        overall = metrics['overall']
        safety = metrics['safety']

        comparison_data.append({
            'Model': model_name,
            'Accuracy': overall['accuracy'],
            'Precision': overall['precision'],
            'Recall': overall['recall'],
            'F1 Score': overall['f1_score'],
            'AUC-ROC': overall.get('auc_roc', 0),
            'False Negative Rate': safety.get('false_negative_rate', 0)
        })

    df = pd.DataFrame(comparison_data)

    # Display table
    st.subheader("Performance Comparison")
    st.dataframe(df.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1 Score': '{:.4f}',
        'AUC-ROC': '{:.4f}',
        'False Negative Rate': '{:.4f}'
    }), use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Performance Metrics")

        fig = go.Figure()

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Model'],
                y=df[metric]
            ))

        fig.update_layout(
            barmode='group',
            yaxis_title='Score',
            xaxis_title='Model',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Safety-Critical Metrics")

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            name='False Negative Rate',
            x=df['Model'],
            y=df['False Negative Rate'],
            marker_color='red'
        ))

        fig2.update_layout(
            yaxis_title='Rate',
            xaxis_title='Model',
            height=400
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Best model summary
    st.markdown("---")
    st.subheader("Best Models by Metric")

    col1, col2, col3 = st.columns(3)

    with col1:
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        st.metric(
            "Best Accuracy",
            f"{best_accuracy['Accuracy']:.4f}",
            delta=None,
            delta_color="normal"
        )
        st.caption(f"Model: {best_accuracy['Model']}")

    with col2:
        best_recall = df.loc[df['Recall'].idxmax()]
        st.metric(
            "Best Recall (Safety)",
            f"{best_recall['Recall']:.4f}",
            delta=None,
            delta_color="normal"
        )
        st.caption(f"Model: {best_recall['Model']}")

    with col3:
        best_fn = df.loc[df['False Negative Rate'].idxmin()]
        st.metric(
            "Lowest False Negative Rate",
            f"{best_fn['False Negative Rate']:.4f}",
            delta=None,
            delta_color="inverse"
        )
        st.caption(f"Model: {best_fn['Model']}")


if __name__ == "__main__":
    main()
