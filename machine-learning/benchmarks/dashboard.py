import streamlit as st
import json
import pandas as pd
import os
from datetime import datetime
import base64
from PyPDF2 import PdfReader
import io
import plotly.express as px
import plotly.graph_objects as go
import redis
import re
from scorer import RAG_MODELS
from extractor import EXTRACTOR_KEYS

def connect_redis():
    """Connect to Redis instance"""
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", 6379))
    return redis.Redis(host=host, port=port, decode_responses=True)

def get_runs():
    """Get available benchmark runs from the runs directory"""
    runs = []
    
    # Look for runs in the runs directory
    runs_dir = os.path.join("runs")
    if not os.path.exists(runs_dir):
        return runs
    
    # Get all subdirectories in the runs directory
    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    
    # Connect to Redis for additional run info
    r = connect_redis()
    
    for run_id in run_dirs:
        # Try to get config from Redis first
        config_json = r.get(f"benchmark:{run_id}:config")
        
        if config_json:
            # Use Redis config
            config = json.loads(config_json)
        else:
            # Try to load config from file
            config_path = os.path.join(runs_dir, run_id, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Create minimal config with defaults
                config = {
                    "dataset_name": "unknown",
                    "timestamp": os.path.getctime(os.path.join(runs_dir, run_id))
                }
        
        # Get task counts from Redis or estimate from directory structure
        total_tasks = int(r.get(f"benchmark:{run_id}:total_tasks") or 0)
        completed_tasks = int(r.get(f"benchmark:{run_id}:completed_tasks") or 0)
        
        # If Redis doesn't have the counts, estimate from directory structure
        if total_tasks == 0 or completed_tasks == 0:
            # Count scoring results directories as completed tasks
            scoring_results_dir = os.path.join(runs_dir, run_id, "scoring_results")
            if os.path.exists(scoring_results_dir):
                completed_docs = len([d for d in os.listdir(scoring_results_dir) 
                                     if os.path.isdir(os.path.join(scoring_results_dir, d))])
                completed_tasks = completed_docs
                
                # Estimate total tasks from dataset info if available
                dataset_info_path = os.path.join(runs_dir, run_id, "dataset_info.json")
                if os.path.exists(dataset_info_path):
                    with open(dataset_info_path, 'r') as f:
                        dataset_info = json.load(f)
                        total_tasks = len(dataset_info.get("doc_ids", []))
                else:
                    # Default to completed + 10% as estimate
                    total_tasks = max(completed_tasks, int(completed_tasks * 1.1))
        
        timestamp = config.get("timestamp", 0)
        runs.append({
            "run_id": run_id,
            "dataset_name": config.get("dataset_name", "unknown"),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "progress": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Sort by timestamp descending (newest first)
    runs.sort(key=lambda x: x["timestamp"], reverse=True)
    return runs

def load_results_jsonl(run_id):
    """Load results from the results.jsonl file for a specific run"""
    results_path = os.path.join("runs", run_id, "scoring_results", "results.jsonl")
    
    if not os.path.exists(results_path):
        return pd.DataFrame()
    
    results = []
    with open(results_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError:
                    continue
    
    return pd.DataFrame(results)

def display_pdf(pdf_file):
    """Display a PDF file in Streamlit."""
    # If pdf_file is a path, open it; if it's already bytes, use it directly
    if isinstance(pdf_file, str):
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    else:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def load_document(doc_id, dataset_name):
    """Load a document by its ID from the dataset directory."""
    # Try in the dataset-specific location
    dataset_pdf_path = os.path.join("data", dataset_name, "pdfs", f"{doc_id}.pdf")
    
    if os.path.exists(dataset_pdf_path):
        with open(dataset_pdf_path, "rb") as f:
            return io.BytesIO(f.read())
    
    # Fallback to old path for backward compatibility
    legacy_pdf_path = os.path.join("pdfs", f"{doc_id}.pdf")
    if os.path.exists(legacy_pdf_path):
        with open(legacy_pdf_path, "rb") as f:
            return io.BytesIO(f.read())
    
    return None

def load_rendered_html(doc_id, processor):
    """Load rendered HTML content for a document."""
    # Check for HTML output in the outputs directory
    html_path = os.path.join("outputs", f"{doc_id}_{processor}.html")
    
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    return None

def create_metrics_chart(results_df):
    """Create a grouped bar chart showing processor and model performance."""
    if results_df.empty:
        return None
    
    # Group by processor and model
    data = []
    
    for (processor, model_name), group in results_df.groupby(['processor', 'model_name']):
        accuracy = group['score'].mean()
        count = len(group)
        
        data.append({
            'Processor': processor,
            'Model': model_name,
            'Accuracy': accuracy,
            'Count': count
        })
    
    if not data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create the figure
    fig = go.Figure()
    
    # Add a trace for each model
    colors = px.colors.qualitative.Plotly
    model_colors = {}
    
    for i, model in enumerate(df['Model'].unique()):
        model_df = df[df['Model'] == model]
        model_colors[model] = colors[i % len(colors)]
        
        fig.add_trace(go.Bar(
            x=model_df['Processor'],
            y=model_df['Accuracy'],
            name=model,
            text=model_df['Accuracy'].apply(lambda x: f"{x:.1%}"),
            textposition='outside',
            marker_color=model_colors[model],
            hovertemplate='<b>%{x}</b><br>Model: ' + model + 
                          '<br>Accuracy: %{y:.1%}<br>Count: %{customdata}',
            customdata=model_df['Count']
        ))
    
    # Customize layout
    fig.update_layout(
        barmode='group',
        height=600,
        yaxis_range=[0, 1.1],
        yaxis_tickformat='.0%',
        xaxis_title="Processor",
        yaxis_title="Accuracy",
        plot_bgcolor='white',
        title=dict(
            text='Document QA Performance by Processor and Model',
            font=dict(size=20),
            x=0.5
        ),
        legend=dict(
            title="Model",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=80, b=60)
    )
    
    # Add gridlines
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor="rgba(0,0,0,0.1)"
    )
    
    # Add sample count annotation
    fig.add_annotation(
        text=f"Total samples: {len(results_df)}",
        xref="paper", yref="paper",
        x=0.5, y=1.06,
        showarrow=False,
        font=dict(size=14, color='#666')
    )
    
    return fig

def show_dashboard():
    st.set_page_config(page_title="PDF Processing Benchmark", layout="wide", page_icon="📊")
    
    st.title("PDF Processing Benchmark Dashboard")
    
    # Get available runs
    runs = get_runs()
    
    if not runs:
        st.warning("No benchmark runs found. Start a benchmark run and refresh this page.")
        return
    
    # Format run options for selectbox
    run_options = {f"{run['run_id']} - {run['dataset_name']} ({run['datetime']})": run['run_id'] for run in runs}
    selected_run = st.selectbox("Select benchmark run:", list(run_options.keys()))
    
    # Get selected run ID and data
    selected_run_id = run_options[selected_run]
    selected_run_data = next((run for run in runs if run['run_id'] == selected_run_id), None)
    
    if not selected_run_data:
        st.error("Selected run data not found.")
        return
    
    # Show run information
    st.subheader("Run Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset", selected_run_data['dataset_name'])
    with col2:
        st.metric("Documents Processed", f"{selected_run_data['completed_tasks']} / {selected_run_data['total_tasks']}")
    with col3:
        st.metric("Progress", f"{selected_run_data['progress']:.1f}%")
    
    # Create progress bar
    st.progress(selected_run_data['progress'] / 100)
    
    # Load results from results.jsonl
    results_df = load_results_jsonl(selected_run_id)
    
    if results_df.empty:
        st.warning("No results found for this run.")
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Performance Summary", "Document Viewer"])
    
    with tab1:
        # Create metrics chart
        metrics_chart = create_metrics_chart(results_df)
        if metrics_chart:
            st.plotly_chart(metrics_chart, use_container_width=True)
        
        # Show summary statistics
        st.subheader("Summary Statistics")
        
        # Count documents and questions
        num_docs = results_df["doc_id"].nunique()
        num_questions = results_df["question_id"].nunique()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", num_docs)
        with col2:
            st.metric("Questions", num_questions)
        
        # Show processor and model coverage
        st.subheader("Model Performance")
        
        # Group by processor and model
        model_stats = results_df.groupby(['processor', 'model_name']).agg({
            'score': ['mean', 'count']
        }).reset_index()
        
        # Flatten the multi-index columns
        model_stats.columns = ['Processor', 'Model', 'Accuracy', 'Questions Answered']
        
        # Format accuracy as percentage
        model_stats['Accuracy'] = model_stats['Accuracy'].apply(lambda x: f"{x:.1%}")
        
        # Display the stats
        st.dataframe(model_stats, use_container_width=True)
    
    with tab2:
        # Get unique documents
        doc_ids = results_df["doc_id"].unique()
        
        # Select a document
        selected_doc = st.selectbox("Select a document:", doc_ids)
        
        # Get questions for this document
        doc_questions = results_df[results_df["doc_id"] == selected_doc]["question"].unique()
        
        # Select a question
        selected_question = st.selectbox("Select a question:", doc_questions)
        
        # Get results for this document and question
        question_results = results_df[
            (results_df["doc_id"] == selected_doc) & 
            (results_df["question"] == selected_question)
        ]
        
        # Display the document and answers side by side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Document")
            
            # Load and display the PDF
            pdf_data = load_document(selected_doc, selected_run_data['dataset_name'])
            if pdf_data:
                display_pdf(pdf_data)
            else:
                st.error(f"PDF document not found for {selected_doc}")
        
        with col2:
            st.subheader("Question & Answers")
            
            # Display the question
            st.markdown(f"**Question:** {selected_question}")
            
            # Display ground truth answers
            ground_truths = question_results["ground_truths"].iloc[0]
            st.markdown("**Ground Truth Answers:**")
            if isinstance(ground_truths, list):
                for truth in ground_truths:
                    st.markdown(f"- {truth}")
            
            # Display model answers
            st.markdown("**Model Answers:**")
            
            # Create tabs for each processor
            processors = question_results["processor"].unique().tolist()
            if processors:  # Check if processors list is not empty
                processor_tabs = st.tabs(processors)
                
                for i, processor in enumerate(processors):
                    with processor_tabs[i]:
                        # Get results for this processor
                        processor_results = question_results[question_results["processor"] == processor]
                        
                        # Display each model's answer
                        for _, row in processor_results.iterrows():
                            model_name = row["model_name"]
                            answer = row["answer"]
                            score = row["score"]
                            
                            st.markdown(f"**Model:** {model_name}")
                            st.markdown(f"**Answer:** {answer}")
                            st.markdown(f"**Score:** {score * 100:.0f}%")
                            
                            # Try to load rendered HTML for this document and processor
                            html_content = load_rendered_html(selected_doc, processor)
                            if html_content:
                                with st.expander("View Rendered HTML"):
                                    st.markdown(html_content, unsafe_allow_html=True)
                            
                            st.markdown("---")
            else:
                st.warning("No processor data available for this question.")

if __name__ == "__main__":
    show_dashboard()