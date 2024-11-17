import streamlit as st
from pathlib import Path

def streamlit_ui():
    st.title("Table OCR Results Viewer")
    
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Get all processed images
    processed_images = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not processed_images:
        st.error("No processed images found in output directory")
        return

    # Initialize session state for current image index if not exists
    if 'current_image_idx' not in st.session_state:
        st.session_state.current_image_idx = 0
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = None
    if 'show_raw' not in st.session_state:
        st.session_state.show_raw = False
    if 'file_type' not in st.session_state:
        st.session_state.file_type = "html"

    # Get current image and available models/prompts
    current_image = processed_images[st.session_state.current_image_idx]
    image_dir = output_dir / current_image.name
    html_files = list(image_dir.glob("*.html"))
    md_files = list(image_dir.glob("*.md"))
    
    # Get model info from both HTML and MD files
    model_prompt_info = []
    for f in html_files + md_files:
        parts = f.stem.split('_')
        model = parts[0]
        prompt = parts[1]
        model_prompt_info.append((model, prompt))
    
    unique_models = sorted(set(model for model, _ in model_prompt_info))
    unique_prompts = sorted(set(prompt for _, prompt in model_prompt_info))

    # Image navigation and model selection
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("←") and st.session_state.current_image_idx > 0:
            st.session_state.current_image_idx -= 1
            st.rerun()
    with col2:
        st.write(f"### Image {st.session_state.current_image_idx + 1} of {len(processed_images)}")
        st.write(f"*{current_image.name}*")
    with col3:
        if st.button("→") and st.session_state.current_image_idx < len(processed_images) - 1:
            st.session_state.current_image_idx += 1
            st.rerun()

    # File type toggle
    st.write("### Select Output Type")
    file_type_cols = st.columns(2)
    with file_type_cols[0]:
        if st.button("HTML", type="primary" if st.session_state.file_type == "html" else "secondary"):
            st.session_state.file_type = "html"
            st.rerun()
    with file_type_cols[1]:
        if st.button("Markdown", type="primary" if st.session_state.file_type == "md" else "secondary"):
            st.session_state.file_type = "md"
            st.rerun()

    # Model selection buttons in a row
    st.write("### Select Model")
    cols = st.columns(len(unique_models))
    for idx, model in enumerate(unique_models):
        with cols[idx]:
            if st.button(model, type="primary" if st.session_state.selected_model == model else "secondary"):
                st.session_state.selected_model = model
                st.rerun()

    # Prompt selection buttons in a row
    st.write("### Select Prompt")
    cols = st.columns(len(unique_prompts))
    for idx, prompt in enumerate(unique_prompts):
        with cols[idx]:
            if st.button(prompt, type="primary" if st.session_state.selected_prompt == prompt else "secondary"):
                st.session_state.selected_prompt = prompt
                st.rerun()

    # Display original and results side by side
    container_style = """
        <style>
            .side-by-side-container {
                display: flex;
                width: 100vw;
                gap: 124px;
                justify-content: center;
            }
            .side-by-side-container > div {
                flex: 0 0 calc(50% - 62px);
            }
        </style>
    """
    st.markdown(container_style, unsafe_allow_html=True)
    
    st.markdown('<div class="side-by-side-container">', unsafe_allow_html=True)
    
    # First column
    st.markdown('<div>', unsafe_allow_html=True)
    st.subheader("Original Image")
    st.image(str(image_dir / "original.jpg"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Second column
    st.markdown('<div>', unsafe_allow_html=True)
    st.subheader("Model Output")
    if st.session_state.selected_model and st.session_state.selected_prompt:
        file_ext = "html" if st.session_state.file_type == "html" else "md"
        result_file = image_dir / f"{st.session_state.selected_model}_{st.session_state.selected_prompt}.{file_ext}"
        
        if result_file.exists():
            with open(result_file) as f:
                content = f.read()
                table_content = content.strip()
                
                # Toggle raw view
                st.session_state.show_raw = st.checkbox(f"Show raw {file_ext.upper()}", value=st.session_state.show_raw)
                
                if st.session_state.show_raw:
                    st.code(table_content, language=file_ext)
                else:
                    if file_ext == "html":
                        # Clean up HTML content
                        if table_content.lower().startswith('<html>') and table_content.lower().endswith('</html>'):
                            table_content = table_content[6:-7].strip()
                            
                        if "```html" in table_content:
                            table_content = table_content[table_content.find("```html"):]
                            if "```" in table_content[7:]:
                                table_content = table_content[:table_content.find("```", 7)]
                            table_content = table_content.replace("```html", "").strip()
                            
                        styled_content = f"""
                        <div style="width: 100vw; overflow-x: auto;">
                            {table_content}
                        </div>
                        """
                        st.markdown(styled_content, unsafe_allow_html=True)
                    else:
                        # Clean up Markdown content
                        if "```markdown" in table_content:
                            table_content = table_content[table_content.find("```markdown"):]
                            if "```" in table_content[11:]:
                                table_content = table_content[:table_content.find("```", 11)]
                            table_content = table_content.replace("```markdown", "").strip()
                        
                        st.markdown(table_content)
                    
                # Display metrics
                csv_file = image_dir / f"{st.session_state.selected_model}_{st.session_state.selected_prompt}.csv"
                if csv_file.exists():
                    with open(csv_file) as f:
                        metrics = f.readlines()[1].strip().split(',')
                        if len(metrics) > 3:  # Check if metrics has enough elements
                            st.caption(f"Processing time: {metrics[3]}")
        else:
            st.error(f"No {file_ext.upper()} results found for {st.session_state.selected_model} with {st.session_state.selected_prompt}")
    else:
        st.info("Select a model and prompt to view results")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    streamlit_ui()