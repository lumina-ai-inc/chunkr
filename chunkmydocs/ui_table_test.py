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

    # Get current image and available models/prompts
    current_image = processed_images[st.session_state.current_image_idx]
    image_dir = output_dir / current_image.name
    html_files = list(image_dir.glob("*.html"))
    
    # Get model info
    model_prompt_info = []
    for f in html_files:
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
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(str(image_dir / "original.jpg"))
    
    with col2:
        st.subheader("Model Output")
        if st.session_state.selected_model and st.session_state.selected_prompt:
            html_file = image_dir / f"{st.session_state.selected_model}_{st.session_state.selected_prompt}.html"
            
            if html_file.exists():
                with open(html_file) as f:
                    html_content = f.read()
                    table_content = html_content.strip()
                    
                    # Toggle raw HTML view
                    st.session_state.show_raw = st.checkbox("Show raw HTML", value=st.session_state.show_raw)
                    
                    if st.session_state.show_raw:
                        st.code(table_content, language="html")
                    else:
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
                        
                    # Display metrics
                    csv_file = image_dir / f"{st.session_state.selected_model}_{st.session_state.selected_prompt}.csv"
                    if csv_file.exists():
                        with open(csv_file) as f:
                            metrics = f.readlines()[1].strip().split(',')
                            if len(metrics) > 3:  # Check if metrics has enough elements
                                st.caption(f"Processing time: {metrics[3]}")
            else:
                st.error(f"No results found for {st.session_state.selected_model} with {st.session_state.selected_prompt}")
        else:
            st.info("Select a model and prompt to view results")

if __name__ == "__main__":
    streamlit_ui()