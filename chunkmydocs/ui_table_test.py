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

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Get available prompts and models from the files
        current_image = processed_images[st.session_state.current_image_idx]
        image_dir = output_dir / current_image.name
        html_files = list(image_dir.glob("*.html"))
        
        # Split filename into model, prompt and check for cot
        model_prompt_info = []
        for f in html_files:
            parts = f.stem.split('_')
            model = parts[0]
            prompt = parts[1]
            cot = "cot1" if f.stem.endswith("_cot1") else "cot0"
            model_prompt_info.append((model, prompt, cot))
        
        unique_models = sorted(set(model for model, _, _ in model_prompt_info))
        unique_prompts = sorted(set(prompt for _, prompt, _ in model_prompt_info))
        unique_cots = sorted(set(cot for _, _, cot in model_prompt_info))
        
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=unique_models,
            default=unique_models[:2]  # Default select first two
        )
        
        selected_prompt = st.selectbox(
            "Select Prompt",
            options=unique_prompts
        )

        selected_cot = st.selectbox(
            "Select Chain of Thought",
            options=unique_cots
        )

    # Image navigation
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

    # Display original image
    st.image(str(image_dir / "original.jpg"))
    
    # Display results for each selected model in a list format
    if selected_models:
        st.subheader("Model Results")
        for model in selected_models:
            st.write(f"**Model: {model}**")
            html_file = image_dir / f"{model}_{selected_prompt}.html"
            
            if html_file.exists():
                with open(html_file) as f:
                    html_content = f.read()
                    
                    # Extract just the table content from the HTML
                    table_content = html_content.strip()
                    
                    # Add toggle for raw HTML view
                    show_raw = st.checkbox("Show raw HTML", value=False, key=f"show_raw_{model}")
                    
                    if show_raw:
                        # Show raw HTML
                        st.code(table_content, language="html")
                    else:
                        # Remove HTML tags if present
                        if table_content.lower().startswith('<html>') and table_content.lower().endswith('</html>'):
                            table_content = table_content[6:-7].strip()
                            
                        # Remove content outside ```html and ```
                        if "```html" in table_content:
                            # Keep only the content between ```html and the next ```
                            table_content = table_content[table_content.find("```html"):]
                            if "```" in table_content[7:]:  # Look for closing ``` after "```html"
                                table_content = table_content[:table_content.find("```", 7)]
                            table_content = table_content.replace("```html", "").strip()
                        # Render table with HTML formatting and styling
                        styled_content = f"""
                        <div style="width: 100%; overflow-x: auto;">
                            {table_content}
                        </div>
                        """
                        st.markdown(styled_content, unsafe_allow_html=True)
                        
                    # Display metrics if available
                    csv_file = image_dir / f"{model}_{selected_prompt}.csv"
                    if csv_file.exists():
                        with open(csv_file) as f:
                            metrics = f.readlines()[1].strip().split(',')
                        st.caption(f"Processing time: {metrics[3]}s")
            else:
                st.error(f"No results found for this model and prompt combination")
            
            st.divider() # Add visual separation between models

if __name__ == "__main__":
    streamlit_ui()