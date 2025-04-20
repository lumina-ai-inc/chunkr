from PIL import Image

def chat_with_image(image_path: str, message: str):
    """Simple function to chat with the model about an image."""
    try:
        # Convert image path to PIL Image if it's not already
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
            
        # Run inference using the global model and processor
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": message}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            input_token_len = inputs['input_ids'].shape[1]
            generated_ids_trimmed = generated_ids[:, input_token_len:]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        return response.strip()
    except Exception as e:
        print(f"Chat error: {e}\n{traceback.format_exc()}")
        return f"Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 📊 Sophris Table VLM\nStatus: {status_message}")

    with gr.Tabs() as tabs:
        with gr.Tab("Chat", id=0):  # Make Chat the first tab
            with gr.Row():
                with gr.Column(scale=1):
                    chat_image_input = gr.Image(
                        label="Upload Image",
                        type="filepath",
                        sources=["upload", "clipboard"]  # Allow clipboard paste
                    )
                    chat_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask anything about the image...",
                        lines=3
                    )
                    chat_button = gr.Button("🚀 Send Message", variant="primary", size="lg")
                with gr.Column(scale=2):
                    chat_output = gr.Textbox(
                        label="Model Response",
                        lines=10,
                        show_copy_button=True  # Add copy button
                    )

        with gr.Tab("Table Extraction"):
            # ... existing table extraction interface ...

    # Add chat handler
    chat_button.click(
        fn=chat_with_image,
        inputs=[chat_image_input, chat_input],
        outputs=[chat_output]
    ) 