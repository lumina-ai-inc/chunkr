from huggingface_hub import hf_hub_download

pdf_tokens_type_model = hf_hub_download(
    repo_id="HURIDOCS/pdf-segmentation",
    filename="pdf_tokens_type.model",
    revision="c71f833500707201db9f3649a6d2010d3ce9d4c9",
)

token_type_finding_config_path = hf_hub_download(
    repo_id="HURIDOCS/pdf-segmentation",
    filename="tag_type_finding_model_config.txt",
    revision="7d98776dd34acb2fe3a06495c82e64b9c84bdc16",
)
