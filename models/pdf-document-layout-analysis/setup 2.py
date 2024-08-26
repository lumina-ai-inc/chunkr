from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

PROJECT_NAME = "pdf-document-layout-analysis"

setup(
    name=PROJECT_NAME,
    packages=["pdf_tokens_type_trainer", "pdf_features", "pdf_token_type_labels", "fast_trainer"],
    package_dir={"": "src"},
    version="0.5",
    url="https://github.com/huridocs/pdf-document-layout-analysis",
    author="HURIDOCS",
    description="This tool is for PDF document layout analysis",
    install_requires=requirements,
    setup_requieres=requirements,
)
