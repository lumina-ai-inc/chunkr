# Text Extraction Benchmark

Goal: Test the text extraction/OCR capabilities of different models and providers.

Plan: 
1. Find a benchmark dataset with text in correct reading order
2. Take same pdfs and extract text with different models and providers
3. Do string similarity to see how close the models are to the original text

It will help us test both text extraction and reading order.

## Benchmark Datasets with Reading Order Ground Truth

1. **DocBank Reading Order**
- Contains full text content with reading order
- Focus on academic papers and articles
- https://github.com/doc-analysis/DocBank/tree/master/reading_order

2. **IAM Handwriting Database**
- Contains transcribed text in correct reading order
- Mostly for handwritten documents but includes some printed text
- https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

3. **FUNSD (Form Understanding Dataset)**
- Contains form documents with text in reading order
- Includes both the original PDFs and ground truth text
- https://guillaumejaume.github.io/FUNSD/

