# Chunk My Docs

We're Lumina. We've built a search engine that's 5x more relevant than Google Scholar. You can check us out at https://lumina.sh. We achieved this by bringing state of the art search technology (the best in dense and sparse vector embeddings) to academic research. 

While search is one problem, sourcing high quality data is another. We needed to process millions of PDFs in house to build Lumina, and we found out that existing solutions to extract structured information from PDFs were too slow and too expensive ($$ per page). 

Chunk my docs provides a self-hostable solution that leverages state-of-the-art (SOTA) vision models for segment extraction and OCR, unifying the output through a Rust Actix server. This solution has models that accomodate for both GPU and CPU environments. Try the UI on https://chunkr.ai!
