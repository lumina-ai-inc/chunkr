# Chunk My Docs

We're Lumina. We've built a search engine that's 5x more relevant than Google Scholar. You can check us out at https://lumina.sh. We achieved this by bringing state of the art search technology (the best in dense and sparse vector embeddings) to academic research. 

While search is one problem, sourcing high quality data is another. We needed to process millions of PDFs in house to build Lumina, and we found out that existing solutions to extract structured information from PDFs were too slow and too expensive ($$ per page). 

Chunk my docs provides a self-hostable solution that leverages state-of-the-art (SOTA) vision models for segment extraction and OCR, unifying the output through a Rust Actix server. This setup allows you to process PDFs and extract segments at an impressive speed of approximately 5 pages per second on a single NVIDIA L4 instance, offering a cost-effective and scalable solution for high-accuracy bounding box segment extraction and OCR. This solution has models that accomodate for both GPU and CPU environments. Try the UI on https://chunkr.ai!


## Licensing

This project is dual-licensed:

1. [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)
2. Commercial License

To use Chunkr privately without complying to the AGPL-3.0 liscence terms you can [contact us](mailto:ishaan@lumina.sh) or visit our [website](https://chunkr.ai).
