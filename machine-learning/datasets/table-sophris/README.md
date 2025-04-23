## train a table model to use with sophris ai's datasheets.

want to find a distillation of azure that gets >85% similarity for tables in the dataset.

### Iteration 1:

- (using segmentation cpu) get azure outputs from chunk in chunks -> segment in chunk.segment where segment_type is table and save that HTML. 
- format messages with image crops from chunkr outputs
- set up basic unsloth script to feintune qwen 7b on A100 VM. 

#### bonus:

- collect iterations of the image descriptions and prompts
- add processor to collect azure SDK outputs directly


#### progress:

- reads datasheets to s3 and saves them.
- processes them through chunkr for tables and saves relevant info in the format:
training/
├─ sophris/
│  ├─ pdfs/
│  │  ├─ texas.pdf
│  ├─ chunkr_outputs/
│  │  ├─ texas.json
│  ├─ tables/
│  │  ├─ texas_{table_id}.json
│  ├─ table_mkd/
│  │  ├─ texas_{table_id}.md
│  ├─ table_html/
│  │  ├─ texas_{table_id}.html
- to run see ./gen_data.sh


S3 BUCKET: chunkr-datasets

#### issues

- storage.py is in 2 places cause of dumb python exports
- take dataset name and bucket name from env for both datasheet upload and data gen
- add format data and training scripts - good to go for first run tmr lfg

datasets:
- tables-vlm-azure-distill-v1
- sophris-datasheet-table-extraction-azure-distill-v1
