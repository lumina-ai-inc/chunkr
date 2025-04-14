the objective is to train a table model to use with sophris ai's datasheets.

want to find a distillation of azure that gets >85% similarity for tables in the dataset.

Iteration 1:

- (using segmentation cpu) get azure outputs from chunk in chunks -> segment in chunk.segment where segment_type is table and save that HTML. 
- format messages with image crops from chunkr outputs
- set up basic unsloth script to feintune qwen 7b on A100 VM. 

bonus:

- collect iterations of the image descriptions and prompts
- add processor to collect azure SDK outputs directly


progress:

- reads datasheets to s3 and saves them.
- processes them through chunkr for tables and saves relevant info in 
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