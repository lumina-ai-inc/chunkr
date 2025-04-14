<<<<<<< HEAD
the objective is to train a table model to use with sophris ai's datasheets.

want to find a distillation of azure that gets >85% similarity for tables in the dataset.

Iteration 1:

- (using segmentation cpu) get azure outputs from chunk in chunks -> segment in chunk.segment where segment_type is table and save that HTML. 
- format messages with image crops from chunkr outputs
- set up basic unsloth script to feintune qwen 7b on A100 VM. 

bonus:

- collect iterations of the image descriptions and prompts
- add processor to collect azure SDK outputs directly
=======
the goal of this Repo is to generate HTML + image pairs for table OCR

## Initial approach

## Collecting Tables

- Use Chunkr YOLO segmenter API to segment out many tables 
- Use Azure processor to collect raw data for the tables (html + bboxes)
- write data formatters for VLM + DETR training run 
>>>>>>> c3f54a4c (setup)
