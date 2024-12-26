import dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from transformers import BertTokenizer
import os
from s2s_ft.modeling_decoding import LayoutlmForSeq2SeqDecoder, BertConfig
import json
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.utils import convert_src_layout_inputs_to_tokens

dotenv.load_dotenv(override=True)
app = FastAPI()

# Model configuration
MODEL_PATH = os.getenv('MODEL_PATH') or "models/layoutreader-base-readingbank"
CONFIG_PATH = os.getenv('CONFIG_PATH') or os.path.join(MODEL_PATH, "config.json")
TOKENIZER_NAME = os.getenv('TOKENIZER_NAME') or "bert-base-uncased"
MAX_SEQ_LENGTH = int(os.getenv('MAX_SEQ_LENGTH') or 1024)
MAX_TGT_LENGTH = int(os.getenv('MAX_TGT_LENGTH') or 511)
USE_FP16 = os.getenv('USE_FP16', '').lower() in ('true', '1', 't')
USE_MULTI_GPU = os.getenv('USE_MULTI_GPU', '').lower() in ('true', '1', 't')
LAYOUTLM_ONLY_LAYOUT = os.getenv('LAYOUTLM_ONLY_LAYOUT', '').lower() in ('true', '1', 't')  

class LayoutInput(BaseModel):
    text: str  
    bboxes: List[List[int]] 

class PredictionResponse(BaseModel):
    reading_order: List[int]

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if USE_MULTI_GPU else 1
    
    tokenizer = BertTokenizer.from_pretrained(
        TOKENIZER_NAME,
        do_lower_case=True,
        max_len=MAX_SEQ_LENGTH
    )
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    config.layoutlm_only_layout_flag = LAYOUTLM_ONLY_LAYOUT 
    
    mask_word_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0]
    eos_word_ids = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
    sos_word_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
    
    model = LayoutlmForSeq2SeqDecoder.from_pretrained(
        MODEL_PATH,
        config=config,
        mask_word_id=mask_word_id,
        search_beam_size=1,
        length_penalty=0,
        eos_id=eos_word_ids,
        sos_id=sos_word_id,
        forbid_duplicate_ngrams=False,
        forbid_ignore_set=None,
        ngram_size=3,
        min_len=1,
        mode="s2s",
        max_position_embeddings=MAX_SEQ_LENGTH,
        pos_shift=False,
    )
    
    preprocess = seq2seq_loader.Preprocess4Seq2seqDecoder(
        list(tokenizer.vocab.keys()),
        tokenizer.convert_tokens_to_ids,
        MAX_SEQ_LENGTH,
        max_tgt_length=MAX_TGT_LENGTH,
        pos_shift=False,
        source_type_id=config.source_type_id,
        target_type_id=config.target_type_id,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.pad_token,
        layout_flag=True
    )
    
    if USE_FP16:
        model.half()
    
    model.to(device)
    
    if USE_MULTI_GPU and n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model.eval()
    
    return model, tokenizer, device, config, preprocess

model, tokenizer, device, config, preprocess = load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict_reading_order(input_data: LayoutInput):
    try:
        words = input_data.text.split()
        if len(words) != len(input_data.bboxes):
            raise HTTPException(
                status_code=400,
                detail=f"Number of words ({len(words)}) doesn't match number of bounding boxes ({len(input_data.bboxes)})"
            )

        max_src_length = MAX_SEQ_LENGTH - 2 - MAX_TGT_LENGTH
        
        source_ids = []
        index_split = {}
        new_token_index = 1  
        
        for i, (word, bbox) in enumerate(zip(words, input_data.bboxes)):
            if not (bbox[2] >= bbox[0] and bbox[3] >= bbox[1]):
                continue
                
            tokens = tokenizer.tokenize(word)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            new_token_ids = []
            for token in token_ids:
                source_ids.append([token] + bbox)  
                new_token_ids.append(new_token_index)
                new_token_index += 1
            index_split[i] = new_token_ids
        
        instance = {
            "source_ids": source_ids,  
            "target_ids": [], 
            "target_index": [],  
            "bleu": 0 
        }
        
        instances_with_tokens = convert_src_layout_inputs_to_tokens(
            [instance], 
            tokenizer.convert_ids_to_tokens, 
            max_src_length,
            layout_flag=True  
        )
        
        instances_with_tokens = sorted(list(enumerate(instances_with_tokens)), key=lambda x: -len(x[1]))
        buf = [x[1] for x in instances_with_tokens]
        
        processed = preprocess(buf[0])
        
        batch = seq2seq_loader.batch_list_to_batch_tensors([processed])
        batch = [t.to(device) if t is not None else None for t in batch]
        
        with torch.no_grad():
            input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
            traces = model(
                input_ids, 
                token_type_ids, 
                position_ids, 
                input_mask, 
                task_idx=task_idx, 
                mask_qkv=mask_qkv
            )
            
            if isinstance(traces, dict):
                output_ids = traces['pred_seq'].tolist()[0]
            else:
                output_ids = traces.tolist()[0]
                
            reading_order = [idx - 1 for idx in output_ids if idx > 0]  
            reading_order = reading_order[:len(input_data.bboxes)]  
            
            return PredictionResponse(
                reading_order=reading_order
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 