import numpy as np
import torch
from torch import nn
from .tokenization_bros import BrosTokenizer

def _init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        
class WordnnEmbedding(nn.Module):
    """Generate chargrid embedding feature map.
    """
    def __init__(self,
                 vocab_size=30552,
                 hidden_size=768,
                 embedding_dim=64,
                 bros_embedding_path="/bros-base-uncased/",
                 use_pretrain_weight=True,
                 use_UNK_text=False):
        """
        Args：
            vocab_size (int): size of vocabulary.
            embedding_dim (int): dim of input features
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding_proj = nn.Linear(hidden_size, embedding_dim, bias=False)
        # self.tokenizer = BrosTokenizer.from_pretrained(bros_embedding_path)
        self.use_pretrain_weight = use_pretrain_weight
        self.use_UNK_text = use_UNK_text
        
        self.init_weights(bros_embedding_path)
        self.apply(_init_weights)

    def init_weights(self, bros_embedding_path):
        if self.use_pretrain_weight:
            state_dict = torch.load(bros_embedding_path + "pytorch_model.bin", map_location='cpu')
            if 'bert' in bros_embedding_path:
                word_embs = state_dict["bert.embeddings.word_embeddings.weight"]
            elif 'bros' in bros_embedding_path:
                word_embs = state_dict["embeddings.word_embeddings.weight"]
            elif 'layoutlm' in bros_embedding_path:
                word_embs = state_dict["layoutlm.embeddings.word_embeddings.weight"]
            else:
                print("Wrong bros_embedding_path!")
            self.embedding = nn.Embedding.from_pretrained(word_embs)
            print("use_pretrain_weight: load model from:", bros_embedding_path)
        
    def forward(self, img, batched_inputs, stride = 1):
        """ Forward computation
        Args:
            img (Tensor): in shape of [B x 3 x H x W]
            batched_inputs (list[dict]): 
        Returns:
            Tensor: in shape of [B x N x L x D], where D is the embedding_dim.
        """
        device = img.device
        batch_b, _, batch_h, batch_w = img.size()

        chargrid_map = torch.zeros((batch_b, batch_h // stride, batch_w // stride ), dtype=torch.int64).to(device)
        
        for iter_b in range(batch_b):
            per_input_ids = batched_inputs[iter_b]["input_ids"]   
            per_input_bbox = batched_inputs[iter_b]["bbox"]
            
            short_length_w = min(len(per_input_ids), len(per_input_bbox)) 
            
            if short_length_w > 0 : 
                for word_idx in range(short_length_w): 
                    per_id = per_input_ids[word_idx]
                    
                    bbox = per_input_bbox[word_idx] / stride
                    w_start, h_start, w_end, h_end = bbox.round().astype(int).tolist()
                            
                    if self.use_UNK_text:
                        chargrid_map[iter_b, h_start:h_end, w_start: w_end] = 100
                    else:
                        chargrid_map[iter_b, h_start:h_end, w_start: w_end] = per_id

        chargrid_map = self.embedding(chargrid_map)
        chargrid_map = self.embedding_proj(chargrid_map)
        
        return chargrid_map.permute(0, 3, 1, 2).contiguous()
        
    