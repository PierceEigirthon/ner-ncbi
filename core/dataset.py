""" Dataset class which prepares inputs for the model """
import torch
from torch.utils.data import Dataset 

from core.config import config

class NCBIDataset(Dataset):
    
    def _truncate_to_max_len(self, x:list) -> list: 
        # need to adjust for addition of [CLS] and [SEP] tokens, so we 
        # truncate wit that in mind
        return x[:self.max_len - 2]
    
    def _pad_seq(self, x:list) -> list:
        # pads input to maximum length
        return x + [0 for _ in range(self.max_len - len(x))]  
    
    def _pt_ll(gself, x:list) -> torch.tensor:
        return torch.tensor(x, dtype=torch.long)
    
    def __init__(self, texts:list, tags:list, loss_plus:bool=True) -> None:
        assert len(texts) == len(tags)
        self.loss_plus = loss_plus
        self.texts = texts
        self.tags = tags 
        self.tok = config["tokenizer"]["TOKENIZER"]
        self.max_len = config["tokenizer"]["MAX_LEN"]
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]
        
        ids, targets = [], [] 
        for idx, word in enumerate(text): 
            # we will add [CLS] and [SEP] once we convet whole sentence
            word_pieces = self.tok.encode(word, add_special_tokens=False)  
            ids.extend(word_pieces)
            # for each word piece asign same tag as original word
            targets.extend([tags[idx] for _ in range(len(word_pieces))])  
                            
                            
        # add [CLS] and [SEP] tokens to sentence
        # this is what bert/distilbert expects included in the input
        ids = [self.tok.cls_token_id] + self._truncate_to_max_len(ids) + [self.tok.sep_token_id] 
        # assing something to cls and sep tokens
        # pretty sure we can assign what ever we want here
        # however, we will need to make sure not to optimizer w.r.t to these outputs
        # so we will have to tell the loss function to ignore these 
        # NOTE: updated, now if loss plus is selected, cls and sep will default to -100 targets, which nn.CrossEntropy ignores
        _s_tag = 0
        if self.loss_plus: _s_tag = -100
        targets = [_s_tag] + self._truncate_to_max_len(targets) + [_s_tag]
         
        # mask, which tokens we pay attention to
        mask = [1 for _ in range(len(ids))] 
        # assign all token_type_ids to 1 sentence, needed for final embedding 
        # NOTE: distilbert does not use sentence embeddings,
        # NOTE: but if you want to use, say original bert you should include these
        # token_type_ids = [0 for _ in range(len(ids))] 
        
        return {
            "input_ids" : self._pt_ll(self._pad_seq(ids)), 
            "attention_mask" : self._pt_ll(self._pad_seq(mask)), 
           #"token_type_ids" : self._pt_ll(self._pad_seq(token_type_ids)), 
            "targets" : self._pt_ll(self._pad_seq(targets)) 
        }
