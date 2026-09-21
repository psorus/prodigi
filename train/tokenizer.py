import json
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

MAXIMUM_INTEGER=100
MAX_SEQ_LEN=96

special_tokens={"<PAD>":0, "<CLS>":1, "<EOS>":2, "<NUM>":3, "<BOOL>":4, "<UNK>":5, "<FILLNUM>":6, "<FILLTOKEN>":7, "<FILLINT>":8, "<FILLBOOL>":9}

PAD_TOKEN=special_tokens["<PAD>"]
CLS_TOKEN=special_tokens["<CLS>"]
EOS_TOKEN=special_tokens["<EOS>"]
NUM_TOKEN=special_tokens["<NUM>"]
BOOL_TOKEN=special_tokens["<BOOL>"]
UNK_TOKEN=special_tokens["<UNK>"]
FILL_NUM=special_tokens["<FILLNUM>"]
FILL_TOKEN=special_tokens["<FILLTOKEN>"]
FILL_INT=special_tokens["<FILLINT>"]
FILL_BOOL=special_tokens["<FILLBOOL>"]

#setup: 5 special tokens (0-4), next special tokens (in the tuple, first elem TOKEN, then second one), then 0-maximum_integer tokens (*max_integer tokens)

def _all_tokens(desc):
    for a,b in desc:
        if a.upper()=="TOKEN":
            yield str(b)

class Vocabulary():
    def __init__(self,desc, maximum_integer=MAXIMUM_INTEGER, num_cls=1, minimum_integer=-1):
        self.dic={key.upper():val for key, val in special_tokens.items()}
        for i in range(1,num_cls):
            self.dic[f"<CLS{i}>"]=len(self.dic)
        self.token_count=len(self.dic)
        #print(_all_tokens(desc))
        meaning_tokens=sorted(list(set(_all_tokens(desc))))
        #print("meaning_tokens",meaning_tokens)
        #exit()
        for token in meaning_tokens:
            self.dic[token.upper()]=self.token_count
            self.token_count+=1
        for bool in [True, False]:
            self.dic["BOOL_"+str(bool).upper()]=self.token_count
            self.token_count+=1
        for i in range(minimum_integer,maximum_integer+1):
            self.dic["INT_"+str(i)]=self.token_count
            self.token_count+=1
        self.idic={integer:token for token, integer in self.dic.items()}

    def all_tokens(self):
        return self.idic
    def all_ints(self):
        return {integer:token for token, integer in self.dic.items() if token.startswith("INT_")}
    def int_to_index(self, integer):
        return self.dic.get("INT_"+str(integer), UNK_TOKEN)
    def bool_to_index(self, boolean):
        return self.dic.get("BOOL_"+str(boolean), UNK_TOKEN)
    def all_bools(self):
        return {integer:token for token, integer in self.dic.items() if token.startswith("BOOL_")}
    def translate(self, token):
        ret= self.dic.get(token.upper(), UNK_TOKEN)
        #if ret==UNK_TOKEN:
        #    print(f"Warning: token '{token}' not in vocabulary, translating to UNK")
        return ret

    def inverse(self, integer):
        integer=int(integer)
        #if integer not in self.idic:
        #    print(f"Warning: integer '{integer}' not in vocabulary, translating to UNK")
        return self.idic.get(integer, "<UNK>")
    def save_vocab(self, path):
        with open(path, "w") as f:
            json.dump(self.dic, f)
    @staticmethod
    def load_vocab(path):
        with open(path, "r") as f:
            dic=json.load(f)
        vocab=Vocabulary([])
        vocab.dic=dic
        vocab.idic={integer:token for token, integer in dic.items()}
        vocab.token_count=len(dic)
        return vocab


class Tokenizer():
    def __init__(self, vocab, max_seq_len=MAX_SEQ_LEN, num_cls=1):
        self.vocab=vocab
        self.max_seq_len=max_seq_len
        self.num_cls=num_cls

    def encode(self, desc):
        #desc contains a list of tuples (type, value). type is either TOKEN, INT, or FLOAT. Treat FLOATS special. Returns (type, value) again. Here INTs are converted to INT_? tokens
        tokens=[]
        tokens.append(("TOKEN", CLS_TOKEN, 0))
        for i in range(1, self.num_cls):
            tokens.append(("TOKEN", self.vocab.translate(f"<CLS{i}>"), 0))
        for t, v in desc:
            t=t.upper()
            if t=="TOKEN":
                v=v.upper()
                tokens.append(("TOKEN",self.vocab.translate(v), 0))
            elif t=="INT":
                if v>MAXIMUM_INTEGER:
                    tokens.append(("TOKEN", UNK_TOKEN, 0))
                else:
                    tokens.append(("TOKEN", self.vocab.translate("INT_"+str(v)), 0))
            elif t=="BOOL":
                tokens.append(("TOKEN", self.vocab.translate("BOOL_"+str(v)), 0))
            elif t=="FLOAT":
                tokens.append(("FLOAT", NUM_TOKEN, v))
            elif t=="FILL":
                v=v.upper()
                if v=="INT":
                    tokens.append(("TOKEN", FILL_INT, 0))
                elif v=="BOOL":
                    tokens.append(("TOKEN", FILL_BOOL, 0))
                elif v=="TOKEN":
                    tokens.append(("TOKEN", FILL_TOKEN, 0))
                elif v=="FLOAT" or v=="NUM":
                    tokens.append(("TOKEN", FILL_NUM, 0))
                else:
                    tokens.append(("TOKEN", UNK_TOKEN, 0))
        if len(tokens)>=self.max_seq_len-1:
            tokens=tokens[:self.max_seq_len-1]
        tokens.append(("TOKEN", EOS_TOKEN, 0))
        attention_mask = [1] * len(tokens) + [0] * (self.max_seq_len - len(tokens))
        while len(tokens)<self.max_seq_len:
            tokens.append(("TOKEN", PAD_TOKEN, 0))

        return {"tokens":tokens, "mask":attention_mask}

    def decode(self, token_ids, numerical_values):
        #returns a list of either (FLOAT, Value), (INT, Value), (TOKEN, TYPE)
        desc=[]
        for token_id, num_val in zip(token_ids, numerical_values):
            if isinstance(num_val, torch.Tensor):
                num_val=num_val.item()

            token_type=self.vocab.inverse(token_id)

            if token_type.startswith("INT_"):
                desc.append(("INT", int(token_type[4:])))
            elif token_type.startswith("BOOL_"):
                desc.append(("BOOL", token_type[5:]=="True"))
            elif token_type=="<NUM>":
                desc.append(("FLOAT", num_val))
            elif token_type=="<PAD>" or token_type=="<CLS>":
                continue
            elif token_type=="<EOS>":
                break
            else:
                desc.append(("TOKEN", token_type))
        return desc





def collate_fn(batch):
    """
    Takes a batch of outputs from your Tokenizer and formats them 
    into clean, parallel numerical tensors ready for the model.
    """
    batch_input_ids = []
    batch_templates = []
    batch_numerical_values = []
    batch_masks = []
    batch_attributes = []
    batch_data=[]
    
    for item in batch:
        # item is {"tokens": [...], "mask": [...]}
        tokens_list = item["tokens"]
        template_list = item["template"]
        mask_list = item["mask"]
        attr_list = item["attributes"] 
        
        # Separate the 3-element tuples into parallel tracking lists
        input_ids = [tup[1] for tup in tokens_list]
        numerical_values = [tup[2] for tup in tokens_list]
        template = [tup[1] for tup in template_list]
        
        batch_input_ids.append(input_ids)
        batch_templates.append(template)
        batch_numerical_values.append(numerical_values)
        batch_masks.append(mask_list)
        batch_attributes.append(attr_list)

        if "data" in item:
            batch_data.append(item["data"])
        
    ret= {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "template_ids": torch.tensor(batch_templates, dtype=torch.long),
        "numerical_values": torch.tensor(batch_numerical_values, dtype=torch.float).unsqueeze(-1), # [Batch, Seq_Len, 1]
        "mask": torch.tensor(batch_masks, dtype=torch.long),
        "attributes": batch_attributes
    }
    if len(batch_data)>0:
        ret["data"]=torch.tensor(np.array(batch_data), dtype=torch.float)
    return ret

class SequenceDataset(IterableDataset):
    def __init__(self, vocab, tokenizer, draw_one, vary_data_samples=False, min_data_samples=64, max_data_samples=1024, default_data_samples=1000):
        self.vocab=vocab
        self.tokenizer=tokenizer
        self.draw_one=draw_one
        self.vary_data_samples=vary_data_samples
        self.min_data_samples=min_data_samples
        self.max_data_samples=max_data_samples
        self.default_data_samples=default_data_samples
        self.choose_size()

    def choose_size(self):
        self.data_samples=self.default_data_samples
        if self.vary_data_samples:
            self.data_samples=int(np.exp(np.random.uniform(np.log(self.min_data_samples), np.log(self.max_data_samples))))
        #print("Choosing dataset size with vary_samples=", self.vary_data_samples, "resulting data_samples=", self.data_samples)


    def __iter__(self):
        while True:
            curr=self.draw_one(self.data_samples)
            if len(curr)==3:
                raw_sequence, template, attributes = curr
                data=None
            elif len(curr)==4:
                raw_sequence, template, attributes, data = curr
            else:
                raise ValueError("draw_one must return a tuple of length 3 or 4")
            tokenized_dict = self.tokenizer.encode(raw_sequence)
            template_dict = self.tokenizer.encode(template)
            #yield tokenized_seq, attributes
            curr= {
                "tokens": tokenized_dict["tokens"],
                "template": template_dict["tokens"],
                "mask": tokenized_dict["mask"],
                "attributes": attributes
            }
            if not data is None:
                curr["data"]=data
            yield curr

