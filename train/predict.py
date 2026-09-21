import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("fork")



import numpy as np

from fast_mmd import fast_metrics
from gmm_prior import GMMPrior
from copula_prior import CopulaPrior
from scm_prior import SCMPrior, SCMError
from mixed_prior import MixedPrior

from tokenizer import Vocabulary, Tokenizer, SequenceDataset, collate_fn, NUM_TOKEN, CLS_TOKEN, EOS_TOKEN, FILL_NUM, FILL_TOKEN, FILL_INT, FILL_BOOL
from model import TransformerAutoencoder
from settransformer import DeepSetEncoder
from multitokensettransformer import MultiTokenDeepSetEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb

from config import Config
import sys
import os

import json
from safe_json import safe_json

from helper import dic_mean, dic_concat, dic_merge, time0, time1, totaltime, resettimes, alltimes
from boso import BMatchingError

heads={FILL_NUM:"float", FILL_TOKEN:"token", FILL_INT:"int", FILL_BOOL:"bool"}

def run(cfg):

    #set all seeds
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)

    if cfg.AUTO_START:
        if cfg.PRETRAINED=="AUTO":
            cfg.PRETRAINED=f"models/{cfg.NAME}/last_base.pt"
        if cfg.FN=="AUTO":
            cfg.FN=f"models/{cfg.NAME}/last.pt"
        if cfg.EPOCH_PATH=="AUTO":
            cfg.EPOCH_PATH=f"models/{cfg.NAME}/epoch"


    if cfg.PRIOR=="GMM":
        prior=GMMPrior(cfg)
    elif cfg.PRIOR=="SCM":
        prior=SCMPrior(cfg)
    elif cfg.PRIOR=="COPULA":
        prior=CopulaPrior(cfg)
    elif cfg.PRIOR=="MIXED":
        prior=MixedPrior(cfg)
    else:
        raise ValueError(f"Unknown prior: {cfg.PRIOR}")

    allowed_args=prior.allowed_args(cfg)
    if not "family" in allowed_args[0]:
        cfg.EVEN_FAMILY=False
    if cfg.EVEN_FAMILY:
        family2arg={}
        for zw in allowed_args:
            fam=zw["family"]
            if not fam in family2arg:
                family2arg[fam]=[]
            family2arg[fam].append(zw)
        all_families=list(family2arg.keys())

    def draw_one(sample_count=None, datamode=None, normalize=None):
        if sample_count is None:
            sample_count=cfg.DATA_SAMPLES
        if datamode is None:
            datamode=cfg.DATAMODE
        if normalize is None:
            normalize=cfg.NORMALIZE
        time0("draw_one")
        if cfg.EVEN_FAMILY:
            fam=np.random.choice(all_families)
            args=family2arg[fam][np.random.choice(len(family2arg[fam]))]
            #args=np.random.choice(family2arg[fam])
        else:
            args=allowed_args[np.random.choice(len(allowed_args))]
            #args=np.random.choice(allowed_args)
        time0("draw_one_gen_dist")
        dist=prior.random_distribution(**args, anomalies=cfg.ANOMALIES)
        if normalize:dist,_=prior.normalize(dist)
        time1("draw_one_gen_dist")

        time0("draw_one_describe")
        desc=prior.describe(dist)
        time1("draw_one_describe")

        time0("draw_one_template")
        temp=prior.template(dist)
        time1("draw_one_template")

        attr=prior.identifying_tuple(dist)
        if datamode:
            time0("draw_one_sampling")
            if cfg.EXACT_SAMPLE:
                samples=prior.sample_exact(dist, sample_count)
            else:
                samples=prior.sample(dist, sample_count)
            dim=samples.shape[1]
            assert dim<=cfg.MAX_DIM, f"Sampled dimension {dim} exceeds MAX_DIM {cfg.MAX_DIM}"
            if dim<cfg.MAX_DIM:
                samples=np.pad(samples, ((0,0),(0,cfg.MAX_DIM-samples.shape[1])), mode='constant', constant_values=0)
            if cfg.ADD_INDICATOR:
                indicator1d=np.zeros(cfg.MAX_DIM, dtype=np.float32)
                indicator1d[:dim]=1.0
                indicator2d=np.tile(indicator1d, (sample_count, 1))
                samples=np.concatenate([samples, indicator2d], axis=1)
            time1("draw_one_sampling")
        time1("draw_one")
        if datamode:
            return desc, temp, attr, samples, dist
        else:
            return desc, temp, attr, dist


    def draw_batch(batch_size=32):
        elems={}
        for _ in range(batch_size):
            if cfg.DATAMODE:
                desc, temp, attr, sample, dist=draw_one()
                elems["data"]=elems.get("data",[])+[sample]
            else:
                desc, temp, attr, dist=draw_one()
            elems["desc"]=elems.get("desc",[])+[desc]
            elems["temp"]=elems.get("temp",[])+[temp]
            elems["attr"]=elems.get("attr",[])+[attr]
            elems["dist"]=elems.get("dist",[])+[dist]
        return elems

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def draw_1k():
        descs=[]
        for _ in range(1000):
            desc=draw_one(datamode=False,normalize=False)[0]
            descs.extend(desc)
        return descs

    minimum_integer=cfg.MINIMUM_INTEGER
    v=Vocabulary(draw_1k(),minimum_integer=minimum_integer)
    t=Tokenizer(v,max_seq_len=cfg.MAX_SEQ_LEN)

    vocab_tokens=v.all_tokens()
    vocab_ints=v.all_ints()
    vocab_bools=v.all_bools()

    @torch.no_grad()
    def tokenize_batch(batch):
        if not "desc" in batch:
            return batch
        desc=batch["desc"]
        tokenized=[t.encode(zw) for zw in desc]
        token_ids, numerical_values, masks=[],[],[]
        for elem in tokenized:
            token_ids.append([zw[1] for zw in elem["tokens"]])
            numerical_values.append([zw[2] for zw in elem["tokens"]])
            masks.append(elem["mask"])
        batch["token_id"]=torch.tensor(token_ids, dtype=torch.long)
        batch["numerical_value"]=torch.tensor(numerical_values, dtype=torch.float)
        batch["mask"]=torch.tensor(masks, dtype=torch.bool)
        return batch
    @torch.no_grad()
    def detokenize_batch(batch):
        token_ids=batch["token_id"]
        numerical_values=batch["numerical_value"]
        descs=[]
        for i in range(token_ids.shape[0]):
            desc=t.decode(token_ids[i], numerical_values[i])
            descs.append(desc)
        batch["desc"]=descs
        return batch
    @torch.no_grad()
    def read_batch(batch):
        #generate dist from desc
        desc=batch["desc"]
        dists=[]
        for i in range(len(desc)):
            dist=prior.match(desc[i])
            dists.append(dist)
        batch["dist"]=dists
        return batch

    #INT+BOOL are shortcuts for token predictors. They only allow predicting INT/BOOL values. To make this work, we need to be able to convert between INT/BOOL space prediction and the tokens this represents
    int_tokens=[zw[0] for zw in vocab_ints.items()]
    bool_tokens=[zw[0] for zw in vocab_bools.items()]

    int_tokens=torch.tensor(int_tokens).to(device, non_blocking=True)
    bool_tokens=torch.tensor(bool_tokens).to(device, non_blocking=True)

    token_count=len(vocab_tokens)
    int_count=len(vocab_ints)

    inverse_int_tokens=torch.zeros(token_count, dtype=torch.long).to(device, non_blocking=True)
    inverse_int_tokens[int_tokens]=torch.arange(len(int_tokens)).to(device, non_blocking=True)

    inverse_bool_tokens=torch.zeros(token_count, dtype=torch.long).to(device, non_blocking=True)
    inverse_bool_tokens[bool_tokens]=torch.arange(len(bool_tokens)).to(device, non_blocking=True)



    def get_template(**args):
        dist=prior.random_distribution(**args)
        temp=prior.template(dist)
        ret=t.encode(temp)
        identifying_tuple=prior.identifying_tuple(dist)
        return ret, identifying_tuple

    templates={}
    attribute_to_index={}
    index_to_attribute={}
    templates_of_dim={dim:[] for dim in range(cfg.MIN_DIM, cfg.MAX_DIM+1)}
    has_heads={key:False for key in heads.keys()}
    for args in allowed_args:
        dim=args["dim"]
        template, identity=get_template(**args)
        templates[identity]=template
        templates_of_dim[dim].append(len(attribute_to_index))
        attribute_to_index[identity]=len(attribute_to_index)
        index_to_attribute[len(index_to_attribute)]=identity
        all_tokens=[zw[1] for zw in templates[identity]["tokens"]]
        all_tokens=set(all_tokens)
        has_heads={key: val or (key in all_tokens) for key, val in has_heads.items()}
    print(f"Found Heads:",{heads[key]:has_heads[key] for key in heads})
    attribute_count=len(attribute_to_index)
    print(f"Working with {attribute_count} template(s)")

    def reconstruct_batch(batch):
        input_ids = batch["input_ids"]
        numerical_values = batch["numerical_values"]
        recons=[]
        for i in range(input_ids.size(0)):
            desc=t.decode(input_ids[i], numerical_values[i])
            recon=prior.match(desc)
            recons.append(recon)
        return recons



    model=TransformerAutoencoder(
                    vocab_size=v.token_count,
                    num_token_id=NUM_TOKEN,
                    d_model=cfg.D_MODEL,
                    nhead=cfg.N_HEADS,
                    num_layers=cfg.N_LAYERS,
                    max_seq_len=cfg.MAX_SEQ_LEN,
                    attribute_count=attribute_count,
                    num_head=has_heads[FILL_NUM],
                    token_head=has_heads[FILL_TOKEN],
                    int_head=has_heads[FILL_INT],
                    bool_head=has_heads[FILL_BOOL],
                    token_count=token_count,
                    int_count=int_count,
                    num_cls=cfg.NUM_CLS,
                    deep_num=cfg.DEEP_NUM,
                    use_encoder=not cfg.DATAMODE,
                    )

    param_count=0
    for name, param in model.decoder.named_parameters():
        param_count+=param.numel()
    print("Decoder parameter count:", param_count)

    if cfg.DATAMODE:
        if cfg.MULTI_TOKEN_ENCODER:
            datamodel=MultiTokenDeepSetEncoder(input_dim=cfg.MAX_DIM*2 if cfg.ADD_INDICATOR else cfg.MAX_DIM, hidden_dim=cfg.D_MODEL, num_tokens=cfg.NUM_CLS, num_layers=cfg.SETENC_NUM_LAYERS, num_heads=cfg.SETENC_NUM_HEADS, use_batchnorm=cfg.SETENC_USE_BATCHNORM,mlp_ratio=cfg.SETENC_MLP_RATIO)
        else:
            datamodel=DeepSetEncoder(input_dim=cfg.MAX_DIM*2 if cfg.ADD_INDICATOR else cfg.MAX_DIM, hidden_dim=cfg.D_MODEL*cfg.NUM_CLS, num_layers=cfg.SETENC_NUM_LAYERS, num_heads=cfg.SETENC_NUM_HEADS, use_batchnorm=cfg.SETENC_USE_BATCHNORM,mlp_ratio=cfg.SETENC_MLP_RATIO)

        param_count=0
        for name, param in datamodel.named_parameters():
            param_count+=param.numel()
        print("Data Encoder parameter count:", param_count)


    if cfg.DATAMODE and cfg.PRETRAINED is not None and os.path.exists(cfg.PRETRAINED):
        model.load_state_dict(torch.load(cfg.PRETRAINED, map_location=device))
        print(f"Pretrained Model loaded from {cfg.PRETRAINED}")

    if cfg.DATAMODE:
        datamodel.to(device, non_blocking=True)

    if cfg.FN is not None and os.path.exists(cfg.FN):
        if cfg.DATAMODE:
            datamodel.load_state_dict(torch.load(cfg.FN, map_location=device))
        else:
            model.load_state_dict(torch.load(cfg.FN, map_location=device))
        print(f"Continuing with Model loaded from {cfg.FN}")

    model.to(device, non_blocking=True)


    times={}

    if cfg.DATAMODE:
        model.eval()
        datamodel.eval()
    else:
        model.eval()

    if cfg.DATAMODE and not cfg.LEARNABLE_BASE:
        for param in model.parameters():
            param.requires_grad = False
        print("Froze parameters of Pretrained Model")

    @torch.no_grad()
    def encode(batch): 
        if (not "token_id" in batch) and not cfg.DATAMODE: 
            batch=tokenize_batch(batch) 
        tried_reconstructions=0 
        failed_reconstructions=0 
        metrics=[] 
        examples=[] 
        all_encoded=[] 
        meta_info=[] 
 
        #desc=batch["desc"]
        #dims=find_dim(desc)
        if cfg.DATAMODE:
            data=batch["data"]
            data=np.array(data)
            data=torch.tensor(data, dtype=torch.float)
            data=data.to(device, non_blocking=True)
        else:
            input_ids = batch["token_id"].to(device, non_blocking=True)
            #template_ids = batch["template_ids"].to(device, non_blocking=True)
            numerical_values = batch["numerical_value"].to(device, non_blocking=True)
            if len(numerical_values.shape)==2:
                #add a dummy dimension at the end
                numerical_values=numerical_values.unsqueeze(-1)
            attention_mask = batch["mask"].to(device, non_blocking=True)
        if cfg.DATAMODE:
            encoded=datamodel(data)
        else:
            encoded=model.encode(input_ids, numerical_values, attention_mask)
        return encoded
    @torch.no_grad()
    def decode(encoded, dims=None):
        predicted_attributes=model.attribute_decoder(encoded)
        if not dims is None:
            predicted=[]
            for j in range(predicted_attributes.shape[0]):
                allowed_indices=templates_of_dim[int(dims[j])]
                pred=predicted_attributes[j]
                mask=torch.ones_like(pred, dtype=torch.bool)
                mask[allowed_indices]=False
                pred[mask]=-float("inf")
                predicted.append(pred.argmax())
            predicted_attributes=torch.stack(predicted, dim=0).cpu().numpy()
        else:
            predicted_attributes=torch.argmax(predicted_attributes, dim=-1).cpu().numpy()
        attributes=[index_to_attribute[pred] for pred in predicted_attributes]
        template_ids=np.array([[zw[1] for zw in templates[tuple(attr)]["tokens"]] for attr in attributes])
        template_ids=torch.tensor(template_ids).to(device, non_blocking=True)
        attention_mask=np.array([templates[tuple(attr)]["mask"] for attr in attributes])
        attention_mask=torch.tensor(attention_mask).to(device, non_blocking=True)
        decoded=model.decode(encoded, template_ids, attention_mask)
        dec_float=decoded["float"].squeeze(-1) if "float" in decoded else None
        dec_token=template_ids
        if "token" in decoded:
            dec_token[dec_token==FILL_TOKEN]=torch.argmax(decoded["token"], dim=-1)[dec_token==FILL_TOKEN]
        if "int" in decoded:
            dec_token[dec_token==FILL_INT]=int_tokens[torch.argmax(decoded["int"], dim=-1)[dec_token==FILL_INT]]
        if "bool" in decoded:
            dec_token[dec_token==FILL_BOOL]=bool_tokens[torch.argmax(decoded["bool"], dim=-1)[dec_token==FILL_BOOL]]
        if "float" in decoded:
            dec_token[dec_token==FILL_NUM]=NUM_TOKEN

        return detokenize_batch({"token_id":dec_token, "numerical_value":dec_float,"attr":attributes})

    return draw_batch, encode, decode, read_batch, prior



def load(task, modelpth="", basemodelpth=""):
    cfgs={"minimal":Config(NAME="minimal", MIN_DIM=2, MAX_DIM=3, MAX_SEQ_LEN=300, STEPS_PER_EPOCH=100, BATCH_SIZE=2, VAL_BATCHES=10, RECON_BATCHES=3),
          "run50":Config(NAME="run50"),
          "run3":Config(NAME="run3", MIN_DIM=2,MAX_DIM=3),
          "run3v2":Config(NAME="run3v2", MIN_DIM=2,MAX_DIM=3, LOG_DIST=True),
          "debug":Config(NAME="debug", MIN_DIM=2,MAX_DIM=3, LOG_DIST=True, STEPS_PER_EPOCH=100),
          "run10":Config(NAME="run10",MIN_DIM=2, MAX_DIM=10, LOG_DIST=True),
          "new50":Config(NAME="new50", LOG_DIST=True),
          "debugGMM":Config(NAME="debugGMM",MIN_DIM=2, MAX_DIM=3, PRIOR="GMM", LOG_DIST=True, STEPS_PER_EPOCH=1000, MAX_SEQ_LEN=300),
          "debugGMM2":Config(NAME="debugGMM2",MIN_DIM=2, MAX_DIM=3, PRIOR="GMM", LOG_DIST=True, STEPS_PER_EPOCH=1000, MAX_SEQ_LEN=300, NORMALIZE=False),
          "debugGMM3":Config(NAME="debugGMM3",MIN_DIM=2, MAX_DIM=3, PRIOR="GMM", LOG_DIST=True, STEPS_PER_EPOCH=1000, MAX_SEQ_LEN=300),
          "debugSCM":Config(NAME="debugSCM",MIN_DIM=2, MAX_DIM=3, PRIOR="SCM", LOG_DIST=True, STEPS_PER_EPOCH=1000, MAX_SEQ_LEN=200),
          "debugSCM2":Config(NAME="debugSCM2",MIN_DIM=2, MAX_DIM=3, PRIOR="SCM", LOG_DIST=True, STEPS_PER_EPOCH=1000, MAX_SEQ_LEN=200),
          "debugSCM3":Config(NAME="debugSCM3",MIN_DIM=10, MAX_DIM=10, PRIOR="SCM", LOG_DIST=True, STEPS_PER_EPOCH=1000, MAX_SEQ_LEN=500),
          "debugSCM4":Config(NAME="debugSCM4",MIN_DIM=10, MAX_DIM=10, PRIOR="SCM", LOG_DIST=True, STEPS_PER_EPOCH=1000, MAX_SEQ_LEN=500, MAX_MAX_PARENTS=5),
          "debugSCM5":Config(NAME="debugSCM5",MIN_DIM=10, MAX_DIM=10, PRIOR="SCM", LOG_DIST=True, STEPS_PER_EPOCH=1000, MAX_SEQ_LEN=500, MAX_MAX_PARENTS=5),#functionally unchanged, but after cleaning up scmprior.py
          "do2":Config(NAME="do2",MIN_DIM=1, MAX_DIM=2, LOG_DIST=True, MAX_SEQ_LEN=300, MAX_MAX_PARENTS=5),
          "do10":Config(NAME="do10",MIN_DIM=1, MAX_DIM=10, LOG_DIST=True, MAX_SEQ_LEN=800, MAX_MAX_PARENTS=5),
          "do50":Config(NAME="do50",MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5),
          "debug10":Config(NAME="debug10",MIN_DIM=1, MAX_DIM=10, LOG_DIST=True, MAX_SEQ_LEN=800, MAX_MAX_PARENTS=5, TIME_MEASURE=True),
          "gmm50":Config(NAME="gmm50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, PRIOR="GMM"),
          "scaleup1":Config(NAME="scaleup1",MIN_DIM=50, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, EPOCHS=1, STEPS_PER_EPOCH=100, DATA_SAMPLES_MIN=4096, DATA_SAMPLES_MAX=4096),
          "scaleup2":Config(NAME="scaleup2",MIN_DIM=50, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, EPOCHS=1, STEPS_PER_EPOCH=100, DATA_SAMPLES_MIN=4096, DATA_SAMPLES_MAX=4096, BATCH_SIZE=32,N_HEADS=8, N_LAYERS=5, SETENC_NUM_LAYERS=5, SETENC_NUM_HEADS=8, SETENC_MLP_RATIO=8 , VAL_BATCHES=50, RECON_BATCHES=2),
          "scaleup3":Config(NAME="scaleup3",MIN_DIM=50, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, EPOCHS=1, STEPS_PER_EPOCH=100, DATA_SAMPLES_MIN=8192, DATA_SAMPLES_MAX=8192, BATCH_SIZE=28,N_HEADS=8, N_LAYERS=6, SETENC_NUM_LAYERS=6, SETENC_NUM_HEADS=8, SETENC_MLP_RATIO=8 , VAL_BATCHES=50, RECON_BATCHES=2),
          "scaleup4":Config(NAME="scaleup4",MIN_DIM=50, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, EPOCHS=1, STEPS_PER_EPOCH=100, DATA_SAMPLES_MIN=8192, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48,N_HEADS=8, N_LAYERS=6, SETENC_NUM_LAYERS=6, SETENC_NUM_HEADS=8, SETENC_MLP_RATIO=8 , VAL_BATCHES=50, RECON_BATCHES=2),
          "large50":Config(NAME="large50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, N_HEADS=8, N_LAYERS=6, SETENC_NUM_LAYERS=6, SETENC_NUM_HEADS=8, SETENC_MLP_RATIO=8, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000),
          "wider50":Config(NAME="wider50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, NUM_CLS=100, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000),
          "float50":Config(NAME="float50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, FLOAT_LOSS_MULTIPLIER=10.0, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000),
          "onlyGMM50":Config(NAME="onlyGMM50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000, EPOCHS=20, PRIOR="GMM"),
          "onlySCM50":Config(NAME="onlySCM50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000, EPOCHS=20, PRIOR="SCM"),
          "lowcomp50":Config(NAME="lowcomp50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000, EPOCHS=20, PRIOR="GMM", MAX_COMPONENTS=3, MINIMUM_WEIGHT=0.1, MINIMUM_DELTA=0.1),
          "limitdepth50":Config(NAME="limitdepth50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000, EPOCHS=20, PRIOR="SCM", DEPTH_MODE="uniform", DEPTH_MIN=1, DEPTH_MAX=5),
          "indicator50":Config(NAME="indicator50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000, EPOCHS=20, ADD_INDICATOR=True),
          "nninit50":Config(NAME="nninit50", MIN_DIM=1, MAX_DIM=50, LOG_DIST=True, MAX_SEQ_LEN=3400, MAX_MAX_PARENTS=5, DATA_SAMPLES_MIN=1024, DATA_SAMPLES_MAX=8192, BATCH_SIZE=48, VAL_BATCHES=200, RECON_BATCHES=6, STEPS_PER_EPOCH=1000, EPOCHS=10000, NUM_CLS=20, NORMALIZE=True, DEPTH_MODE="uniform", DEPTH_MIN=1, DEPTH_MAX=5, AUTO_START=True),
          

          }
    cfg=cfgs.get(task)
    cfg.FN=modelpth if modelpth!="" else cfg.FN
    cfg.PRETRAINED=basemodelpth if basemodelpth!="" else cfg.PRETRAINED
    if torch.cuda.is_available()==False:
        print("Working in local mode")
        cfg.TIME_MEASURE=True
    print("Loading RUN with config:", cfg.NAME)
    
    return run(cfg)


if __name__=="__main__":
    draw_batch, encode, decode, read_batch, prior=load("nninit50", "4pred/nninit50/epoch_270.pt", "4pred/nninit50/epoch_270_base.pt")
    b=draw_batch(2)
    e=encode(b)
    d=decode(e)
    b2=read_batch(d)


