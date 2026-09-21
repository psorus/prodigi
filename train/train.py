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


    wandb.init(project=cfg.WANDB_PROJECT, name=cfg.NAME, config=vars(cfg))
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
        if normalize:dist,_=prior.normalize(dist, method=cfg.NORMALIZE_METHOD)
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
            return desc, temp, attr, samples
        else:
            return desc, temp, attr

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
    ds=SequenceDataset(v,t,draw_one, vary_data_samples=cfg.VARY_DATA_SAMPLES, min_data_samples=cfg.DATA_SAMPLES_MIN, max_data_samples=cfg.DATA_SAMPLES_MAX, default_data_samples=cfg.DATA_SAMPLES)

    vocab_tokens=v.all_tokens()
    vocab_ints=v.all_ints()
    vocab_bools=v.all_bools()

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
    #exit()
    ds.data_samples=cfg.DATA_SAMPLES
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
    if cfg.TIME_MEASURE:
        dl=DataLoader(ds, batch_size=cfg.BATCH_SIZE, collate_fn=collate_fn)
    else:
        dl=DataLoader(ds, batch_size=cfg.BATCH_SIZE, collate_fn=collate_fn, num_workers=cfg.DATALOADER_NUM_WORKERS,prefetch_factor=cfg.DATALOADER_PREFETCH_FACTOR,persistent_workers=True,pin_memory=True, worker_init_fn=worker_init_fn)
    data_iter = iter(dl)

    validation_data=[next(data_iter) for _ in tqdm(range(cfg.VAL_BATCHES), desc="gen val data")]

    def reconstruct_batch(batch):
        input_ids = batch["input_ids"]
        numerical_values = batch["numerical_values"]
        recons=[]
        for i in range(input_ids.size(0)):
            desc=t.decode(input_ids[i], numerical_values[i])
            recon=prior.match(desc)
            recons.append(recon)
        return recons

    validation_dists=[]
    for batch in validation_data:
        try:
            validation_dists.append(reconstruct_batch(batch))
        except BMatchingError as e:
            print("Error reconstructing batch:", e)
            print(batch)
            raise e
    def find_dim(batch):
        return torch.tensor([prior.dim(zw) for zw in batch], dtype=torch.long)
    validation_dims=[find_dim(batch) for batch in validation_dists]



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
        datamodel.train()
    else:
        model.train()

    if cfg.DATAMODE and not cfg.LEARNABLE_BASE:
        for param in model.parameters():
            param.requires_grad = False
        print("Froze parameters of Pretrained Model")

    params=[]
    if cfg.DATAMODE:
        if cfg.LEARNABLE_BASE:
            params=list(model.parameters())+list(datamodel.parameters())
        else:
            params=list(datamodel.parameters())
    else:
        params=list(model.parameters())

        
    optimizer = torch.optim.AdamW(params, lr=cfg.LR)
        
    attr_criterion = nn.CrossEntropyLoss()
    criterions={}
    if has_heads[FILL_TOKEN]:
        if cfg.TOKEN_LOSS_MULTIPLIER==1.0:
            criterions["token"]=nn.CrossEntropyLoss()
        else:
            criterions["token"]=lambda x,y: nn.functional.cross_entropy(x,y)*cfg.TOKEN_LOSS_MULTIPLIER

    if has_heads[FILL_NUM]:
        if cfg.FLOAT_LOSS_MULTIPLIER==1.0:
            criterions["float"]=nn.MSELoss()
        else:
            criterions["float"]=lambda x,y: nn.functional.mse_loss(x,y)*cfg.FLOAT_LOSS_MULTIPLIER


    if has_heads[FILL_INT]:
        if cfg.INT_LOSS_MULTIPLIER:
            criterions["int"]=nn.CrossEntropyLoss()
        else:
            criterions["int"]=lambda x,y: nn.functional.cross_entropy(x,y)*cfg.INT_LOSS_MULTIPLIER

    if has_heads[FILL_BOOL]:
        if cfg.BOOL_LOSS_MULTIPLIER:
            criterions["bool"]=nn.CrossEntropyLoss()
        else:
            criterions["bool"]=lambda x,y: nn.functional.cross_entropy(x,y)*cfg.BOOL_LOSS_MULTIPLIER

    if cfg.EMBED_LOSS:
        embed_criterion=nn.MSELoss()

    def handle_iterable(iterable, accuracy=False, validation=False):
        for index,step in enumerate(iterable):
            time0("dataprep_val" if validation else "dataprep_train")
            if validation:
                batch=validation_data[index]
            else:
                ds.choose_size()
                batch = next(data_iter)
            time1("dataprep_val" if validation else "dataprep_train")

            if cfg.DATAMODE:
                data=batch["data"].to(device, non_blocking=True)
                time0("encoder_data")
                data_enc=datamodel(data)
                time1("encoder_data")
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)  
            template_ids = batch["template_ids"].to(device, non_blocking=True)  
            numerical_values = batch["numerical_values"].to(device, non_blocking=True)
            attention_mask = batch["mask"].to(device, non_blocking=True)   
            attributes = batch["attributes"]
            attributes = [attribute_to_index[tuple(attr)] for attr in attributes]

            if cfg.DATAMODE and cfg.EMBED_LOSS:
                time0("encoder_program")
                true_enc=model.encode(input_ids, numerical_values, attention_mask)
                time1("encoder_program")
           
            time0("forward_val" if validation else "forward_train")
            if cfg.DATAMODE:
                time0("decoder_from_data")
                prediction = model.decode(data_enc, template_ids, attention_mask, evalmode=False)
                time1("decoder_from_data")
            else:
                prediction = model(input_ids, numerical_values, attention_mask, template_ids, attention_mask)
            time1("forward_val" if validation else "forward_train")
            time0("loss_val" if validation else "loss_train")
            pred_attr=prediction["attributes"]
            pred_index=torch.argmax(pred_attr, dim=-1)
            attribute_correct=(pred_index==torch.tensor(attributes).to(device, non_blocking=True))

            prediction={key:val for key,val in prediction.items() if not key in ["attributes"]}
            truth={}
            indice={}
            if "float" in prediction:
                truth["float"]=numerical_values
                indice["float"]=template_ids==FILL_NUM
            if "token" in prediction:
                truth["token"]=input_ids
                indice["token"]=template_ids==FILL_TOKEN
            if "int" in prediction:
                truth["int"]=inverse_int_tokens[input_ids]
                indice["int"]=template_ids==FILL_INT
            if "bool" in prediction:
                truth["bool"]=inverse_bool_tokens[input_ids]
                indice["bool"]=template_ids==FILL_BOOL

            if cfg.ZERO_OUT:
                #for each indice: if attribute correct is false, set all indices to false (ignore entire sample for that head if attribute prediction is wrong)

                for key in indice:
                    #problem: attribute correct has shape (64)
                    #indice has shape (64, seq_len)
                    indice[key]=indice[key] & attribute_correct.unsqueeze(1)


            loss_terms={}
            if accuracy:accuracies={}

            for key in prediction:
                pred=prediction[key]
                true=truth[key]
                ind=indice[key]
                criterion=criterions[key]
                if ind.any():
                    loss=criterion(pred[ind], true[ind])
                    loss_terms[key]=loss
                    if accuracy and key!="float":
                        acc=(pred[ind].argmax(dim=-1)==true[ind]).float().mean().item()
                        accuracies[key]=acc
                        

            loss_terms["attribute"]=attr_criterion(pred_attr, torch.tensor(attributes).to(device, non_blocking=True))
            loss_terms["attribute"]*=cfg.ATTR_LOSS_MULTIPLIER
            if accuracy:
                attr_acc=(pred_attr.argmax(dim=-1)==torch.tensor(attributes).to(device, non_blocking=True)).float().mean().item()
                accuracies["attribute"]=attr_acc
            if cfg.DATAMODE and cfg.EMBED_LOSS:
                loss_terms["embed"]=embed_criterion(data_enc, true_enc.detach())
            time1("loss_val" if validation else "loss_train")
            if accuracy:
                yield loss_terms, accuracies
            else:
                yield loss_terms

    def validation(self):
        time0("validation")
        model.eval()
        if cfg.DATAMODE:
            datamodel.eval()
        with torch.no_grad():
            all_losses=[]
            for loss_terms, accs in handle_iterable(tqdm(validation_data, desc="Validating"), accuracy=True, validation=True):

                loss_terms={key:val.item() for key,val in loss_terms.items()}
                loss_terms["total"]=sum(loss_terms.values())

                all_losses.append(loss_terms)


            val_losses=dic_mean(all_losses)
            val_losses={f"val/{key}_loss":val for key,val in val_losses.items()}
            val_losses.update({f"val/{key}_acc":val for key,val in accs.items()})
            time1("validation")
            model.train()
            if cfg.DATAMODE:
                datamodel.train()
            return val_losses

    def try_reconstruction(plotone=False, trueshape=True):
        time0("reconstruction"+str(trueshape))
        model.eval()
        if cfg.DATAMODE:
            datamodel.eval()
        with torch.inference_mode():
            tried_reconstructions=0
            failed_reconstructions=0
            mmd_distances=[]
            mmd_pvalues=[]
            metrics=[]
            examples=[]
            all_encoded=[]
            meta_info=[]
            families=[]

            means=[]
            stds=[]

            for i in tqdm(range(cfg.RECON_BATCHES)):
                batch=validation_data[i]
                dists=validation_dists[i]
                dims=validation_dims[i]
                if cfg.DATAMODE:
                    data=batch["data"].to(device, non_blocking=True)
                meta_info.append(dic_merge([prior.meta_info(dist) for dist in dists]))
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                template_ids = batch["template_ids"].to(device, non_blocking=True)
                numerical_values = batch["numerical_values"].to(device, non_blocking=True)
                attention_mask = batch["mask"].to(device, non_blocking=True)
                time0("encoding"+str(trueshape))
                if cfg.DATAMODE:
                    encoded=datamodel(data)
                else:
                    if cfg.DEBUG:print("calling model.encode with", input_ids.shape, numerical_values.shape, attention_mask.shape)
                    encoded=model.encode(input_ids, numerical_values, attention_mask)
                    if cfg.DEBUG:print("got encoded with shape", encoded.shape)
                time1("encoding"+str(trueshape))
                all_encoded.append(encoded.cpu().detach().numpy())

                if not trueshape:
                    predicted_attributes=model.attribute_decoder(encoded)

                    #only allow predicted attributes that have the right dimensionality
                    #I have: The true dimensionality in dims, the allowed attributes for a given dimensionality in templates_of_dim[dims[j].item()]. Simply ignore all likelihoods that are not allowed
                    predicted=[]
                    for j in range(predicted_attributes.shape[0]):
                        allowed_indices=templates_of_dim[dims[j].item()]
                        pred=predicted_attributes[j]
                        mask=torch.ones_like(pred, dtype=torch.bool)
                        mask[allowed_indices]=False
                        pred[mask]=-float("inf")
                        predicted.append(pred.argmax())
                    predicted_attributes=torch.stack(predicted, dim=0).cpu().numpy()
                    #predicted_attributes=predicted_attributes.argmax(dim=-1).cpu().numpy()
                    attributes=[index_to_attribute[pred] for pred in predicted_attributes]
                    template_ids=np.array([[zw[1] for zw in templates[tuple(attr)]["tokens"]] for attr in attributes])
                    template_ids=torch.tensor(template_ids).to(device, non_blocking=True)
                    attention_mask=np.array([templates[tuple(attr)]["mask"] for attr in attributes])
                    attention_mask=torch.tensor(attention_mask).to(device, non_blocking=True)

                time0("decoding"+str(trueshape))
                decoded=model.decode(encoded, template_ids, attention_mask)
                time1("decoding"+str(trueshape))
                dec_float=decoded["float"] if "float" in decoded else None
                dec_token=template_ids
                if "token" in decoded:
                    dec_token[dec_token==FILL_TOKEN]=torch.argmax(decoded["token"], dim=-1)[dec_token==FILL_TOKEN]
                if "int" in decoded:
                    dec_token[dec_token==FILL_INT]=int_tokens[torch.argmax(decoded["int"], dim=-1)[dec_token==FILL_INT]]
                if "bool" in decoded:
                    dec_token[dec_token==FILL_BOOL]=bool_tokens[torch.argmax(decoded["bool"], dim=-1)[dec_token==FILL_BOOL]]
                if "float" in decoded:
                    dec_token[dec_token==FILL_NUM]=NUM_TOKEN

                for j in range(cfg.BATCH_SIZE):
                    tried_reconstructions+=1

                    decoded_desc=t.decode(dec_token[j], dec_float[j])
                    if "UNK" in str(decoded_desc).upper():
                        print(decoded_desc)
                        exit()


                    try:
                        recon_dist=prior.match(decoded_desc)
                        if "family" in recon_dist:
                            families.append(recon_dist["family"])
                        else:
                            families.append("unknown")
                    except BMatchingError as e:
                        failed_reconstructions+=1
                        examples.append(["FAILED_RE",prior.describe(dists[j]), decoded_desc])
                        continue
                    except SCMError as e:
                        failed_reconstructions+=1
                        examples.append(["FAILED_SCM",prior.describe(dists[j]), decoded_desc])

                    if prior.dim(recon_dist)!=prior.dim(dists[j]):
                        examples.append(["FAILED_DIM",prior.describe(dists[j]), decoded_desc])
                        failed_reconstructions+=1
                        continue
                    examples.append(["FUNCTIONAL",prior.describe(dists[j]), decoded_desc])
                    time0("mmd"+str(trueshape))
                    mmd_metrics=fast_metrics(prior,dists[j],recon_dist)
                    time1("mmd"+str(trueshape))
                    metrics.append(mmd_metrics)

                    if cfg.LOG_DIST:
                        samp=prior.sample_exact(recon_dist, 1000)
                        means.append(np.mean(np.mean(samp, axis=0)))
                        stds.append(np.mean(np.std(samp, axis=0)))

            recon_fail_rate=failed_reconstructions/tried_reconstructions
            pvalues=[zw["recon/pvalue"] for zw in metrics if "recon/pvalue" in zw]
            metrics=dic_mean(metrics)
            meta_info=dic_concat(meta_info)
            #family wise pvalues
            if "family" in meta_info:
                #families=meta_info["family"]
                families=np.array(families)
                for family in np.unique(families):
                    curr_pv=np.array(pvalues)[families==family]
                    metrics["familywise_recon/pvalue_"+str(family)]=np.mean(curr_pv)
                    metrics["familywise_recon/family_count_"+str(family)]=len(curr_pv)

            pvalue_quantiles=[0.0,0.05,0.25,0.5,0.75,0.95,1.0]
            identifier="trueshape" if trueshape else "predshape"
            if len(pvalues)>0:
                metrics.update({f"pvalues_{identifier}/quantile_{int(q*100)}": np.quantile(pvalues, q) for q in pvalue_quantiles})

            dic= {"recon/fail_rate":recon_fail_rate}
            dic.update(metrics)
            if cfg.LOG_DIST:
                dic["recon/means"]=np.mean(means)
                dic["recon/stds"]=np.mean(stds)
                dic["recon/means_std"]=np.std(means)
                dic["recon/stds_std"]=np.std(stds)
            if len(examples)>0:
                save_path=f"examples/{cfg.NAME}/"
                os.makedirs(save_path, exist_ok=True)
                with open(f"{save_path}/{identifier}_examples_{plotone:05}.json", "w") as f:

                    json.dump(safe_json(examples), f, indent=2)


            dic={key.replace("recon/", f"recon_{identifier}/"):val for key,val in dic.items()}
            if len(all_encoded)>0:
                save_path2=f"repr/"
                os.makedirs(save_path2, exist_ok=True)
                fn=f"{save_path2}/{cfg.NAME}_{identifier}.npz"
                all_encoded=np.concatenate(all_encoded, axis=0)
                np.savez_compressed(fn, data=all_encoded, **meta_info)

            time1("reconstruction"+str(trueshape))

            model.train()
            if cfg.DATAMODE:
                datamodel.train()

            return dic


    if cfg.JUST_RECON:
        print(try_reconstruction("debug", trueshape=False))

        exit()

    epoch0=cfg.EPOCH0
    if cfg.AUTO_START:
        if os.path.exists(cfg.EPOCH_PATH):
            with open(cfg.EPOCH_PATH, "r") as f:
                try:
                    epoch0=int(f.read())+1
                except:
                    print(f"Could not read epoch from {cfg.EPOCH_PATH}, starting from epoch {epoch0}")

    #backup all current saves to a epoch0 stamped folder
    if cfg.AUTO_START:
        new_path=f"models/{cfg.NAME}/at_{epoch0}"
        os.makedirs(new_path, exist_ok=True)
        for fn in ["last.pt", "lowest_loss.pt", "lowest_mmd.pt", "last_base.pt", "lowest_loss_base.pt", "lowest_mmd_base.pt"]:
            old_fn=f"models/{cfg.NAME}/{fn}"
            if os.path.exists(old_fn):
                #copy file
                print("Backing up", old_fn, "to", new_path)
                os.system(f"cp {old_fn} {new_path}/{fn}")



    print(f"Training started on device: {device}")
    lowest_val_loss=float('inf') 
    lowest_avg_mmd=float('inf')
    for epoch in range(epoch0, cfg.EPOCHS + 1):
        resettimes()
        iterable=tqdm(range(1, cfg.STEPS_PER_EPOCH + 1), desc=f"Epoch {epoch}/{cfg.EPOCHS}")
        all_losses=[]
        time0("train")
        for step,loss_terms in enumerate(handle_iterable(iterable)):

            total_loss=sum(loss_terms.values())

            loss_terms={key:val.item() for key,val in loss_terms.items()}
            loss_terms["total"]=total_loss.item()

            all_losses.append(loss_terms)

            time0("backward")
            (total_loss/cfg.BATCH_ACCUMULATE).backward()
            time1("backward")
            
            if step % cfg.BATCH_ACCUMULATE == 0:
                time0("optimizer_step")
                optimizer.step()
                optimizer.zero_grad()
                time1("optimizer_step")

        train_losses=dic_mean(all_losses)
        train_losses={f"train/{key}_loss":val for key,val in train_losses.items()}
        time1("train")

        torch.cuda.empty_cache()

        val_metrics = validation(model)
        metrics={}
        metrics.update(train_losses)
        metrics.update(val_metrics)
        metrics["epoch"]=epoch
        if cfg.RECON_MODE=="DUAL":
            metrics.update(try_reconstruction(epoch, trueshape=False))
            metrics.update(try_reconstruction(epoch, trueshape=True))
        elif cfg.RECON_MODE=="TRUE":
            metrics.update(try_reconstruction(epoch, trueshape=True))
        elif cfg.RECON_MODE=="PRED":
            metrics.update(try_reconstruction(epoch, trueshape=False))
        else:
            raise ValueError(f"Unknown RECON_MODE: {cfg.RECON_MODE}")
        if cfg.LOG_TIMES:metrics.update(alltimes())

        def save_all(path):
            if cfg.DATAMODE:
                if cfg.LEARNABLE_BASE:
                    torch.save(model.state_dict(), path.replace(".pt","")+"_base.pt")
                    torch.save(datamodel.state_dict(), path)
                else:
                    torch.save(datamodel.state_dict(), path)

            else:
                torch.save(model.state_dict(), model_path)

        if metrics["val/total_loss"]<lowest_val_loss:
            lowest_val_loss=metrics["val/total_loss"]
            model_path=f"models/{cfg.NAME}/lowest_loss.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            #torch.save(model.state_dict(), model_path)
            save_all(model_path)
            print("reached new lowest val loss, model saved")
        if "recon/avg_mmd" in metrics and metrics["recon/avg_mmd"]>=0.0 and metrics["recon/avg_mmd"]<lowest_avg_mmd:
            lowest_avg_mmd=metrics["recon/avg_mmd"]
            model_path=f"models/{cfg.NAME}/lowest_mmd.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            #torch.save(model.state_dict(), model_path)
            save_all(model_path)
            print("reached new lowest avg mmd, model saved")
        model_path=f"models/{cfg.NAME}/last.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        #torch.save(model.state_dict(), model_path)
        save_all(model_path)
        if epoch%cfg.SAVE_EPOCH_INTERVAL==0:
            model_path=f"models/{cfg.NAME}/epoch_{epoch}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            save_all(model_path)
        with open(f"models/{cfg.NAME}/epoch","w") as f:
            f.write(str(epoch))

        wandb.log(metrics)
        print(json.dumps(safe_json(metrics), indent=2))




if __name__=="__main__":
    cfgs={"minimal":Config(NAME="minimal", MIN_DIM=2, MAX_DIM=3, MAX_SEQ_LEN=300, STEPS_PER_EPOCH=100, BATCH_SIZE=2, VAL_BATCHES=10, RECON_BATCHES=3),
          "pretrain50":Config(NAME="pretrain50", EPOCHS=200, NORMALIZE=False),
          "prodigi50_start":Config(NAME="final50", EPOCHS=20000, NORMALIZE=True,FN="models/pretrain50/last.pt",PRETRAINED="models/pretrain50/last_base.pt", EPOCH0=1),
          "prodigi50":Config(NAME="final50", EPOCHS=20000, NORMALIZE=True),
          

          }
    task="minimal"
    if len(sys.argv)>1:
        task=sys.argv[1]
    cfg=cfgs.get(task)
    if torch.cuda.is_available()==False:
        print("Working in local mode")
        cfg.TIME_MEASURE=True
    print("Starting RUN with config:", cfg.NAME)
    
    run(cfg)
