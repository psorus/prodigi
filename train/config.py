from dataclasses import dataclass, asdict

@dataclass
class Config:   
    D_MODEL : int=512
    N_HEADS : int=4
    N_LAYERS : int=3
    MAX_SEQ_LEN : int=3400#96#256
    EPOCHS : int=10000
    STEPS_PER_EPOCH : int=1500
    BATCH_ACCUMULATE : int=1
    BATCH_SIZE : int=48
    LR : float=5e-5
    VAL_BATCHES : int=200
    RECON_BATCHES : int=6
    SEED : int=0
    WANDB_PROJECT : str="grange"
    NAME : str="unspecified"
    MIN_DIM: int=1
    MAX_DIM: int=50
    MIN_COMPONENTS: int=1
    MAX_COMPONENTS: int=10
    MIN_MAX_PARENTS: int=1
    MAX_MAX_PARENTS: int=5
    SIMPLE_MAX_PARENTS:bool=True
    #MIN_MAX_CHILDREN: int=2
    #MAX_MAX_CHILDREN: int=2
    #MIN_DEPTH: int=1
    #MAX_DEPTH: int=5
    #MIN_WIDTH: int=1
    #MAX_WIDTH: int=20
    MAX_ALLOWED_HIDDEN: int=0
    #MAX_JUST_NOISE: int=5
    NUM_CLS: int=20
    FN: str="AUTO"
    DEEP_NUM: bool= True
    EPOCH0: int=1
    PRETRAINED: str="AUTO"
    DUAL_EVAL: bool=True
    #USE_ALL_LOSSES: bool=False
    RECON_MODE: str="DUAL"#TRUE, PRED, DUAL
    PRIOR: str="MIXED"
    LOG_TIMES:bool=True
    DATAMODE:bool=True
    DATA_SAMPLES: int=1024
    VARY_DATA_SAMPLES: bool=True
    DATA_SAMPLES_MIN: int=1024
    DATA_SAMPLES_MAX: int=8192
    MINIMUM_INTEGER: int=-1

    LEARNABLE_BASE: bool=True
    ZERO_OUT: bool=False
    ATTR_LOSS_MULTIPLIER: float=1.0
    FLOAT_LOSS_MULTIPLIER: float=1.0
    TOKEN_LOSS_MULTIPLIER: float=1.0
    INT_LOSS_MULTIPLIER: float=1.0
    BOOL_LOSS_MULTIPLIER: float=1.0
    SETENC_NUM_LAYERS: int=4
    SETENC_NUM_HEADS: int=4
    SETENC_USE_BATCHNORM: bool=True
    SETENC_MLP_RATIO: int=4
    EMBED_LOSS:bool=False
    JUST_RECON:bool=False
    DATALOADER_NUM_WORKERS:int=4
    DATALOADER_PREFETCH_FACTOR:int=2
    ANOMALIES: bool=False#no effect. Anomalies=False always
    DEBUG: bool=False
    TIME_MEASURE: bool=False
    EVEN_FAMILY: bool=True
    AUTO_START: bool=True
    EPOCH_PATH: str="AUTO"
    EXACT_SAMPLE: bool=True
    MULTI_TOKEN_ENCODER: bool=True
    SAVE_EPOCH_INTERVAL: int=10
    LOG_DIST:bool=True
    NORMALIZE:bool=True
    NORMALIZE_METHOD:str="zscore"
    MINIMUM_WEIGHT:float=0.01
    MINIMUM_DELTA:float=0.01
    DEPTH_MODE: str="uniform"
    DEPTH_MIN: int=1
    DEPTH_MAX: int=5
    ADD_INDICATOR: bool=False
    TEMPLATE_PATH_BEFORE: str=""
    TEMPLATE_PATH_AFTER: str=""
    ADVANCED_CORR: bool=True
    ETA_MIN: float=0.1
    ETA_MAX: float=10.0


    def to_dict(self):
        return asdict(self)

#    def __dict__(self):
#        return {
#            "D_MODEL": self.D_MODEL,
#            "N_HEADS": self.N_HEADS,
#            "N_LAYERS": self.N_LAYERS,
#            "MAX_SEQ_LEN": self.MAX_SEQ_LEN,
#            "EPOCHS": self.EPOCHS,
#            "STEPS_PER_EPOCH": self.STEPS_PER_EPOCH,
#            "BATCH_SIZE": self.BATCH_SIZE,
#            "LR": self.LR,
#            "FLOAT_LOSS_MULTIPLIER": self.FLOAT_LOSS_MULTIPLIER,
#            "VAL_BATCHES": self.VAL_BATCHES,
#            "RECON_BATCHES": self.RECON_BATCHES,
#            "SEED": self.SEED,
#            "WANDB_PROJECT": self.WANDB_PROJECT,
#            "WANDB_RUN": self.WANDB_RUN,
#        }

