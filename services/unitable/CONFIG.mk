##################################################
#                 Configurations                 #
##################################################

#
# Datasets
#

# label type
LABEL_IMAGE = ++trainer.label_type="image"
LABEL_HTML = ++trainer.label_type="html" "++trainer.train.loss_weights.html=1"
LABEL_CELL = ++trainer.label_type="cell" "++trainer.train.loss_weights.cell=1"
LABEL_BBOX = ++trainer.label_type="bbox" "++trainer.train.loss_weights.bbox=1"
MEAN = [0.86597056,0.88463002,0.87491087]
STD = [0.20686628,0.18201602,0.18485524]

# augmentation
AUG_VQVAE = dataset/augmentation=vqvae
AUG_BEIT = dataset/augmentation=beit \
	++dataset.augmentation.mean=$(MEAN) ++dataset.augmentation.std=$(STD)
AUG_RESIZE_NORM = dataset/augmentation=resize_normalize \
	++dataset.augmentation.transforms.2.mean=$(MEAN) ++dataset.augmentation.transforms.2.std=$(STD)

# single dataset
DATA_SINGLE = dataset=single_dataset
PUBTABNET = $(DATA_SINGLE) \
	+dataset/pubtabnet@dataset.train_dataset=train_dataset \
	+dataset/pubtabnet@dataset.valid_dataset=valid_dataset \
	+dataset/pubtabnet@dataset.test_dataset=test_dataset
MINIPUBTABNET = $(DATA_SINGLE) \
	+dataset/mini_pubtabnet@dataset.train_dataset=train_dataset \
	+dataset/mini_pubtabnet@dataset.valid_dataset=valid_dataset \
	+dataset/mini_pubtabnet@dataset.test_dataset=test_dataset

# multiple datasets
DATA_MULTI = dataset=concat_dataset
PUBTABNET_M = +dataset/pubtabnet@dataset.train.d1=train_dataset \
	+dataset/pubtabnet@dataset.valid.d1=valid_dataset \
	+dataset/pubtabnet@dataset.test.d1=test_dataset
SYN_MARKET_M = +dataset/synthtabnet_marketing@dataset.train.d2=train_dataset \
	+dataset/synthtabnet_marketing@dataset.valid.d2=valid_dataset \
	+dataset/synthtabnet_marketing@dataset.test.d2=test_dataset
SYN_FIN_M = +dataset/synthtabnet_fintabnet@dataset.train.d3=train_dataset \
	+dataset/synthtabnet_fintabnet@dataset.valid.d3=valid_dataset \
	+dataset/synthtabnet_fintabnet@dataset.test.d3=test_dataset
SYN_SPARSE_M = +dataset/synthtabnet_sparse@dataset.train.d4=train_dataset \
	+dataset/synthtabnet_sparse@dataset.valid.d4=valid_dataset \
	+dataset/synthtabnet_sparse@dataset.test.d4=test_dataset
SYN_PUB_M = +dataset/synthtabnet_pubtabnet@dataset.train.d5=train_dataset \
	+dataset/synthtabnet_pubtabnet@dataset.valid.d5=valid_dataset \
	+dataset/synthtabnet_pubtabnet@dataset.test.d5=test_dataset
PUBTABLES_M = +dataset/pubtables1m@dataset.train.d7=train_dataset \
	+dataset/pubtables1m@dataset.valid.d7=valid_dataset \
	+dataset/pubtables1m@dataset.test.d7=test_dataset
TABLEBANK_M = +dataset/tablebank@dataset.train.d8=train_dataset \
	+dataset/tablebank@dataset.valid.d8=valid_dataset \
	+dataset/tablebank@dataset.test.d8=test_dataset
FINTABNET_M = +dataset/fintabnet@dataset.train.d9=train_dataset \
	+dataset/fintabnet@dataset.valid.d9=valid_dataset \
	+dataset/fintabnet@dataset.test.d9=test_dataset

DATA_VQVAE_1M = $(DATA_MULTI) \
	$(PUBTABNET_M) $(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M)
DATA_VQVAE_2M = $(DATA_MULTI) \
	$(PUBTABNET_M) $(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M) \
	$(PUBTABLES_M) $(TABLEBANK_M)

PUBTABLES1M = $(DATA_MULTI) $(PUBTABLES_M)
FINTABNET = $(DATA_MULTI) $(FINTABNET_M)

PUB_SYN = $(DATA_MULTI) \
	$(PUBTABNET_M) $(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M)

PUB_SYN_FIN = $(DATA_MULTI) $(PUBTABNET_M) $(FINTABNET_M) \
	$(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M)

PUB_SYN_PUB1M = $(DATA_MULTI) $(PUBTABNET_M) $(PUBTABLES_M) \
	$(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M)

SYN = $(DATA_MULTI) $(SYN_MARKET_M) $(SYN_FIN_M) $(SYN_SPARSE_M) $(SYN_PUB_M)

SYN_fin = $(DATA_MULTI) $(SYN_FIN_M)
SYN_market = $(DATA_MULTI) $(SYN_MARKET_M)
SYN_pub = $(DATA_MULTI) $(SYN_PUB_M)
SYN_sparse = $(DATA_MULTI) $(SYN_SPARSE_M)

#
# Vocab
#
VOCAB_NONE = vocab=empty
VOCAB_HTML = vocab=html
VOCAB_BBOX = vocab=bbox
VOCAB_CELL = vocab=cell


#
# Trainer
#

# trainer type
TRAINER_VQVAE = trainer=vqvae
TRAINER_BEIT = trainer=beit
TRAINER_TABLE = trainer=table

# input image size
I224 = ++trainer.img_size=[224,224]
I448 = ++trainer.img_size=[448,448]
I112_448 = ++trainer.img_size=[112,448]

# max sequence length
SEQ200 = trainer.max_seq_len=200
SEQ512 = trainer.max_seq_len=512
SEQ1024 = trainer.max_seq_len=1024

# batch size + epoch
BATCH24 = ++trainer.train.dataloader.batch_size=24 ++trainer.valid.dataloader.batch_size=24
BATCH48 = ++trainer.train.dataloader.batch_size=48 ++trainer.valid.dataloader.batch_size=48
BATCH72 = ++trainer.train.dataloader.batch_size=72 ++trainer.valid.dataloader.batch_size=72
BATCH80 = ++trainer.train.dataloader.batch_size=80 ++trainer.valid.dataloader.batch_size=80
BATCH96 = ++trainer.train.dataloader.batch_size=96 ++trainer.valid.dataloader.batch_size=96
BATCH256 = ++trainer.train.dataloader.batch_size=256 ++trainer.valid.dataloader.batch_size=256
BATCH384 = ++trainer.train.dataloader.batch_size=384 ++trainer.valid.dataloader.batch_size=384

EPOCH24 = ++trainer.train.epochs=24
EPOCH30 = ++trainer.train.epochs=30
EPOCH48 = ++trainer.train.epochs=48

# optimizer
OPT_ADAMW = trainer/train/optimizer=adamw
OPT_WD5e2 = ++trainer.train.optimizer.weight_decay=5e-2

# lr + scheduler
LR_5e4 = ++trainer.train.optimizer.lr=5e-4
LR_3e4 = ++trainer.train.optimizer.lr=3e-4
LR_1e4 = ++trainer.train.optimizer.lr=1e-4
LR_8e5 = ++trainer.train.optimizer.lr=8e-5

LR_cosine = trainer/train/lr_scheduler=cosine ++trainer.train.lr_scheduler.lr_lambda.min_ratio=5e-3
LR_cosine93k_warm6k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=93400 ++trainer.train.lr_scheduler.lr_lambda.warmup=5800
LR_cosine77k_warm8k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=76600 ++trainer.train.lr_scheduler.lr_lambda.warmup=7660
LR_cosine30k_warm4k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=30500 ++trainer.train.lr_scheduler.lr_lambda.warmup=4000
LR_cosine8k_warm1k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=7600 ++trainer.train.lr_scheduler.lr_lambda.warmup=800
LR_cosine44k_warm6k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=44100 ++trainer.train.lr_scheduler.lr_lambda.warmup=5500
LR_cosine118k_warm15k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=117800 ++trainer.train.lr_scheduler.lr_lambda.warmup=14700
LR_cosine216k_warm27k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=216000 ++trainer.train.lr_scheduler.lr_lambda.warmup=27000
LR_cosine32k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=32000 ++trainer.train.lr_scheduler.lr_lambda.warmup=0
LR_cosine118k = $(LR_cosine) ++trainer.train.lr_scheduler.lr_lambda.total_step=118000 ++trainer.train.lr_scheduler.lr_lambda.warmup=0

GRAD_CLIP12 = ++trainer.train.grad_clip=12

# vqvae
VQVAE_TEMP_1M = ++trainer.train.starting_temp=1. \
	++trainer.train.temp_min=5e-3 ++trainer.train.temp_anneal_rate=1e-3
VQVAE_TEMP_2M = ++trainer.train.starting_temp=1. \
	++trainer.train.temp_min=1e-3 ++trainer.train.temp_anneal_rate=2e-4

# pretraining specific
TRANS448_VQVAE224_GRID28_MASK300 = ++trainer.trans_size=[448,448] ++trainer.vqvae_size=[224,224] ++trainer.grid_size=28 ++trainer.num_mask_patches=300
VQVAE1M_WEIGHTS = $(MODEL_VQVAE) ++trainer.vqvae_weights="../unitable_weights/vqvae_1m.pt"
VQVAE2M_WEIGHTS = $(MODEL_VQVAE_L) ++trainer.vqvae_weights="../unitable_weights/vqvae_2m.pt"

# finetuning specific
WEIGHTS_mtim_1m_base = ++trainer.trainer.beit_pretrained_weights="../unitable_weights/ssp_1m_base.pt"
WEIGHTS_mtim_1m_large = ++trainer.trainer.beit_pretrained_weights="../unitable_weights/ssp_1m_large.pt"
WEIGHTS_mtim_2m_base = ++trainer.trainer.beit_pretrained_weights="../unitable_weights/ssp_2m_base.pt"
WEIGHTS_mtim_2m_large = ++trainer.trainer.beit_pretrained_weights="../unitable_weights/ssp_2m_large.pt"
LOCK_MTIM_4 = ++trainer.trainer.freeze_beit_epoch=4

#
# Models
#

# model type
MODEL_VQVAE = model=vqvae
MODEL_VQVAE_L = $(MODEL_VQVAE) ++model.codebook_tokens=16384 ++model.hidden_dim=512
MODEL_BEIT = model=beit
MODEL_ENCODER_DECODER = model=encoderdecoder

# backbone for input preprocessing: resnet, linear projection, and convstem
IMGCNN = model/model/backbone=imgcnn
IMGLINEAR = model/model/backbone=imglinear
IMGCONVSTEM = model/model/backbone=imgconvstem

# number of layers
E4 = ++model.model.encoder.nlayer=4
E12 = ++model.model.encoder.nlayer=12
E24 = ++model.model.encoder.nlayer=24
D4 = ++model.model.decoder.nlayer=4

# transformer layer: attention heads, hidden size, activation, norm
FF4 = ++model.ff_ratio=4

NHEAD8 = ++model.nhead=8
NHEAD12 = ++model.nhead=12

NORM_FIRST = ++model.norm_first=true
NORM_LAST = ++model.norm_first=false

ACT_RELU = ++model.activation="relu"
ACT_GELU = ++model.activation="gelu"

D_MODEL512 = ++model.d_model=512
D_MODEL768 = ++model.d_model=768

# regularization
REG_d00 = ++model.dropout=0.0
REG_d02 = ++model.dropout=0.2

# linear projection patch size
P16 = ++model.backbone_downsampling_factor=16
P28 = ++model.backbone_downsampling_factor=28
P32 = ++model.backbone_downsampling_factor=32

# cnn backbone
R18 = ++model.model.backbone.backbone._target_=torchvision.models.resnet18 \
	++model.model.backbone.output_channels=512

MTIM_BASE = $(MODEL_BEIT) $(IMGLINEAR) $(NHEAD8) $(FF4) $(ACT_GELU) \
	$(NORM_FIRST) $(D_MODEL512) $(REG_d02) $(P16) $(E4)
MTIM_LARGE = $(MODEL_BEIT) $(IMGLINEAR) $(NHEAD12) $(FF4) $(ACT_GELU) \
	$(NORM_FIRST) $(D_MODEL768) $(REG_d02) $(P16) $(E12)

ARCH_BASE = $(MTIM_BASE) $(MODEL_ENCODER_DECODER) $(D4)
ARCH_LARGE = $(MTIM_LARGE) $(MODEL_ENCODER_DECODER) $(D4)


###############################################
#                 Experiments                 #
###############################################

TRAIN_vqvae := $(VOCAB_NONE) \
	$(LABEL_IMAGE) $(AUG_VQVAE) $(I224) \
	$(TRAINER_VQVAE) $(OPT_ADAMW) $(LR_1e4) $(EPOCH24)

TRAIN_mtim := $(VOCAB_NONE) \
	$(LABEL_IMAGE) $(AUG_BEIT) \
	$(TRAINER_BEIT) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_5e4) \
	$(TRANS448_VQVAE224_GRID28_MASK300)

#
# mini_pubtabnet pretraining example (dataset code: mini)
#

# vq-vae
# > make experiments/vqvae_mini/.done_pretrain
EXP_vqvae_mini := $(TRAIN_vqvae) $(MINIPUBTABNET) $(VQVAE_TEMP_2M) $(BATCH80) $(MODEL_VQVAE) $(LR_cosine32k)

# visual encoder pretraining - masked tabular image modeling (MTIM)
# > make experiments/mtim_mini_base/.done_pretrain
EXP_mtim_mini_base := $(TRAIN_mtim) $(MINIPUBTABNET) $(VQVAE2M_WEIGHTS) $(MTIM_BASE) \
	$(BATCH384) $(LR_cosine8k_warm1k) $(EPOCH24)

#
# mini_pubtabnet finetuning example
#

# table structure (task code: html)
# > make experiments/ssp_2m_mini_html_base/.done_finetune
TRAIN_mini_html := $(VOCAB_HTML) \
	$(MINIPUBTABNET) $(LABEL_HTML) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ512) \
	$(EPOCH48) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_8e5)

EXP_ssp_2m_mini_html_base := $(TRAIN_mini_html) $(ARCH_BASE) \
	$(WEIGHTS_mtim_2m_base) $(LOCK_MTIM_4) $(BATCH72) $(LR_cosine93k_warm6k)

# table cell bbox (task code: bbox)
# > make experiments/ssp_2m_mini_bbox_base/.done_finetune
TRAIN_mini_bbox := $(VOCAB_BBOX) \
	$(MINIPUBTABNET) $(LABEL_BBOX) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ1024) \
	$(EPOCH30) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_3e4) $(GRAD_CLIP12)

EXP_ssp_2m_mini_bbox_base := $(TRAIN_mini_bbox) $(ARCH_BASE) \
	$(WEIGHTS_mtim_2m_base) $(LOCK_MTIM_4) $(BATCH48) $(LR_cosine77k_warm8k)

# table cell content (task code: cell)
# > make experiments/ssp_2m_mini_cell_base/.done_finetune
TRAIN_mini_cell := $(VOCAB_CELL) \
	$(MINIPUBTABNET) $(LABEL_CELL) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I112_448) $(SEQ200) \
	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_8e5) $(GRAD_CLIP12)

EXP_ssp_2m_mini_cell_base := $(TRAIN_mini_cell) $(ARCH_BASE) \
	$(WEIGHTS_mtim_2m_base) $(LOCK_MTIM_4) $(BATCH24) $(LR_cosine216k_warm27k)

#
# cross-dataset pretraining
#

# vq-vae
EXP_vqvae_1M := $(TRAIN_vqvae) $(DATA_VQVAE_1M) $(VQVAE_TEMP_1M) $(BATCH80) $(MODEL_VQVAE) $(LR_cosine32k)
EXP_vqvae_2M := $(TRAIN_vqvae) $(DATA_VQVAE_2M) $(VQVAE_TEMP_2M) $(BATCH48) $(MODEL_VQVAE_L) $(LR_cosine118k)

# visual encoder pretraining
EXP_mtim_1M_base := $(TRAIN_mtim) $(PUB_SYN) $(VQVAE1M_WEIGHTS) $(MTIM_BASE) \
	$(BATCH384) $(LR_cosine8k_warm1k) $(EPOCH24)
EXP_mtim_1M_large := $(TRAIN_mtim) $(PUB_SYN) $(VQVAE1M_WEIGHTS) $(MTIM_LARGE) \
	$(BATCH96) $(LR_cosine30k_warm4k) $(EPOCH24)
EXP_mtim_2M_base := $(TRAIN_mtim) $(DATA_VQVAE_2M) $(VQVAE2M_WEIGHTS) $(MTIM_BASE) \
	$(BATCH256) $(LR_cosine44k_warm6k) $(EPOCH48)
EXP_mtim_2M_large := $(TRAIN_mtim) $(DATA_VQVAE_2M) $(VQVAE2M_WEIGHTS) $(MTIM_LARGE) \
	$(BATCH96) $(LR_cosine118k_warm15k) $(EPOCH48)

#
# cross-dataset finetuning
#

# table structure
# > make experiments/ssp_2m_syn_pub_html_medium/.done_finetune
TRAIN_syn_pub_html := $(VOCAB_HTML) \
	$(PUB_SYN) $(LABEL_HTML) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ512) \
	$(EPOCH48) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_8e5)

EXP_ssp_2m_syn_pub_html_large := $(TRAIN_syn_pub_html) $(ARCH_LARGE) \
	$(WEIGHTS_mtim_2m_large) $(LOCK_MTIM_4) $(BATCH72) $(LR_cosine93k_warm6k)

# table cell bbox
# > make experiments/ssp_2m_syn_pub_bbox_medium/.done_finetune
TRAIN_syn_pub_bbox := $(VOCAB_BBOX) \
	$(PUB_SYN) $(LABEL_BBOX) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I448) $(SEQ1024) \
	$(EPOCH30) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_3e4) $(GRAD_CLIP12)

EXP_ssp_2m_syn_pub_bbox_large := $(TRAIN_syn_pub_bbox) $(ARCH_LARGE) \
	$(WEIGHTS_mtim_2m_large) $(LOCK_MTIM_4) $(BATCH48) $(LR_cosine77k_warm8k)

# table cell content
# > make experiments/syn_pub_pub1m_cell_medium/.done_finetune
TRAIN_syn_pub_pub1m_cell := $(VOCAB_CELL) \
	$(PUB_SYN_PUB1M) $(LABEL_CELL) $(AUG_RESIZE_NORM) \
	$(TRAINER_TABLE) $(I112_448) $(SEQ200) \
	$(EPOCH24) $(OPT_ADAMW) $(OPT_WD5e2) $(LR_8e5) $(GRAD_CLIP12)

EXP_ssp_2m_syn_pub_pub1m_cell_large := $(TRAIN_syn_pub_pub1m_cell) $(ARCH_LARGE) \
	$(WEIGHTS_mtim_2m_base) $(LOCK_MTIM_4) $(BATCH24) $(LR_cosine216k_warm27k)