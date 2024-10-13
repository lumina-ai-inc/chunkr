SHELL := /bin/bash
VENV_NAME := unitable
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_NAME)
PYTHON := $(CONDA_ACTIVATE) && python
PIP := $(CONDA_ACTIVATE) && pip3
# Stacked single-node multi-worker: https://pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker 
TORCHRUN = $(CONDA_ACTIVATE) && torchrun --rdzv-backend=c10d --rdzv_endpoint localhost:0 --nnodes=1 --nproc_per_node=$(NGPU)

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

#
# Virtual Environment Targets
#
clean:
> rm -f .venv_done

.done_venv: clean
> conda create -n $(VENV_NAME) python=3.9 -y
> $(PIP) install -r requirements.txt
> $(PIP) install -e .
> touch $@

#
# Download pretrained and UniTable model weights
#
WEIGHTS_PATH = experiments/unitable_weights
M_VQVAE_1M = $(WEIGHTS_PATH)/vqvae_1m.pt
M_VQVAE_2M = $(WEIGHTS_PATH)/vqvae_2m.pt
M_SSP_1M_BASE = $(WEIGHTS_PATH)/ssp_1m_base.pt
M_SSP_1M_LARGE = $(WEIGHTS_PATH)/ssp_1m_large.pt
M_SSP_2M_BASE = $(WEIGHTS_PATH)/ssp_2m_base.pt
M_SSP_2M_LARGE = $(WEIGHTS_PATH)/ssp_2m_large.pt
UNITABLE_HTML = $(WEIGHTS_PATH)/unitable_large_structure.pt
UNITABLE_BBOX = $(WEIGHTS_PATH)/unitable_large_bbox.pt
UNITABLE_CELL = $(WEIGHTS_PATH)/unitable_large_content.pt

.done_download_weights:
ifeq ("$(words $(wildcard $(WEIGHTS_PATH)/*.pt))", "9")
> $(info All 9 model weights have already been downloaded to $(WEIGHTS_PATH).)
else
> $(info There should be 9 weights file under $(WEIGHTS_PATH), but only $(words $(wildcard $(WEIGHTS_PATH)/*.pt)) are found.)
> $(info Begin downloading weights from HuggingFace ...)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/vqvae_1m.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/vqvae_2m.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/ssp_1m_base.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/ssp_1m_large.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/ssp_2m_base.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/ssp_2m_large.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/unitable_large_structure.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/unitable_large_bbox.pt -P $(WEIGHTS_PATH)
> wget -c https://huggingface.co/poloclub/UniTable/resolve/main/unitable_large_content.pt -P $(WEIGHTS_PATH)
> $(info Completed!)
endif

#
# Python Targets
#
include CONFIG.mk
SRC := src
BEST_MODEL = "../$(word 1,$(subst -, ,$*))/model/best.pt"
RESULT_JSON := html.json
TEDS_STRUCTURE = -f "../experiments/$*/$(RESULT_JSON)" -s

######################
NGPU := 1  # number of gpus used in the experiments

.SECONDARY:

# vq-vae and self-supervised pretraining
experiments/%/.done_pretrain:
> @echo "Using experiment configurations from variable EXP_$*"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train"
> touch $@

# finetuning from SSP weights for table structure, cell bbox and cell content
experiments/%/.done_finetune:
> @echo "Finetuning phase 1 - using experiment configurations from variable EXP_$*"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train"
> @echo "Finetuning phase 2 - starting from epoch 4"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train" ++trainer.trainer.snapshot="epoch3_snapshot.pt" ++trainer.trainer.beit_pretrained_weights=null
> touch $@