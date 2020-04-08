#ULMFiT attention

This package implements a **"Branching Attention"** classifier head (described [here](https://towardsdatascience.com/explainable-data-efficient-text-classification-888cc7a1af05?source=friends_link&sk=284db458af96fe4f5eee2a0d731384b5%27)).

It is meant to be an improvement of the ULMFiT text classification algorithm - increasing
**data efficiency**, as well as adding some degree of **explainability** (see 
[demo](https://ulmfit.purecode.pl) and the screencast below).

![Attention visualization screencast](https://github.com/tpietruszka/ulmfit_attention/raw/master/docs/figures/screencast.gif)

The package also contains a complete environment necessary to test different 
configurations of the architecture (+baseline) on the IMDb movie review dataset.

It is based on the [hyperspace_explorer](https://github.com/tpietruszka/hyperspace_explorer) 
framework.

### Installation, setup
To install "as an app" - to reproduce results, run experiments, etc:

Requirements: 
- a CUDA-enabled GPU, recent enough to run PyTorch on, min 8GB memory
- NVidia drivers installed (CUDA not necessary) 
- Conda


Warning: by default some data and a pre-trained encoder will be placed in
`${HOME}/.fastai/`. This can be changed, but then a `FASTAI_HOME`
variable has to be set at all times.
  
```shell script
git clone https://github.com/tpietruszka/ulmfit_attention.git
cd ulmfit_attention
conda env create -f environment.yml
source activate ulmfit_attention
pip install -e . 

# setup data, pretrained encoder
python -c "from fastai.text import untar_data, URLs; untar_data(URLs.IMDB)"
IMDB=${HOME}/.fastai/data/imdb
wget -O ${IMDB}/itos.pkl https://static.purecode.pl/ulmfit_attention/imdb/itos.pkl
mkdir ${IMDB}/models
wget -O ${IMDB}/models/fwd_enc.pth https://static.purecode.pl/ulmfit_attention/imdb/fwd_enc.pth
```

To test the install (including training a model on IMDB) run:
 `pytest --runslow`

In order to use the PyTorch modules defined here, this package can be also installed
using pip, directly from github  

### Usage

The main usage mode is to start a worker, which will process queued runs
from a MongoDB-based queue and store the results there. To do that, go to the
inner `ulmfit_attention` and run:

`hyperspace_worker.py ../tasks/ ulmfit_attention`

For other usage modes - including interactive experimentation in Jupyter, see
[hyperspace_explorer](https://github.com/tpietruszka/hyperspace_explorer).


