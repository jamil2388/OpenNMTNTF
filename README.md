# OpenNMTNTF
This is a baseline project for Neural Team Formation with OpenNMT


# Project Setup
```
git clone https://github.com/jamil2388/OpenNMTNTF.git
cd OpenNMTNTF
pip install -r requirements.txt
```
OpenNMT by default installs _pytorch_ _cpu_ version. In case of gpu usage, install the appropriate torch by running <br>
```
pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # sample command for cuda 11.8 compatible GPU
```

# Notes on Steps

### Opennmt-py Test Run

- Prepare the _source_ and _target_ data for translation in the data folder
- Prepare _config.yaml_ file for run configurations
- In case of GPU usage, edit the config file and enable the options "_world_size_" and "_gpu_ranks_" as the total number of gpus and id of the gpu device to use respectively
- Build the vocabulary against the source and target train files (example.vocab.src, example.vocab.tgt)
- ```onmt_build_vocab -config toy_en_de.yaml -n_sample 10``` creates the vocab files by only taking _10_ rows (10 samples) each from the train files
- The vocabulary will be the count of the distinct words within the _selected number of samples_


### Structure of "_data_" folder

- The _data_ folder must contain all forms of data
- _preprocessed_ : contains the preprocessed _src_ and _tgt_ corpus for translation from sparse matrix data from opentf
- _raw_ : contains any form of raw data ready to be preprocessed (example : sparse matrix from dblp data)
- _input_ : the train test valid splits of the preprocessed data to be fed for translation
- _output_ : <br>
  1) the generated vocabulary of the model in _vocabs_ folder, 
  2) the generated models in the _models_ folder and 
  3) the generated translations in the _translation_ folder

### Run Commands for Sample data (dblp) 

```
# This scenario is for dblp/toy.dblp.v12.json data
# preprocess the data
python main.py

# build vocabulary
onmt_build_vocab -config config.yaml -n_sample -1 

# train the data
onmt_train -config config.yaml

# predict the translations
onmt_translate -model data/output/models/dblp/toy.dblp.v12.json/_step_100.pt -src data/input/dblp/toy.dblp.v12.json/src_test.txt -output data/output/translations/toy.dblp.v12.json.pred_100.txt -verbose
```