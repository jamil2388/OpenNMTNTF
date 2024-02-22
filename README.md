# OpenNMTNTF
This is a baseline project for Neural Team Formation with OpenNMT


# Project Setup


# Notes on Steps

### Opennmt-py
- Prepare the _source_ and _target_ data for translation in the data folder
- Prepare _config.yaml_ file for run configurations
- Build the vocabulary against the source and target train files (example.vocab.src, example.vocab.tgt)
- ```onmt_build_vocab -config toy_en_de.yaml -n_sample 10``` creates the vocab files by only taking _10_ rows (10 samples) each from the train files
- The vocabulary will be the count of the distinct words within the _selected number of samples_

### _data_ folder

- The data folder must contain all forms of data
- _preprocessed_ : contains the preprocessed _src_ and _tgt_ corpus for translation from sparse matrix data from opentf
- _raw_ : contains any form of raw data ready to be preprocessed (example : sparse matrix from dblp data)
- _input_ : the train test valid splits of the preprocessed data to be fed for translation
- _output_ : 1) the generated vocabulary of the model in _vocabs_ folder, 2) the generated models in the _models_ folder and 3) the generated translations in the _translation_ folder
- 