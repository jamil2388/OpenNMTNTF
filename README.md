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
- 