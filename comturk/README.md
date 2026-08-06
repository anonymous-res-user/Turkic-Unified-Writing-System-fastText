By running mergeshuffle.py, this script merges and shuffles the punctuation reserved and punctuation cleaned text files of each language to a single merged text file named `comturk_mergeshuffledata.txt`.

By running fasttext_pretrain.py, this script trains a fastText using the data `comturk_mergeshuffledata.txt`. The resulting fastText model is the CTS one used in our experiment. The hyperparameters of training are in this script.
