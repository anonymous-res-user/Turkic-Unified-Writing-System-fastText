By running mergeshuffle.py, this script merges and shuffles the punctuation reserved and punctuation cleaned text files of each language to a single merged text file named `uroman_mergeshuffledata.txt`.

By running fasttext_pretrain.py, this script trains a fastText using the data `uroman_mergeshuffledata.txt`. The resulting fastText model is the uroman one used in our experiment. The hyperparameters of training are in this script.

For training the uroman WikiANN NER model, please run `GITHUBall_wikiann_mean_train.py`. Hyperparameters are all in this script.
