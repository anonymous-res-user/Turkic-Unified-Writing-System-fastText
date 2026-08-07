By running each {iso639-3}\_cta.py, 
you can get text files uroman\_{iso639-1}.txt and comturk\_{iso639-1}.txt

These two files are CTA and uroman transliterated text files of Wikipedia edition for each language (with transliteration, UAX#29 tokenization, and space joining).

Then, by running to\_ft\_train.py,
you can get the line concatenation without punctuation cleaning version and the line concatenation with punctuation cleaning version text files of each uroman\_{iso639-1}.txt (keep the *uroman* string in line 6 and line 221. These processed result files are stroed on the folder `uroman`.) or comturk\_{iso639-1}.txt (change the *uroman* string in line 6 and line 221 to *comturk*. These processed result files are stroed on the folder `comturk`.)

For the following steps, please refer to the READMEs in the files `comturk` and `uroman`.

**‼️⚠️All WikiANN NER and UD POS training freezes the fastText embeddings.⚠️‼️** For training details and hyperparameters, please refer to the READEMEs and training scripts in the folders `comturk` and `uroman`.

**The fastText models and the fastText training corpora are too large. We have some trouble uplaoding them. However, you can still reproduce our work through the scripts provided here. We are trying our best to solve this problem.**

For the CANINE training script, please refer to the script in the folder `turkic_ud_2_18`. The data and path requirement for running this script are also in this script.
