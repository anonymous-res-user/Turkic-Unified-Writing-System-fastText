Multilingual UD UPOS fine-tuning with locally downloaded CANINE-c.

Experiment design
-----------------
- Automatically discovers every treebank directory beside this script that
  contains dataset_dict.json.
- Mixes all available train splits into one training set.
- Mixes all available validation/dev/eval splits into one validation set.
- Evaluates every available test split separately by treebank.
- Uses the original UD token strings; no comturk/uroman conversion is applied.
- Gives the model no language ID and no treebank ID.
- Fine-tunes the full CANINE-c encoder by default.
- Uses mean pooling over the contextual character representations belonging
  to each gold UD token, followed by dropout and a linear UPOS classifier.
- Selects the best model by validation macro-F1.
- Keeps the best trainable parameters only in CPU memory.
- Writes no checkpoint, prediction, cache, metric, or log file.

Expected directory layout
-------------------------
turkic_ud_2_18/
├── all_canine_c_pos_train.py
├── canine-c/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── az_tuecl/
│   ├── dataset_dict.json
│   ├── test/
│   └── ...
└── ...

Notes
-----
CANINE operates on Unicode code points. Each sentence is reconstructed by
joining its gold UD tokens with one ASCII space. The gold token boundaries are
used only to pool character-level representations into one vector per token.
