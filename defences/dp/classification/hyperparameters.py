# ===========
# BoE
# ===========

class HP_IMDB_BOE:
    batch_size = 64
    learning_rate = 1e-3
    learning_rate_dpsgd = 1e-3
    patience = 5
    tgt_class = 1
    sequence_length = 512


class HP_DBPedia_BOE:
    batch_size = 256
    learning_rate = 1e-3
    learning_rate_dpsgd = 1e-3
    patience = 2
    tgt_class = 1   # start from 0
    sequence_length = 256


class HP_Trec50_BOE:
    batch_size = 128
    learning_rate = 5e-4
    learning_rate_dpsgd = 5e-4
    patience = 10
    tgt_class = 32  # start from 0
    sequence_length = 128


class HP_Trec6_BOE:
    batch_size = 16
    learning_rate = 1e-4
    learning_rate_dpsgd = 1e-4
    patience = 10
    tgt_class = 1  # start from 0
    sequence_length = 128

# ===========
# CNN
# ===========

class HP_IMDB_CNN:
    batch_size = 64
    learning_rate = 1e-3
    learning_rate_dpsgd = 1e-3
    patience = 5
    tgt_class = 1
    sequence_length = 512


class HP_DBPedia_CNN:
    batch_size = 32
    learning_rate = 1e-3
    learning_rate_dpsgd = 1e-3
    patience = 2
    tgt_class = 1   # start from 0
    sequence_length = 256


class HP_Trec50_CNN:
    batch_size = 128
    learning_rate = 5e-4
    learning_rate_dpsgd = 5e-4
    patience = 10
    tgt_class = 32  # start from 0
    sequence_length = 128


class HP_Trec6_CNN:
    batch_size = 16
    learning_rate = 1e-4
    learning_rate_dpsgd = 1e-4
    patience = 10
    tgt_class = 1  # start from 0
    sequence_length = 128

# ===========
# BERT
# ===========

class HP_IMDB_BERT:
    batch_size = 32
    learning_rate = 1e-4
    learning_rate_dpsgd = 1e-4
    patience = 2
    tgt_class = 1
    sequence_length = 512


class HP_DBPedia_BERT:
    batch_size = 32
    learning_rate = 1e-4
    learning_rate_dpsgd = 1e-4
    patience = 1
    tgt_class = 1   # start from 0
    sequence_length = 256


class HP_Trec50_BERT:
    batch_size = 128
    learning_rate = 5e-4
    learning_rate_dpsgd = 5e-4
    patience = 10
    tgt_class = 32  # start from 0
    sequence_length = 128


class HP_Trec6_BERT:
    batch_size = 16
    learning_rate = 1e-4
    learning_rate_dpsgd = 1e-4
    patience = 5
    tgt_class = 1  # start from 0
    sequence_length = 128
