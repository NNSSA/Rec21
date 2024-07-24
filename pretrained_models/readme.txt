Both models were trained on 100k boxes for 5 epochs, see paper for details.


Parameters 43M model:

in_channels = 1,
out_channels = 1,
num_blocks = 4,
out_channels_first_layer = 48,
embedding_dim = 48,
cond_embed_pref = 4,
dropout = 0.,
normalization = "instance",
preactivation = True,
padding = 1



Parameters 11M model:

in_channels = 1,
out_channels = 1,
num_blocks = 4,
out_channels_first_layer = 24,
embedding_dim = 48,
cond_embed_pref = 4,
dropout = 0.,
normalization = "instance",
preactivation = True,
padding = 1