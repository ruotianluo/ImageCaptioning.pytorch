import torch
trans = torch.load('log_trans/model.pth')
bart = torch.load('log_bart/model.pth')

mappings = [
    ('q_proj', 'linears.0'),
    ('k_proj', 'linears.1'),
    ('v_proj', 'linears.2'),
    ('out_proj', 'linears.3'),
    ('fc1', 'feed_forward.w_1'),
    ('fc2', 'feed_forward.w_2'),
    ('self_attn_layer_norm.weight', 'sublayer.0.norm.a_2'),
    ('self_attn_layer_norm.bias', 'sublayer.0.norm.b_2'),
    ('encoder_attn_layer_norm.weight', 'sublayer.1.norm.a_2'),
    ('encoder_attn_layer_norm.bias', 'sublayer.1.norm.b_2'),
    ('encoder_attn', 'src_attn'),
    ('layernorm_embedding.weight', 'norm.a_2'),
    ('layernorm_embedding.bias', 'norm.b_2'),
    ('layer_norm.weight', 'norm.a_2'),
    ('layer_norm.bias', 'norm.b_2'),
]

import copy
new_bart = copy.deepcopy(bart)
for k in set(bart.keys()).intersection(set(trans.keys())):
    new_bart[k].copy_(trans[k])
    del bart[k]
    del trans[k]

# for k in list(bart.keys()):
#     kk = k.replace('bart.', '')
#     for mapping in mappings:
#         kk = kk.replace(mapping[0], mapping[1])
#     if kk in trans:
#         assert bart[k].shape == trans[kk].shape
#         del bart[k]
#         del trans[kk]

for k in list(bart.keys()):
    kk = k.replace('bart.', '')
    for mapping in mappings:
        kk = kk.replace(mapping[0], mapping[1])
    if kk in trans:
        new_bart[k].copy_(trans[kk])
        assert bart[k].shape == trans[kk].shape
        del bart[k]
        del trans[kk]

mappings = [
    ('encoder', 'final_layer_norm.weight', 'sublayer.1.norm.a_2'),
    ('encoder', 'final_layer_norm.bias', 'sublayer.1.norm.b_2'),
    ('decoder', 'final_layer_norm.weight', 'sublayer.2.norm.a_2'),
    ('decoder', 'final_layer_norm.bias', 'sublayer.2.norm.b_2'),
]


for k in list(bart.keys()):
    kk = k.replace('bart.', '')
    for mapping in mappings:
        if mapping[0] in k:
            kk = kk.replace(mapping[1], mapping[2])
    if kk in trans:
        new_bart[k].copy_(trans[kk])
        assert bart[k].shape == trans[kk].shape
        del bart[k]
        del trans[kk]

new_bart['model.bart.shared.weight'][1:].copy_(trans['model.tgt_embed.0.lut.weight'])
new_bart['model.bart.decoder.embed_tokens.weight']
new_bart['model.bart.decoder.embed_positions.weight'].copy_(trans['model.tgt_embed.1.pe'][0])

torch.save(new_bart, 'log_bart/model.pth')