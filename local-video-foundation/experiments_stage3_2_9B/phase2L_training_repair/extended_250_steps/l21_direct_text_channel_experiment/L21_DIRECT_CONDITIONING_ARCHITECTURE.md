# L.21 Architecture

conv_in expanded 3->515 channels (3 RGB + 64pos x 8feat token-projection). TokenProj Linear(256->8) per token, mask-zeroed, flattened, broadcast spatially, concatenated to x_t. Legacy cross-attn fed zero text_emb (constant contribution). Token proj std 0.02, new conv weights std 0.02.
