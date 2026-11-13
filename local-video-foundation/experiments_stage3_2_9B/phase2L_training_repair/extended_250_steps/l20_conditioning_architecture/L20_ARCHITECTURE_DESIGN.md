# L.20 Architecture Design

Arm A (LearnedQueryFiLM): 8 learned queries [8,256] -> K/V projections of text_emb -> masked softmax attention -> concat contexts [2048] -> MLP -> per-ConditionalResidualBlock FiLM heads (zero-init gamma/beta). Post-hooks modulate block outputs.
Arm B (PooledFiLM): masked-mean pooled text_emb -> MLP -> same per-block FiLM heads.
Both: UNet + FiLM trainable, TE frozen, identical data/order/seeds/lr.
