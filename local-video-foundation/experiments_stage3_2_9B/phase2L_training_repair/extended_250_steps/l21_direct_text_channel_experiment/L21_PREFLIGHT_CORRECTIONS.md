# L.21 Preflight Corrections (post-hoc verification)

Applied after L.21 main run; all invariants verified on the same code path.

## Baselines (4-way, fixed input/timestep)
- Original Stage2.1 native cross-attn vs L21-zero-text: mean|d|=0.061 (expected - different conditioning route)
- L21 zero vs correct text: 0.056 (text channels causally active)
- Correct vs wrong caption: 0.049 (caption discrimination present but modest)
- Native vs correct: 0.089

## Bit-identity
- conv_in RGB weights + bias: BIT-IDENTICAL to Stage2.1 (verified torch.equal)
- Text encoder: bit-identical across rebuilds (te_bit_identical=true)

## Gradient audit (single train step, loss=0.0188)
- token_proj grad norm: 0.340 (NONZERO - conditioning path trains)
- conv_in text-slice grad: 0.319 vs RGB-slice grad: 0.055 (text weights get ~6x gradient of RGB slice)
- other UNet params: 3.96 (bulk of learning remains spatial)

## Ablation
- correct vs zeroed channels pred diff: 0.056 - zeroing removes conditioning contribution as required.

## Verdict
All preflight invariants PASS: finite non-negligible gradients through both direct conditioning components (token_proj 0.34, conv text-slice 0.32); RGB weights preserved; TE frozen bit-identical; Arm A/B matched init/data/noise. The 750-step run satisfied these preconditions (same code path).
