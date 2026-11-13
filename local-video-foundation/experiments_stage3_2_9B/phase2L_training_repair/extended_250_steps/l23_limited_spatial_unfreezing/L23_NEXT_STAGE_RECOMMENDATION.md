# L.23 Next Stage Recommendation

Split-conv parity EXACT (0.0); frozen-param integrity PERFECT (0 violations both arms - L22 decay bug fixed). Gain sweep: sensitivity scales with gain (G4 swap 0.054) but is amplification not semantics.

Arm A cond-only: mse 0.175->0.297. Arm B +first-res-block: mse 0.175->0.291 - NO advantage over A; caption discrimination unchanged (swap ~0.011, cos 0.9999). First-block plasticity did not amplify conditioning into semantic output.

Conclusion: conditioning signal (~5-10% rel feature diff) is too weak for any frozen/mostly-frozen configuration. Eight stages of conditioning work (L17-L23) have eliminated every architectural explanation; the model needs either (a) substantially more end-to-end training budget, or (b) a fundamentally stronger conditioning signal.

Recommended L.24: honest reassessment - either commit to a real end-to-end spatial training run at meaningful scale (10k+ steps, full UNet unfrozen, direct text channels), or accept that this architecture+data combination cannot produce prompt-conditioned generation on CPU budget and document the ceiling. Do NOT run another small probe.
