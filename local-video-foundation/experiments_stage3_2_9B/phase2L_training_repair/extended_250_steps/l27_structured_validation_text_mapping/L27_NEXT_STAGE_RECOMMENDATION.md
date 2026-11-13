# L.27 Next Stage Recommendation

Phase A: structured generation at step 10000 - accs {8000:0.792, 8500:0.667, 9000:0.667, 9500:0.917, 10000:0.708} (noisy n=24 evals but consistently >>chance, val_loss 0.0013). VALIDATED as learnable.

Phase B: text->color adapter test_acc 1.0 (300 prompts, 10 template families, held-out templates) - frozen TE retains color info; masked-mean pooling + 2-layer MLP perfectly classifies red/blue/yellow. E2E pipeline works.

BREAKTHROUGH: the full chain prompt->color->circle now works for a constrained task. The original text-conditioning failure was the DIFFUSION conditioning pathway, not text understanding - the frozen encoder + adapter reads color fine.

Recommended L.28: expand the structured task - add shape (circle/square) and position classes to the one-hot condition (e.g., 3 colors x 2 shapes = 6 classes), retrain generator, extend adapter to predict joint labels. This tests whether the architecture scales to richer structured conditioning before attempting open-vocabulary text.
