# Localizing and repairing the fc4 backdoor via singular value decomposition

Approach summary:

I identified the poisoned layer by testing each hidden layer's causal effect on the model's prediction bias, rather than relying on raw weight magnitudes (which turned out to be misleading — see below).

    Weight inspection: fc4 had a standout weight (max abs = 5.0, vs 0.05-0.2 for other layers) that looked like an obvious candidate.

    Causal test: zeroing fc4's single largest weight (and even the top 20) barely moved the model's bias at all — this weight is a decoy.

    SVD analysis: computing the singular value spectrum of each candidate layer's weight matrix (fc2-fc6, since only these are square 56x56 matching the submission format), fc4 showed the most extreme top1/top2 singular value ratio (47.4).

    Causal confirmation: removing each layer's top singular component entirely and re-running the model, fc4 was the only layer where bias collapsed substantially (from ~0.0032 to ~0.0007) while NSE held steady (0.682 vs 0.680 baseline). Every other layer either did nothing meaningful or badly damaged NSE (fc2/fc3/fc6 went negative).

    Repair: rather than fully zeroing fc4's top singular component (which produced too large an edit relative to the layer's normal weight scale), I swept a scaling factor (alpha) from 0.1 to 1.0 on that component's removal. NSE peaked at alpha=0.5 (0.688, slightly better than baseline), with a meaningful bias reduction (~39%) and a moderate edit magnitude. I selected alpha=0.5 as the best tradeoff between leak removal, skill preservation, and edit size.

    Trigger direction: I used the right-singular vector of fc4's top singular component (Vt[0]) as my estimate of the poisoned feature direction.

What didn't work: correlating layer activations directly against the prediction residual (both via mean-difference between high/low-residual timesteps, and via ridge regression) gave weak, flat signal across all layers (R^2 under 0.05 everywhere) — the poison's effect is too small relative to the model's baseline prediction noise (NSE 0.68) for this to isolate it. The SVD + causal ablation approach was far more decisive.

Limitations: without access to the true clean model, I can't directly verify the recovered direction or repaired weights — alpha=0.5 was chosen from the bias/NSE tradeoff observed empirically, not from known ground truth on the poison's magnitude.