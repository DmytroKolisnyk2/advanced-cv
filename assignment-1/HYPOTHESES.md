# Hypotheses log

Each block was written and committed before its experiments were run. The "Outcome" sections were
added after the results (see the git history of this file).

## Block 0: metric, validation, tiles

Setup: 24 training samples, several photos each, 6 group folds (a physical sample is never in both train and
validation, folds stratified by D50), tiles of a fixed 20 mm x 20 mm resampled to 224 px (11.2 px/mm).

- **H0.1 Scale errors are expensive.** If the whole curve stays inside the sieve range, an error of factor 2 in
  pixels-per-mm shifts it by log10(2) = 0.30 decades, i.e. EMD = 100 * 0.30 = 30. That is more than the gap between
  a naive and a good model. *Prediction:* when the pipeline gets ppm x 2 or x 0.5, CV EMD rises by 5-20. The rise
  is smaller than 30 because the models are only partly scale-sensitive, and the invisible fine part of the curve
  does not move.
- **H0.2 The fine fractions are not visible.** At 11.2 px/mm one pixel is 0.09 mm, so clay and silt (< 0.063 mm)
  are below the pixel size. They can only be inferred from colour and how the surface looks. *Prediction:* fine
  soils get the highest EMD in every family.
- **H0.3 The image-free baseline is strong.** With 24 curves, predicting the training mean is a strong reference.
  *Prediction:* the best single model improves on the training-mean EMD by 20-35%, not more.
- **H0.4 Camera is not a shortcut.** *Prediction:* a camera/ppm-only predictor is no better than the training mean.
  If it is clearly better, the camera correlates with the soil type, and image models could learn the camera.

## Block 1: CNNs

- **H1.1** AdamW and SGD+momentum give ResNet-18 the same result within one fold-std. A pretrained CNN with
  BatchNorm is easy to optimise, and AdamW has no big advantage on 600 steps.
- **H1.2 BatchNorm with correlated batches.** If a batch of 8 tiles comes from only 2 physical samples, BN
  normalises away the between-sample differences in brightness and texture, and those differences carry the label.
  The running statistics then also differ from the per-batch statistics seen in training. *Prediction:* the
  "bs8, grouped, trainable BN" run is at least 2 EMD worse than the same run with frozen BN statistics. GroupNorm
  and frozen BN are both fine with normal batches.
- **H1.3** ConvNeXt-T <= ResNet-50 <= ResNet-18 in EMD, but the differences are within fold std (24 samples).
- **H1.4 Tile size.** 10 mm tiles lose the context of gravel grains (2-63 mm). 40 mm tiles at the same 224 px
  have half the resolution and lose sand detail. *Prediction:* 20 mm is best or ties with 40 mm.
- **H1.5** Feeding the 20 mm tile at 112 px (half the pixel density) costs 1-2 EMD (sand at 0.2-2 mm becomes 1-10 px).
- **H1.6** Strong augmentation (blur, noise, stronger colour jitter) hurts. Blur changes the apparent texture
  scale, which is close to changing the label.

## Block 2: ViTs

- **H2.1** A ViT trained from scratch on about 20 samples reaches about the training-mean baseline. It has no
  locality or translation prior and too little data to learn them. A pretrained DeiT-S is much better but not
  better than the pretrained ResNet-18. Pretraining matters more for ViTs, because the CNN gets locality and weight
  sharing for free and the ViT has to learn them from data.
- **H2.2** SGD fails for DeiT-S (> 3 EMD worse than AdamW), unlike for the CNN. The curvature of a ViT's loss
  differs a lot across parameter groups (attention logits, LayerNorm, embeddings), and a single SGD step size
  cannot fit all of them.
- **H2.3** Removing warmup hurts the ViT more than the CNN. Layer-wise LR decay helps a little (< 1 EMD).
  Weight decay on norms and biases hurts a little.
- **H2.4** Swin-T and MaxViT-T (local attention, convolutional stem in MaxViT) are better than DeiT-S on this
  texture task.

## Block 3: self-supervised encoders

- **H3.1** Frozen DINOv2-S / DINOv3-S features with a ridge head on per-sample mean features come within 1-2 EMD
  of the best fine-tuned CNN. A 24-sample problem favours few trainable degrees of freedom.
- **H3.2** For texture statistics the mean patch token is better than [CLS], and an intermediate block (8 of 12)
  is at least as good as the last one.
- **H3.3** Full fine-tuning is not better than LP-FT or LoRA here. FT starts from a random head, so the
  first gradients distort the features. *Prediction:* CKA(before, after) is lower for FT than for LP-FT.
- **H3.4 Domain adaptation.** Continued DINO on a few thousand unlabelled training tiles gives at most a small
  gain (<= 1 EMD) for frozen features, and may hurt through forgetting. Scale-changing augmentations
  (random-resized crops, zoom, blur) destroy the label; we use only dihedral flips/rotations, mild colour jitter and
  local crops cut without resizing.

## Block 4: interpretation and controls

- **H4.1** For coarse soils the maps of all three families focus on large grains and grain boundaries. For fine
  soils they are diffuse (colour-driven).
- **H4.2 Controls.** Grey input raises EMD clearly (the models use colour). Shuffling 56 px blocks inside a tile
  raises EMD only slightly (the models use texture statistics more than layout). A colour-statistics-only ridge
  lands between the training mean and the image models.

## Blocks 6-7: robustness and ensemble

- **H6** Halving the pixel density at the correct ppm costs the CNN less than the ViTs (16 px patches lose more).
- **H7** OOF errors of the family-best models are highly correlated (> 0.7) because all of them fail on the same
  fine soils. An equal-weight ensemble of diverse models gains 3-8% over the best single model.
