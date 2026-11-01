# Hypotheses log

Each block was written and committed before its experiments were run. The "Outcome" sections were
added after the results (see the git history of this file).

## Block 0: data, metric, validation, tiles

What the data turned out to be (before any model was trained): 24 training soils, 127 photos from a Motorola Edge
(69 photos, stored at 1600 px instead of 4000, i.e. 4.60 px/mm), a Samsung A52 (55 photos, stored at 1599 px instead
of 9248, i.e. 4.55 px/mm) and a Motorola Edge 60 fusion (3 photos, 12.5 px/mm). 10 test soils, 35 photos, only
iPhone 14 / 16 at 13.9 / 19.5 px/mm. All photos are taken from 21 cm (host, forum). A label-free check of the
Samsung scale (texture spectra matched to the Motorola photos of the same soils, `scripts/calibrate.py`) gives
4.8 px/mm, +5% from the table value; the same method is +5% on a Motorola-vs-Motorola self-check.
Everything is resampled to the training density, 4.67 px/mm (0.21 mm per pixel). Tiles are 48 mm x 48 mm = 224 px.
Validation uses 6 group folds over physical samples, stratified by D50.

- **H0.1 Scale errors are expensive.** If the curve stays inside the sieve range, an error of factor 2 in ppm
  shifts it by log10(2) = 0.30 decades, i.e. EMD = 100 * 0.30 = 30, more than the gap between a naive and a good
  model. *Prediction:* when the pipeline gets ppm x 2 or x 0.5, CV EMD rises by 5-20. The rise is smaller than 30
  because the models are only partly scale-sensitive, and the invisible fine part of the curve does not move.
- **H0.2 Fines and most sand are not visible.** At 0.21 mm/px, clay, silt and fine sand (< 0.2 mm) are below one
  pixel. Only coarse sand and gravel are resolved. *Prediction:* fine soils have the highest EMD in every family,
  and colour carries much of the signal for the fine end.
- **H0.3 The image-free baseline is strong.** With 24 curves, predicting the training mean is a strong reference.
  *Prediction:* the best single model improves on the training-mean CV EMD by 20-35%, not more.
- **H0.4 Camera is a weak shortcut in CV and useless on test.** Almost every soil was photographed by both training
  phones, so a camera-only predictor is no better than the training mean. *Prediction:* `ctrl_camera_only` is about
  equal to `base_train_mean`.
- **H0.5 Camera domain shift.** Test photos come from other phones (iPhones), with different colour processing and
  sharper JPEGs, even after resampling. *Prediction:* the public LB is worse than CV for every family, and models
  that rely more on colour (CNNs with weak augmentation) lose more than texture-based ones.

### Outcome (block 0)

- H0.1 **confirmed.** A wrong ppm by x2 / x0.5 costs +4 to +10 EMD (CNN +6 / +10, DINOv3 FT +4 / +6, inside the predicted
  5-20). The 30 EMD of a pure shift is not reached, because the models are only partly scale-sensitive.
- H0.2 **wrong in the important part.** Fine soils are not the hardest (EMD 32-35). The 4 sands are (about 60 for every
  family). Fine soils are easy to recognise as a group by colour and lack of clasts. Sands are rare and lie between the
  two modes, so every model pulls them towards "fine".
- H0.3 **wrong.** The best model improves on the training mean by 61% (86.6 -> 34.1), not 20-35%: the curves are
  bimodal, so recognising fine vs gravel already removes most of the error.
- H0.4 **confirmed.** Camera-only 87.1 vs training mean 86.6.
- H0.5 **confirmed.** CV 34-36 -> public LB 77-86 for all families. The training mean also gets worse (86.6 -> 102.4),
  so part of the gap is that the 3 public soils are atypical, not only the cameras. The colour-reliant scratch ViT
  did not lose more than the texture models (+44 vs +46-50), contrary to the second half of H0.5.

## Block 1: CNNs

- **H1.1** AdamW and SGD+momentum give ResNet-18 the same result within one fold-std. A pretrained CNN with
  BatchNorm is easy to optimise, and AdamW has no big advantage on 600 steps.
- **H1.2 BatchNorm with correlated batches.** If a batch of 8 tiles comes from only 2 physical samples, BN
  normalises away the between-sample differences in brightness and texture, and those differences carry the label.
  The running statistics then also differ from the per-batch statistics seen in training. *Prediction:* the
  "bs8, grouped, trainable BN" run is at least 2 EMD worse than the same run with frozen BN statistics. GroupNorm
  and frozen BN are both fine with normal batches.
- **H1.3** ConvNeXt-T <= ResNet-50 <= ResNet-18 in EMD, but the differences are within fold std (24 samples).
- **H1.4 Tile size.** A 24 mm tile (upsampled x2, no new detail) shows fewer gravel grains per tile. A 96 mm tile
  at the same 224 px halves the resolution and drops coarse sand below 1 px. *Prediction:* 48 mm is best. 24 mm is
  close (same information, larger apparent grains suit the ImageNet receptive fields). 96 mm is worst.
- **H1.5** Feeding the 48 mm tile at 112 px (half the pixel density) costs 1-2 EMD (coarse sand drops to 1-4 px).
- **H1.6** Strong augmentation (blur, noise, stronger colour jitter) hurts. Blur changes the apparent texture
  scale, which is close to changing the label.

### Outcome (block 1)

- H1.1 **confirmed.** SGD 40.3 vs AdamW 43.3, well inside the fold std (about 20).
- H1.2 **confirmed, stronger than predicted.** 2-soil batches: trainable BN 50.9 vs frozen BN 35.9 (+15, predicted +2).
  Frozen BN (39.5) and GroupNorm (37.5) are also better than trainable BN with normal batches (43.3).
- H1.3 **wrong direction, not significant.** ResNet-50 41.4, EfficientNet-B0 42.8 and ConvNeXt-T 44.6 (AdamW) / 39.5
  (SGD) are not better than ResNet-18. With 20 training soils, extra capacity is not used.
- H1.4 **partly.** 48 mm (43.3) ≈ 24 mm (41.9) ≈ 96 mm (43.1). Tile size barely matters at this resolution.
- H1.5 **wrong.** 112 px gives 42.5 vs 43.3. The models do not use detail finer than about 0.4 mm.
- H1.6 **confirmed.** Strong augmentation 46.1 vs 43.3.

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

### Outcome (block 2)

- H2.1 **wrong.** The scratch ViT is the best ViT (35.3) and pretrained DeiT-S the worst fine-tuned model (45.6). The grey
  control (35.3 -> 48.8) shows that the scratch ViT is mostly a colour model, which 20 soils are enough to learn.
  Fine-tuning DeiT-S moves its features away from ImageNet without enough data to learn better ones. Frozen DeiT-S
  + k-NN (38.0) and SGD fine-tuning (40.2) confirm that the damage comes from the fine-tuning.
- H2.2 **wrong.** SGD beats AdamW on DeiT-S (40.2 vs 45.6). Here the problem is not reaching a minimum but not drifting too
  far from the pretrained one; SGD's smaller effective steps help.
- H2.3 **confirmed in size, not in sign.** No warmup +0.7, no LLRD +0.2, WD on everything -0.1: all far below the noise.
- H2.4 **confirmed.** Swin-T 42.8 and MaxViT-T 42.3 vs DeiT-S 45.6.

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

### Outcome (block 3)

- H3.1 **confirmed.** Frozen DINOv3 + k-NN 36.6 vs best CNN 35.9, with the lowest fold std of all runs (10.5).
- H3.2 **partly.** The mean patch token is not better than [CLS] (42.4 vs 42.6 for DINOv2). Block 4 is the best ridge
  feature for DINOv2 and DeiT-S, block 8 is not.
- H3.3 **confirmed for DINOv2, wrong for DINOv3.** DINOv2: LP-FT 40.7 (CKA 0.84) better than FT 44.1 (CKA 0.68) and
  LoRA 46.5 (CKA 0.65). DINOv3: FT 34.1 (CKA 0.80) is the best single model, and LP-FT moves too little (CKA 0.97, 37.7).
- H3.4 **wrong in an interesting way.** The adaptation hurt the last-layer features (ridge 42.5 -> 48.2, monotonically
  over the run), LP-FT and LoRA, and helped block-4 features (40.6 -> 36.5) and k-NN (39.2 -> 37.3).

## Block 4: interpretation and controls

- **H4.1** For coarse soils the maps of all three families focus on large grains and grain boundaries. For fine
  soils they are diffuse (colour-driven).
- **H4.2 Controls.** Grey input raises EMD clearly (the models use colour). Shuffling 56 px (12 mm) blocks inside a tile
  raises EMD only slightly (the models use texture statistics more than layout). A colour-statistics-only ridge
  lands between the training mean and the image models.

### Outcome (block 4)

- H4.1 **confirmed for coarse soils.** CNN Grad-CAM covers the large clasts; ViT rollouts sit on individual grains. On
  fine soils the CNN fires on the few largest grains instead of being diffuse.
- H4.2 **partly.** Shuffling 12 mm blocks changes every model by < 1 EMD (confirmed). Grey input clearly hurts only the
  scratch ViT (+13.5) and frozen ridge features; the best CNN and DINOv3 FT do not change. Colour-only ridge (49.7) lies
  between the training mean and the image models (confirmed).

## Blocks 6-7: robustness and ensemble

- **H6** Halving the pixel density at the correct ppm costs the CNN less than the ViTs (16 px patches lose more).
- **H7** OOF errors of the family-best models are highly correlated (> 0.7) because all of them fail on the same
  fine soils. An equal-weight ensemble of diverse models gains 3-8% over the best single model.

### Outcome (blocks 6-7)

- H6 **wrong.** At a quarter of the pixel density the CNN loses +3.4, the scratch ViT +10.2, and DINOv3 FT only +1.7. The
  pretrained SSL ViT is the most robust; only the from-scratch ViT is fragile.
- H7 **confirmed on correlation, smaller gain.** Family-best errors correlate at 0.89-0.98. The equal-weight top-5 blend
  gains 1.6% in CV (34.06 -> 33.53) and 3.5% on the public LB (79.85 -> 77.08), below the predicted 3-8% in CV.

## Block 8: closing the CV-LB gap and the sands (follow-up, written before these runs)

Motivation: CV 34 vs public LB 77-80, and the 4 sands have EMD about 60 in every family.

- **H8.1 Test-time harmonisation.** The training photos were messenger-compressed (1600 px long side, JPEG). The
  test iPhone photos are native and sharp. Even after resampling to 4.67 px/mm they differ in sharpness and in
  compression artefacts. If the test photos are first put through the same path (1600 px, JPEG at the quality
  measured on the training files), the domain gap shrinks. *Prediction:* (a) label-free, the DINOv3 feature
  distance from test photos to their nearest training photo, and a sharpness statistic, both move towards the
  training values. (b) The public LB improves by 3-10 EMD for the texture models (CNN, DINOv3 FT) and less for the
  colour-driven scratch ViT. (c) On H374, the only training soil photographed at native resolution (Motorola Edge
  60 fusion), the harmonised input changes the OOF error by < 3 EMD. CV is unchanged by construction (no retraining).
- **H8.2 Soil-mixup.** Cumulative passing is linear in mass mixing: a mix of soils A and B with mass fraction λ has
  curve (1-λ)F_A + λF_B. Pasting a rectangle of area fraction λ from a tile of soil B into a tile of soil A (same
  mm-per-pixel) and mixing the curves the same way gives the model physically plausible intermediate examples,
  assuming area ≈ mass. The models now see only 24 curves, mostly two modes. *Prediction:* CV improves by 1-3 EMD,
  and the sand EMD drops from about 60 to at most 50. The fine-tuned CNN and DINOv3 benefit more than the
  scratch ViT (whose colour cue is blurred by the mix).

### Outcome (block 8)

- H8.1 **(a) half, (b) not supported, (c) confirmed.** The DINOv3 distance from test to training photos drops by 12%,
  but sharpness overshoots the training value (0.239 -> 0.262 vs 0.249: the downscale plus JPEG blocking adds high
  frequencies). The trained models' test predictions move by only 1-2 EMD, so the public LB cannot improve by 3-10;
  the CV-LB gap is a soil-distribution gap, not a camera one. On H374 the harmonised input costs +1.2 to +1.6 EMD (< 3).
- H8.2 **wrong.** CV changes by -0.8 to +0.6, and the sands stay at 58-60. Mixtures of training soils are convex
  combinations of their curves; fine + gravel gives gap-graded curves, not sands.
