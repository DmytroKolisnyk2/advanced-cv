# Results

CV = mean ± std over the 6 group folds of the per-sample log-EMD (lower is better). LB = Kaggle public leaderboard (private is hidden until the competition ends).


## Baselines and shortcut controls

| name             | CV (mean ± std)   |   D50_err_dec |   lb_public |   gpu_hours |   peak_mem_gb | setup   |
|:-----------------|:------------------|--------------:|------------:|------------:|--------------:|:--------|
| ctrl_colour_only | 49.689 ± 15.760   |         0.424 |         nan |       0.000 |         0.000 |         |
| base_train_mean  | 86.632 ± 3.037    |         0.923 |         nan |       0.000 |         0.000 |         |
| ctrl_camera_only | 87.055 ± 2.713    |         0.929 |         nan |       0.000 |         0.000 |         |
| base_uniform     | 102.020 ± 5.477   |         0.914 |         nan |       0.000 |         0.000 |         |


## Part 1 — CNNs

| name                         | CV (mean ± std)   |   D50_err_dec |   lb_public |   gpu_hours |   peak_mem_gb | setup                                                                                                    |
|:-----------------------------|:------------------|--------------:|------------:|------------:|--------------:|:---------------------------------------------------------------------------------------------------------|
| cnn_r18_bs8_grouped_frozenbn | 35.878 ± 17.187   |         0.313 |         nan |       0.053 |         3.210 | opt=adamw, lr=0.0003, norm=frozen_bn, aug=basic, sampler=grouped, batch=8, mode=ft, tile_mm=48           |
| cnn_r18_groupnorm            | 37.527 ± 16.186   |         0.330 |         nan |       0.018 |         5.030 | opt=adamw, lr=0.0003, norm=groupnorm, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48        |
| cnn_r18_frozenbn             | 39.497 ± 18.297   |         0.337 |         nan |       0.015 |         3.210 | opt=adamw, lr=0.0003, norm=frozen_bn, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48        |
| cnn_convnext_t_sgd           | 39.525 ± 18.650   |         0.351 |         nan |       0.036 |         4.520 | opt=sgd, lr=0.01, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48              |
| cnn_r18_sgd                  | 40.340 ± 21.109   |         0.363 |         nan |       0.016 |         3.210 | opt=sgd, lr=0.01, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48              |
| cnn_r50_adamw                | 41.354 ± 18.897   |         0.364 |         nan |       0.035 |         3.750 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_r18_noaug                | 41.422 ± 16.618   |         0.373 |         nan |       0.015 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=none, sampler=by_sample, batch=32, mode=ft, tile_mm=48           |
| cnn_r18_tile24mm             | 41.894 ± 19.339   |         0.382 |         nan |       0.015 |         2.360 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=24          |
| cnn_r18_const_nowarm         | 42.205 ± 17.394   |         0.372 |         nan |       0.016 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_r18_img112               | 42.464 ± 19.404   |         0.371 |         nan |       0.015 |         2.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, img=112, tile_mm=48 |
| cnn_effb0_adamw              | 42.798 ± 18.207   |         0.384 |         nan |       0.096 |         3.710 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_r18_tile96mm             | 43.072 ± 17.428   |         0.387 |         nan |       0.015 |         1.200 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=96          |
| cnn_r18_adamw                | 43.280 ± 20.463   |         0.390 |         nan |       0.018 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_convnext_t_adamw         | 44.576 ± 20.304   |         0.398 |         nan |       0.071 |         4.520 | opt=adamw, lr=0.0001, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_r18_strongaug            | 46.087 ± 22.261   |         0.432 |         nan |       0.015 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=strong, sampler=by_sample, batch=32, mode=ft, tile_mm=48         |
| cnn_r18_bs8_grouped          | 50.851 ± 22.334   |         0.481 |         nan |       0.056 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=grouped, batch=8, mode=ft, tile_mm=48             |


## Part 2 — ViTs

| name                 | CV (mean ± std)   |   D50_err_dec |   lb_public |   gpu_hours |   peak_mem_gb | setup                                                                                                                      |
|:---------------------|:------------------|--------------:|------------:|------------:|--------------:|:---------------------------------------------------------------------------------------------------------------------------|
| vit_scratch          | 35.280 ± 17.046   |         0.302 |         nan |       0.038 |         2.530 | opt=adamw, lr=0.0005, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48                            |
| vit_deit_s_sgd       | 40.226 ± 20.866   |         0.351 |         nan |       0.030 |         3.330 | opt=sgd, lr=0.01, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48              |
| vit_maxvit_t         | 42.268 ± 19.321   |         0.381 |         nan |       0.141 |         7.600 | opt=adamw, lr=0.0001, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48                            |
| vit_deit_s_img112    | 42.396 ± 19.926   |         0.384 |         nan |       0.033 |         2.640 | opt=adamw, lr=0.0001, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, img=112, tile_mm=48 |
| vit_swin_t           | 42.764 ± 21.414   |         0.393 |         nan |       0.052 |         5.470 | opt=adamw, lr=0.0001, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| vit_deit_s_strongaug | 44.426 ± 20.994   |         0.398 |         nan |       0.036 |         3.410 | opt=adamw, lr=0.0001, layer_decay=0.75, norm=default, aug=strong, sampler=by_sample, batch=32, mode=ft, tile_mm=48         |
| vit_deit_s_wd_all    | 45.522 ± 20.240   |         0.418 |         nan |       0.031 |         3.410 | opt=adamw, lr=0.0001, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| vit_deit_s           | 45.627 ± 20.508   |         0.419 |         nan |       0.032 |         3.400 | opt=adamw, lr=0.0001, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| vit_deit_s_nollrd    | 45.848 ± 18.826   |         0.427 |         nan |       0.030 |         3.410 | opt=adamw, lr=0.0001, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48                            |
| vit_deit_s_nowarm    | 46.344 ± 20.437   |         0.426 |         nan |       0.033 |         3.410 | opt=adamw, lr=0.0001, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| vit_deit_s_dp0       | 47.675 ± 18.989   |         0.440 |         nan |       0.032 |         3.410 | opt=adamw, lr=0.0001, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |


## Part 3 — self-supervised encoders

| name                       | CV (mean ± std)   |   D50_err_dec |   lb_public |   gpu_hours |   peak_mem_gb | setup                                                                                                                           |     cka |
|:---------------------------|:------------------|--------------:|------------:|------------:|--------------:|:--------------------------------------------------------------------------------------------------------------------------------|--------:|
| ssl_dinov3_s_ft            | 34.059 ± 16.043   |         0.286 |         nan |       0.048 |         3.710 | opt=adamw, lr=5e-05, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48                |   0.802 |
| dinov2_s_da_ridge_block4   | 36.508 ± 12.872   |         0.290 |         nan |       0.001 |         3.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=block4, tile_mm=48    | nan     |
| dinov3_s_knn_clspatch      | 36.564 ± 10.473   |         0.305 |         nan |       0.001 |         3.250 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=knn, features=cls+patch, tile_mm=48   | nan     |
| dinov2_s_da_knn_clspatch   | 37.331 ± 13.412   |         0.321 |         nan |       0.001 |         3.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=knn, features=cls+patch, tile_mm=48   | nan     |
| ssl_dinov3_s_lpft          | 37.693 ± 17.814   |         0.326 |         nan |       0.056 |         3.710 | opt=adamw, lr=2e-05, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=lpft, tile_mm=48              |   0.966 |
| deit_s_knn_clspatch        | 37.966 ± 13.966   |         0.320 |         nan |       0.000 |         3.160 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=knn, features=cls+patch, tile_mm=48   | nan     |
| dinov2_s_knn_clspatch      | 39.227 ± 13.032   |         0.327 |         nan |       0.001 |         3.440 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=knn, features=cls+patch, tile_mm=48   | nan     |
| deit_s_ridge_block4        | 40.588 ± 16.638   |         0.345 |         nan |       0.000 |         3.160 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=block4, tile_mm=48    | nan     |
| dinov2_s_ridge_block4      | 40.606 ± 17.393   |         0.325 |         nan |       0.001 |         3.440 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=block4, tile_mm=48    | nan     |
| ssl_dinov2_s_lpft          | 40.700 ± 20.682   |         0.375 |         nan |       0.043 |         3.840 | opt=adamw, lr=2e-05, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=lpft, tile_mm=48              |   0.839 |
| deit_s_ridge_block8        | 40.860 ± 17.068   |         0.344 |         nan |       0.000 |         3.160 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=block8, tile_mm=48    | nan     |
| deit_s_lp_clspatch         | 41.384 ± 19.258   |         0.354 |         nan |       0.001 |         3.160 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=lp, features=cls+patch, tile_mm=48    | nan     |
| dinov3_s_ridge_cls         | 41.433 ± 15.368   |         0.330 |         nan |       0.001 |         3.250 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=cls, tile_mm=48       | nan     |
| deit_s_ridge_patch         | 41.707 ± 16.169   |         0.334 |         nan |       0.000 |         3.160 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=patch, tile_mm=48     | nan     |
| dinov3_s_ridge_clspatch    | 41.760 ± 16.889   |         0.328 |         nan |       0.001 |         3.250 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=cls+patch, tile_mm=48 | nan     |
| deit_s_ridge_clspatch      | 41.885 ± 17.533   |         0.361 |         nan |       0.000 |         3.160 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=cls+patch, tile_mm=48 | nan     |
| dinov3_s_ridge_patch       | 42.046 ± 17.249   |         0.329 |         nan |       0.001 |         3.250 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=patch, tile_mm=48     | nan     |
| dinov2_s_ridge_patch       | 42.430 ± 15.250   |         0.337 |         nan |       0.001 |         3.440 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=patch, tile_mm=48     | nan     |
| dinov2_s_ridge_clspatch    | 42.466 ± 15.823   |         0.338 |         nan |       0.001 |         3.440 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=cls+patch, tile_mm=48 | nan     |
| dinov2_s_ridge_cls         | 42.631 ± 15.415   |         0.339 |         nan |       0.001 |         3.440 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=cls, tile_mm=48       | nan     |
| dinov3_s_ridge_block4      | 42.876 ± 15.690   |         0.362 |         nan |       0.001 |         3.250 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=block4, tile_mm=48    | nan     |
| dinov2_s_ridge_block8      | 43.041 ± 16.322   |         0.337 |         nan |       0.001 |         3.440 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=block8, tile_mm=48    | nan     |
| dinov3_s_ridge_block8      | 43.481 ± 17.434   |         0.343 |         nan |       0.001 |         3.250 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=block8, tile_mm=48    | nan     |
| ssl_dinov2_s_ft            | 44.089 ± 21.964   |         0.397 |         nan |       0.037 |         3.850 | opt=adamw, lr=5e-05, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48                |   0.683 |
| dinov2_s_lp_clspatch       | 44.283 ± 16.658   |         0.395 |         nan |       0.002 |         3.440 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=lp, features=cls+patch, tile_mm=48    | nan     |
| dinov2_s_da_ridge_block8   | 44.313 ± 15.742   |         0.351 |         nan |       0.001 |         3.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=block8, tile_mm=48    | nan     |
| ssl_dinov3_s_lora          | 44.895 ± 21.064   |         0.429 |         nan |       0.048 |         3.070 | opt=adamw, lr=0.001, norm=default, aug=basic, sampler=by_sample, batch=32, mode=lora, tile_mm=48                                |   0.694 |
| dinov3_s_lp_clspatch       | 45.036 ± 17.564   |         0.395 |         nan |       0.002 |         3.250 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=lp, features=cls+patch, tile_mm=48    | nan     |
| deit_s_ridge_cls           | 45.201 ± 20.007   |         0.391 |         nan |       0.000 |         3.160 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=cls, tile_mm=48       | nan     |
| dinov2_s_da_lp_clspatch    | 45.792 ± 17.431   |         0.404 |         nan |       0.002 |         3.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=lp, features=cls+patch, tile_mm=48    | nan     |
| ssl_dinov2_s_lora          | 46.539 ± 21.161   |         0.441 |         nan |       0.036 |         3.130 | opt=adamw, lr=0.001, norm=default, aug=basic, sampler=by_sample, batch=32, mode=lora, tile_mm=48                                |   0.647 |
| dinov2_s_da_ridge_patch    | 46.943 ± 15.203   |         0.383 |         nan |       0.001 |         3.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=patch, tile_mm=48     | nan     |
| ssl_dinov2_s_da_lpft       | 47.469 ± 19.303   |         0.443 |         nan |       0.044 |         3.850 | opt=adamw, lr=2e-05, layer_decay=0.75, norm=default, aug=basic, sampler=by_sample, batch=32, mode=lpft, tile_mm=48              |   0.868 |
| dinov2_s_da_ridge_clspatch | 48.168 ± 14.346   |         0.404 |         nan |       0.001 |         3.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=cls+patch, tile_mm=48 | nan     |
| ssl_dinov2_s_da_lora       | 48.942 ± 17.979   |         0.463 |         nan |       0.035 |         3.130 | opt=adamw, lr=0.001, norm=default, aug=basic, sampler=by_sample, batch=32, mode=lora, tile_mm=48                                |   0.622 |
| dinov2_s_da_ridge_cls      | 49.502 ± 14.554   |         0.430 |         nan |       0.001 |         3.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, head=ridge, features=cls, tile_mm=48       | nan     |


## Part 7 — ensembles

| name                | CV (mean ± std)   |   D50_err_dec |   lb_public |   gpu_hours |   peak_mem_gb | setup   |
|:--------------------|:------------------|--------------:|------------:|------------:|--------------:|:--------|
| ens_top5            | 33.525 ± 15.809   |         0.286 |         nan |       0.000 |         0.000 |         |
| ens_diverse3        | 33.675 ± 14.055   |         0.283 |         nan |       0.000 |         0.000 |         |
| ens_best_per_family | 33.920 ± 16.921   |         0.296 |         nan |       0.000 |         0.000 |         |
| ens_top3            | 33.920 ± 16.921   |         0.296 |         nan |       0.000 |         0.000 |         |
| ens_diverse2        | 33.927 ± 15.188   |         0.293 |         nan |       0.000 |         0.000 |         |


## Controls and scale robustness (CV EMD)

| name                         |   cv_mean |   ctrl_shuffle |   ctrl_grey |   scale_ref |   res_0.5 |   res_0.25 |   ppm_x0.5 |   ppm_x2 |
|:-----------------------------|----------:|---------------:|------------:|------------:|----------:|-----------:|-----------:|---------:|
| cnn_r18_bs8_grouped_frozenbn |     35.88 |          35.66 |       35.13 |       36.39 |     36.90 |      39.74 |      46.35 |    42.39 |
| cnn_r18_groupnorm            |     37.53 |          37.42 |       42.41 |       37.84 |     38.37 |      41.18 |      41.83 |    39.65 |
| cnn_r18_frozenbn             |     39.50 |          39.23 |       45.61 |       39.50 |     36.59 |      38.97 |      41.64 |    41.19 |
| cnn_convnext_t_sgd           |     39.52 |          39.79 |       43.55 |       39.84 |     39.87 |      41.03 |      39.11 |    39.58 |
| cnn_r18_sgd                  |     40.34 |          40.17 |       39.06 |       40.20 |     38.60 |      51.62 |      47.28 |    39.54 |
| cnn_r50_adamw                |     41.35 |          40.08 |       39.15 |       41.59 |     37.50 |      51.86 |      43.56 |    40.10 |
| cnn_r18_noaug                |     41.42 |          42.22 |       47.22 |       41.33 |     43.47 |      45.05 |      45.22 |    38.54 |
| cnn_r18_const_nowarm         |     42.20 |          42.15 |       45.92 |       42.14 |     46.08 |      52.17 |      42.62 |    38.37 |
| cnn_r18_img112               |     42.46 |          42.64 |       42.94 |       42.16 |     43.78 |      40.88 |      43.19 |    34.47 |
| cnn_effb0_adamw              |     42.80 |          44.06 |       38.53 |       42.83 |     42.55 |      58.79 |      47.76 |    39.46 |
| cnn_r18_adamw                |     43.28 |          43.42 |       44.06 |       43.37 |     43.24 |      42.72 |      43.63 |    36.04 |
| cnn_convnext_t_adamw         |     44.58 |          45.03 |       43.46 |       44.48 |     42.15 |      36.03 |      40.83 |    43.91 |
| cnn_r18_strongaug            |     46.09 |          46.95 |       47.46 |       45.82 |     42.61 |      40.20 |      41.38 |    40.33 |
| cnn_r18_bs8_grouped          |     50.85 |          51.65 |       55.80 |       51.11 |     53.36 |      59.14 |      53.65 |    57.98 |
| ssl_dinov3_s_ft              |     34.06 |          33.77 |       34.15 |       34.19 |     34.38 |      35.87 |      39.81 |    37.93 |
| dinov2_s_da_ridge_block4     |     36.51 |          37.06 |       54.66 |       37.60 |     48.83 |      61.99 |      53.71 |    53.75 |
| dinov3_s_knn_clspatch        |     36.56 |          38.80 |       38.38 |       36.88 |     38.31 |      38.31 |      39.03 |    37.85 |
| dinov2_s_da_knn_clspatch     |     37.33 |          39.30 |       34.76 |       38.29 |     39.98 |      48.82 |      36.70 |    45.30 |
| ssl_dinov3_s_lpft            |     37.69 |          37.87 |       37.74 |       37.72 |     37.51 |      37.31 |      39.98 |    40.57 |
| deit_s_knn_clspatch          |     37.97 |          39.74 |       40.23 |       38.59 |     40.09 |      39.87 |      37.93 |    37.78 |
| dinov2_s_knn_clspatch        |     39.23 |          38.41 |       38.60 |       40.75 |     39.87 |      39.16 |      41.59 |    36.80 |
| deit_s_ridge_block4          |     40.59 |          41.10 |       51.91 |       41.28 |     48.29 |      60.89 |      52.53 |    53.64 |
| dinov2_s_ridge_block4        |     40.61 |          41.40 |       45.86 |       41.72 |     53.58 |      67.52 |      49.65 |    60.40 |
| ssl_dinov2_s_lpft            |     40.70 |          41.01 |       43.19 |       40.97 |     41.06 |      40.21 |      40.80 |    41.77 |
| deit_s_ridge_block8          |     40.86 |          43.25 |       45.03 |       41.85 |     46.66 |      54.38 |      47.55 |    51.66 |
| deit_s_lp_clspatch           |     41.38 |          42.35 |       56.74 |       41.57 |     41.95 |      40.40 |      45.25 |    45.19 |
| dinov3_s_ridge_cls           |     41.43 |          41.59 |       44.44 |       42.41 |     42.84 |      45.17 |      41.51 |    45.46 |
| deit_s_ridge_patch           |     41.71 |          44.96 |       42.64 |       42.56 |     49.20 |      54.58 |      46.08 |    47.28 |
| dinov3_s_ridge_clspatch      |     41.76 |          42.01 |       42.58 |       42.42 |     42.56 |      44.68 |      42.80 |    47.37 |
| deit_s_ridge_clspatch        |     41.88 |          43.58 |       45.32 |       42.45 |     48.51 |      55.04 |      45.49 |    49.37 |
| dinov3_s_ridge_patch         |     42.05 |          42.57 |       41.38 |       42.52 |     42.80 |      45.70 |      43.91 |    50.22 |
| dinov2_s_ridge_patch         |     42.43 |          44.90 |       41.60 |       42.77 |     43.32 |      48.81 |      41.92 |    48.02 |
| dinov2_s_ridge_clspatch      |     42.47 |          44.77 |       41.42 |       43.10 |     43.68 |      48.11 |      41.31 |    47.59 |
| dinov2_s_ridge_cls           |     42.63 |          45.08 |       41.48 |       43.62 |     44.41 |      47.80 |      41.05 |    47.34 |
| dinov3_s_ridge_block4        |     42.88 |          43.88 |       50.44 |       44.21 |     56.35 |      61.13 |      53.29 |    65.33 |
| dinov2_s_ridge_block8        |     43.04 |          45.59 |       43.73 |       43.23 |     45.05 |      54.18 |      46.63 |    51.25 |
| dinov3_s_ridge_block8        |     43.48 |          46.38 |       44.26 |       43.57 |     46.82 |      60.84 |      47.83 |    51.21 |
| ssl_dinov2_s_ft              |     44.09 |          44.28 |       42.98 |       44.47 |     41.11 |      39.30 |      41.12 |    42.37 |
| dinov2_s_lp_clspatch         |     44.28 |          48.36 |       50.23 |       44.51 |     43.82 |      41.76 |      42.51 |    49.35 |
| dinov2_s_da_ridge_block8     |     44.31 |          45.15 |       45.65 |       44.82 |     46.48 |      58.54 |      46.02 |    55.85 |
| ssl_dinov3_s_lora            |     44.89 |          45.73 |       47.25 |       44.51 |     47.97 |      50.07 |      51.56 |    40.89 |
| dinov3_s_lp_clspatch         |     45.04 |          46.02 |       49.34 |       44.87 |     45.49 |      50.43 |      48.63 |    45.35 |
| deit_s_ridge_cls             |     45.20 |          45.54 |       50.59 |       45.07 |     49.40 |      59.33 |      47.29 |    54.76 |
| dinov2_s_da_lp_clspatch      |     45.79 |          46.96 |       47.04 |       45.35 |     48.08 |      47.69 |      48.76 |    40.70 |
| ssl_dinov2_s_lora            |     46.54 |          46.88 |       45.13 |       46.13 |     44.42 |      40.82 |      41.71 |    38.18 |
| dinov2_s_da_ridge_patch      |     46.94 |          48.33 |       46.86 |       47.85 |     49.98 |      58.35 |      46.69 |    54.68 |
| ssl_dinov2_s_da_lpft         |     47.47 |          47.76 |       46.47 |       47.43 |     44.34 |      43.81 |      41.39 |    45.56 |
| dinov2_s_da_ridge_clspatch   |     48.17 |          49.16 |       48.28 |       49.20 |     51.26 |      59.26 |      47.90 |    56.49 |
| ssl_dinov2_s_da_lora         |     48.94 |          48.91 |       40.93 |       48.81 |     45.05 |      47.85 |      43.83 |    41.56 |
| dinov2_s_da_ridge_cls        |     49.50 |          49.69 |       49.86 |       50.45 |     52.88 |      60.50 |      49.26 |    59.23 |
| vit_scratch                  |     35.28 |          35.57 |       48.83 |       34.91 |     36.92 |      45.06 |      45.32 |    39.04 |
| vit_deit_s_sgd               |     40.23 |          40.76 |       42.33 |       40.12 |     38.87 |      39.88 |      39.73 |    39.11 |
| vit_maxvit_t                 |     42.27 |          41.98 |       36.74 |       42.00 |     40.46 |      43.45 |      36.11 |    39.95 |
| vit_deit_s_img112            |     42.40 |          42.73 |       49.62 |       42.39 |     41.13 |      39.03 |      39.08 |    43.77 |
| vit_swin_t                   |     42.76 |          43.15 |       43.91 |       42.78 |     39.76 |      39.90 |      40.72 |    40.80 |
| vit_deit_s_strongaug         |     44.43 |          44.96 |       48.45 |       44.08 |     43.42 |      37.02 |      40.55 |    43.82 |
| vit_deit_s_wd_all            |     45.52 |          46.33 |       46.56 |       45.38 |     43.74 |      38.07 |      39.87 |    44.00 |
| vit_deit_s                   |     45.63 |          46.37 |       46.93 |       45.46 |     43.95 |      38.29 |      40.02 |    44.10 |
| vit_deit_s_nollrd            |     45.85 |          45.95 |       43.83 |       45.94 |     41.80 |      40.37 |      42.97 |    41.60 |
| vit_deit_s_nowarm            |     46.34 |          47.19 |       46.06 |       45.88 |     44.59 |      38.27 |      42.06 |    45.36 |
| vit_deit_s_dp0               |     47.68 |          48.13 |       44.50 |       47.29 |     45.76 |      39.07 |      41.88 |    44.29 |
