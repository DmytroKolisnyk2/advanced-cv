# Results

CV = mean ± std over the 6 group folds of the per-sample log-EMD (lower is better). LB = Kaggle public leaderboard (private is hidden until the competition ends).


## Baselines and shortcut controls

| name             | CV (mean ± std)   |   D50_err_dec |   lb_public |   gpu_hours |   peak_mem_gb | setup   |
|:-----------------|:------------------|--------------:|------------:|------------:|--------------:|:--------|
| ctrl_colour_only | 49.689 ± 15.760   |         0.424 |     nan     |       0.000 |         0.000 |         |
| base_train_mean  | 86.632 ± 3.037    |         0.923 |     102.372 |       0.000 |         0.000 |         |
| ctrl_camera_only | 87.055 ± 2.713    |         0.929 |     nan     |       0.000 |         0.000 |         |
| base_uniform     | 102.020 ± 5.477   |         0.914 |     nan     |       0.000 |         0.000 |         |


## Part 1 — CNNs

| name                         | CV (mean ± std)   |   D50_err_dec |   lb_public |   gpu_hours |   peak_mem_gb | setup                                                                                                    |
|:-----------------------------|:------------------|--------------:|------------:|------------:|--------------:|:---------------------------------------------------------------------------------------------------------|
| cnn_r18_bs8_grouped_frozenbn | 35.878 ± 17.187   |         0.313 |      86.289 |       0.053 |         3.210 | opt=adamw, lr=0.0003, norm=frozen_bn, aug=basic, sampler=grouped, batch=8, mode=ft, tile_mm=48           |
| cnn_r18_groupnorm            | 37.527 ± 16.186   |         0.330 |     nan     |       0.018 |         5.030 | opt=adamw, lr=0.0003, norm=groupnorm, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48        |
| cnn_r18_frozenbn             | 39.497 ± 18.297   |         0.337 |     nan     |       0.015 |         3.210 | opt=adamw, lr=0.0003, norm=frozen_bn, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48        |
| cnn_r18_sgd                  | 40.340 ± 21.109   |         0.363 |     nan     |       0.016 |         3.210 | opt=sgd, lr=0.01, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48              |
| cnn_r50_adamw                | 41.354 ± 18.897   |         0.364 |     nan     |       0.035 |         3.750 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_r18_noaug                | 41.422 ± 16.618   |         0.373 |     nan     |       0.015 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=none, sampler=by_sample, batch=32, mode=ft, tile_mm=48           |
| cnn_r18_tile24mm             | 41.894 ± 19.339   |         0.382 |     nan     |       0.015 |         2.360 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=24          |
| cnn_r18_const_nowarm         | 42.205 ± 17.394   |         0.372 |     nan     |       0.016 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_r18_img112               | 42.464 ± 19.404   |         0.371 |     nan     |       0.015 |         2.460 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, img=112, tile_mm=48 |
| cnn_effb0_adamw              | 42.798 ± 18.207   |         0.384 |     nan     |       0.096 |         3.710 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_r18_tile96mm             | 43.072 ± 17.428   |         0.387 |     nan     |       0.015 |         1.200 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=96          |
| cnn_r18_adamw                | 43.280 ± 20.463   |         0.390 |     nan     |       0.018 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_convnext_t_adamw         | 44.576 ± 20.304   |         0.398 |     nan     |       0.071 |         4.520 | opt=adamw, lr=0.0001, norm=default, aug=basic, sampler=by_sample, batch=32, mode=ft, tile_mm=48          |
| cnn_r18_strongaug            | 46.087 ± 22.261   |         0.432 |     nan     |       0.015 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=strong, sampler=by_sample, batch=32, mode=ft, tile_mm=48         |
| cnn_r18_bs8_grouped          | 50.851 ± 22.334   |         0.481 |     nan     |       0.056 |         3.210 | opt=adamw, lr=0.0003, norm=default, aug=basic, sampler=grouped, batch=8, mode=ft, tile_mm=48             |


## Controls and scale robustness (CV EMD)

| name                         |   cv_mean |   ctrl_shuffle |   ctrl_grey |   scale_ref |   res_0.5 |   res_0.25 |   ppm_x0.5 |   ppm_x2 |
|:-----------------------------|----------:|---------------:|------------:|------------:|----------:|-----------:|-----------:|---------:|
| cnn_r18_bs8_grouped_frozenbn |     35.88 |          35.66 |       35.13 |       36.39 |     36.90 |      39.74 |      46.35 |    42.39 |
| cnn_r18_groupnorm            |     37.53 |          37.42 |       42.41 |       37.84 |     38.37 |      41.18 |      41.83 |    39.65 |
| cnn_r18_frozenbn             |     39.50 |          39.23 |       45.61 |       39.50 |     36.59 |      38.97 |      41.64 |    41.19 |
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
