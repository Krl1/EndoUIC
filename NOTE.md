How to infer from outside a Docker container (from `low-light-image-enhancement/externals/EndoUIC`):

`sudo docker exec -it endouic_rtx3080 bash -c 'cd /app/EndoUIC && python endouic/train.py -opt options/infer_cec.yaml'`

How to calculate metrics (from `low-light-image-enhancement`):

`python scripts/calculate_metrics.py   --enh_dir externals/EndoUIC/experiments/infer_cec/visualization   --ref_dir datasets/cec_dataset_only_dim/test/clean   --output endouic_net_g_latest_results.csv   --method EndoUIC_net_g_latest   --comparison_file experiments_comparison.csv`