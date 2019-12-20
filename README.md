# RL_Project
1. use tools tools/conv_*.py to prepare datasets; read the headers of those files; data is expected to be in ../data
2. pretrained HPC models are in trained_hpc, or you can use tools/hpc_svm.py to recreate them; they are needed in ../data
3. run python3.6 main.py --dataset [dataset] --flambda [lambda] --use_hpc [0|1] --pretrain [0|1], choose dataset from config_datasets/
4. you can also evaluate the agent on the test set with eval.py --dataset [dataset] --flambda [lambda]


Reference for Code : https://github.com/jaromiru/cwcf
