# mosFoundation


python train_graphclas.py --pretrain 0  --bn --l2_normalize --alpha 0.003 --mask Edge --train_dataset ROSMAP --load 1 --num_train_epoch 100 --fold_n 1

python train_graphclas-analysis.py --pretrain 0  --bn --l2_normalize --alpha 0.003 --mask Edge --train_dataset ROSMAP