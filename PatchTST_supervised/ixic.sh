if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=40
model_name=PatchTST

root_path_name=./data/
data_path_name=^IXIC.csv
model_id_name=ixic
data_name=custom
pred_len=5
random_seed=2021
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MS \
    --target Close \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 5 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 64 \
    --d_ff 64 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 10\
    --stride 1\
    --des 'Exp' \
    --train_epochs 100\
    --patience 20\
    --embed timeF\
    --freq b\
    --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 