for n in {1..10}
do
  python -u run.py --dataset 'Indian' --epoches 200 --channel 200 --patches 9 --s_patches 9 --aug_num 3 \
                  --lambdaC 0.01 --sample_num 5 --batch_size 16 --learning_rate 0.001 --decay_num 5\
                  --d_model 256 --out_dims 32 --dropout 0.1 --itr $n
  sleep 2
done

python -u main.py --dataset 'Indian'

for n in {1..10}
do
  python -u run.py --dataset 'Pavia' --epoches 200 --channel 103 --patches 9 --s_patches 9 --aug_num 2 \
                  --lambdaC 0.01 --sample_num 5 --batch_size 36 --learning_rate 0.0001 --decay_num 10 \
                  --d_model 256 --out_dims 32 --dropout 0.1 --itr $n
  sleep 2
done

python -u main.py --dataset 'Pavia'

for n in {1..10}
do
  python -u run.py --dataset 'Salinas' --epoches 200 --channel 204 --patches 9 --s_patches 9 --aug_num 3 \
                  --lambdaC 0.01 --sample_num 5 --batch_size 32 --learning_rate 0.001 --decay_num 5 \
                  --d_model 256 --out_dims 32 --dropout 0.1 --itr $n
  sleep 2
done

python -u main.py --dataset 'Salinas'

for n in {1..10}
do
  python -u run.py --dataset 'Houston' --epoches 200 --channel 144 --patches 9 --s_patches 9 --aug_num 3 \
                  --lambdaC 0.01 --sample_num 5 --batch_size 30 --learning_rate 0.001 --decay_num 5 \
                  --d_model 256 --out_dims 32 --dropout 0.1 --itr $n
  sleep 2
done

python -u main.py --dataset 'Houston'
