#
echo 'python main.py --base_dir /data/lingyun/coswara-data/'
python main.py --base_dir /data/lingyun/coswara-data/

#
echo 'python main.py --base_dir /data/lingyun/coswara-data/ -bs 8'
python main.py --base_dir /data/lingyun/coswara-data/ -bs 8
echo 'python main.py --base_dir /data/lingyun/coswara-data/ -bs 16'
python main.py --base_dir /data/lingyun/coswara-data/ -bs 16
echo 'python main.py --base_dir /data/lingyun/coswara-data/ -bs 64'
python main.py --base_dir /data/lingyun/coswara-data/ -bs 64

#
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --hidden_dim 32'
python main.py --base_dir /data/lingyun/coswara-data/ --hidden_dim 32
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --hidden_dim 128'
python main.py --base_dir /data/lingyun/coswara-data/ --hidden_dim 128

#
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --num_layers 2'
python main.py --base_dir /data/lingyun/coswara-data/ --num_layers 2
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --num_layers 8'
python main.py --base_dir /data/lingyun/coswara-data/ --num_layers 8

#
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --num_layers 2'
python main.py --base_dir /data/lingyun/coswara-data/ --num_layers 2
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --num_layers 8'
python main.py --base_dir /data/lingyun/coswara-data/ --num_layers 8

#
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --dropout 0.1'
python main.py --base_dir /data/lingyun/coswara-data/ --dropout 0.1
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --dropout 0.5'
python main.py --base_dir /data/lingyun/coswara-data/ --dropout 0.5

#
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --optimizer sgd'
python main.py --base_dir /data/lingyun/coswara-data/ --optimizer sgd

#
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --lr 0.1'
python main.py --base_dir /data/lingyun/coswara-data/ --lr 0.1
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --lr 0.01'
python main.py --base_dir /data/lingyun/coswara-data/ --lr 0.01
echo 'python main.py --base_dir /data/lingyun/coswara-data/ --lr 0.0001'
python main.py --base_dir /data/lingyun/coswara-data/ --lr 0.0001
