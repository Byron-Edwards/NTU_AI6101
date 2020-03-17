#!/bin/sh
#python a2c_main.py --cuda --name 1 --batchsize 32 --envsize 32  & 
#python a2c_main.py --cuda --name 2 --batchsize 64 --envsize 32  &
#python a2c_main.py --cuda --name 3 --batchsize 128 --envsize 32  &
#python a2c_main.py --cuda --name 4 --batchsize 128 --envsize 64  &
#python a2c_main.py --cuda --name 5 --batchsize 32 --envsize 8  &
#python a2c_main.py --cuda --name 6 --batchsize 32 --envsize 50 --lr 0.001 &
#python a2c_main.py --cuda --name 9 --batchsize 32 --envsize 32  &
#python a2c_main.py --cuda --name 10 --batchsize 50  --envsize 50  &
#python a2c_main.py --cuda --name 11 --batchsize 32 --envsize 32 --game boxing --stopreward 100 &
#python a2c_main.py --cuda --name 12 --batchsize 50  --envsize 50 --game boxing --stopreward 100 &
#python a2c_main.py --cuda --name 13 --batchsize 32 --envsize 50 --game boxing --stopreward 100 --lr 0.00001 &
#python a2c_main.py --cuda --name 14 --batchsize 32  --envsize 50 --game boxing --stopreward 100 --lr 0.0001 &
#python a2c_main.py --cuda --name 15 --batchsize 32  --envsize 50 --game boxing --stopreward 100 --lr 0.0001 --entropy 0.03  &


## FOR ZIANG
python a2c_main.py --cuda --name 1  --batchsize 32  --envsize 50 --game boxing --stopreward 100 --lr 0.0001 &
python a2c_main.py --cuda --name 2  --batchsize 32  --envsize 50 --game boxing --stopreward 100 --lr 0.00001 &
python a2c_main.py --cuda --name 3  --batchsize 32  --envsize 16 --game boxing --stopreward 100 --lr 0.00001 &
python a2c_main.py --cuda --name 4  --batchsize 64  --envsize 32 --game boxing --stopreward 100 --lr 0.00001 &
python a2c_main.py --cuda --name 5  --batchsize 128  --envsize 32 --game boxing --stopreward 100 --lr 0.00001 &
python a2c_main.py --cuda --name 6  --batchsize 128  --envsize 64 --game boxing --stopreward 100 --lr 0.00001 &



## FOR ZIANG
