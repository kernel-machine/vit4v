# python -u timesformer_train2.py --dataset /home/nonroot/dataset/manual_video_16 --epochs 100 --devices 0,1 --batch_size 4
for i in $(seq $1 $2); 
do 
echo "Running vivit_$i"
../env/bin/python -u test_video.py --model ../runs2/vivit_${i}_2/model.pth --video /dataset/raw/dataset_v2_raw_s32_val --window_size 32
done