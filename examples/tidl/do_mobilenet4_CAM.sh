cp ./output4/mobileNet2_batch4/tidl_*.bin .
export FRAMECNT=3000
export VIDEO_PORT=1
export CLIP=/usr/share/ti/tidl/examples/classification/clips/test10.mp4
export BATCH_SIZE=4
export LABELS=labels_mobilenet_quant_v1_224.txt
export INPUT_NODE=input
./run_mobilenet_cv_mt -m ./output4/mobileNet2_batch4 -p $VIDEO_PORT -d cpu -i $INPUT_NODE -b $BATCH_SIZE -c $CLIP -f $FRAMECNT -l $LABELS
