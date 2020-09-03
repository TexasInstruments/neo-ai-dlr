artifacts_path=$(realpath $1)
echo $artifacts_path
export TIDL_SUBGRAPH_DIR=$artifacts_path
export TIDL_SUBGRAPH_DYNAMIC_OUTSCALE=1
export TIDL_SUBGRAPH_DYNAMIC_INSCALE=1
export FRAMECNT=3000
export VIDEO_PORT=1
export CLIP=/usr/share/ti/tidl/examples/classification/clips/test10.mp4
export BATCH_SIZE=4
export LABELS=labels_mobilenet_quant_v1_224.txt
export INPUT_NODE=input
./run_mobilenet_cv_mt -m $artifacts_path -p $VIDEO_PORT -d cpu -i $INPUT_NODE -b $BATCH_SIZE -c $CLIP -f $FRAMECNT -l $LABELS
