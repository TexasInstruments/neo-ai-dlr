artifacts_path=$(realpath $1)
echo $artifacts_path
export TIDL_SUBGRAPH_DIR=$artifacts_path
export TIDL_SUBGRAPH_DYNAMIC_OUTSCALE=1
export TIDL_SUBGRAPH_DYNAMIC_INSCALE=1
python3 ./tidl_dlr4.py $artifacts_path 4 input
