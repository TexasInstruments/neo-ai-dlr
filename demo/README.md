# Demo
This folder contains demo projects for different frontend languages that can be built and deployed on target platforms.

## CPP and CV 
C++ demo source code that can be built into executables on Linux or Android platforms 

### Build 
Follow corresponding sections of [Installing DLR](https://neo-ai-dlr.readthedocs.io/en/latest/install.html) document to build the C library from source on Linux or Android. Then run 

`make demo`

to build executables.

In order to build CV example, which depends on OpenCV library, use:

`make democv`

to build the executable. Please check README.md in examples/tidl for more details.

### Running demo executables:
**Model_peeker**: a light-weight utility that prints out TVM model metadata.  
usage: 
`./model_peeker <model_dir> [device_type]`  
where device_type defaults to 'cpu'.

**Run_resnet**: a simple example that takes an image in the format of numpy array file (.npy), and outputs prediction result from typical image classification models like resnet or mobilenet.  
usage: 
`./run_resnet <model_dir> <ndarray file> [device_type] [input name]`  
where device_type defaults to "cpu", and input_name defaults to "data". 

**Run_mobilenet_cv_mt**: an example that is using live camera input or video clip, as input for DLR inference, and outputs prediction label overalid on live display output.
usage (check examples/tidl/do_mobilenet_CLIP4.sh, or do_mobilenet_CAM4.sh):

`./run_mobilenet_cv_mt -m <path to folder with TVM compiled model> -p <camera video port or -1 for video clip decoding> -d cpu -i <name of network input node> -b <batch size, default is 4> -c <video clip name to be decoded> -f <frame count to process> -l <file with class id to string mapping>`


## Python
Python demos coming soon.
