NOTE: These examples can be tested only on AM5729/49 SoC. More information at: http://software-dl.ti.com/processor-sdk-linux/esd/docs/latest/linux/index.html
Linux distro for AM5729/49 is preinstalled with all the necessary frameworks, libraries and drivers.

- For subgraph0.cfg modifications take a look at /usr/share/ti/tidl/examples/mobilenet_subgraph
  Also check more details (TIDL inference configuration file) at: http://downloads.ti.com/mctools/esd/docs/tidl-api/api.html#configuration-file
- Before running this on target, TVM compiled models need to be copied to AM5729/49 SoC. 
  Models can be compiled using https://github.com/TexasInstruments/tvm/tree/dev/apps/tidl_deploy/NeoTvmCodeGen.py. Please check README.md in this repo.
  Batch size must be the like batch size used for code generation (4 is min required to utilize 4 EVEs)
  Bigger batch size improves throughput, but increases latency as well.

- NEO-AI-DLR inference can be executed using helper scripts:
  - do_mobilenet4_CLIP.sh
    Decodes video clip on PLSDK filesystem, does image preprocessing, DLR inference and live display
    E.g.: ./do_mobilenet4_CLIP.sh ./output4/mobilenet1
  - do_mobilenet4_CAM.sh
    Captures live video input, does image preprocessing, DLR inference and live display
    E.g.: ./do_mobilenet4_CAM.sh ./output4/mobilenet2
  - do_tidl4.sh
    Uses dog.npy image, replicates as many times as batch size is, and does file-to-stdout NEO-AI-DLR inference.
    Inference is executed twice and execution time reported for both iterations. First inference includes model initialization, and takes few seconds. 
 
