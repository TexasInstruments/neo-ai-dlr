**NOTE:** These examples can be tested only on **TI's AM5729/49 SoC** using Processor Linux SDK! 

More information can be found at: http://software-dl.ti.com/processor-sdk-linux/esd/docs/latest/linux/index.html, a Linux distro for AM5729/49 coming with all the necessary frameworks, libraries and drivers preinstalled.

- Before running this on AM5729/49 SoC, TVM compiled models need to be copied to the target filesystem.
  Models can be compiled using https://github.com/TexasInstruments/tvm/tree/dev/apps/tidl_deploy/NeoTvmCodeGen.py. Please check README.md in https://github.com/TexasInstruments/tvm/tree/dev/apps/tidl_deploy. 

- Batch size must be the same like batch size used for code generation (4 is min required to utilize 4 EVEs). Bigger batch size improves throughput, but also increases latency.
    
- For ``subgraph0.cfg`` (TIDL inference configuration file) check for more details at: http://downloads.ti.com/mctools/esd/docs/tidl-api/api.html#configuration-file. 
    Additional relevant example is ``/usr/share/ti/tidl/examples/mobilenet_subgraph`` (in AM5729/49 target filesystem).

- NEO-AI-DLR inference can be executed using helper scripts:

  - ``do_mobilenet4_CLIP.sh``: Decodes video clip preloaded on PLSDK filesystem, does image preprocessing, DLR inference and live display with the inference results overlaid, e.g. ``/do_mobilenet4_CLIP.sh ./output4/mobilenet1``

  - ``do_mobilenet4_CAM.sh``: Captures live video input, does image preprocessing, DLR inference and live display with inference results overlaid, e.g. ``./do_mobilenet4_CAM.sh ./output4/mobilenet2``

  - ``do_tidl4.sh``: Uses ``dog.npy`` image, replicates as many times as batch size is, and does file-to-stdout NEO-AI-DLR inference.
    Inference call is executed twice and execution time reported for both iterations (each iteration does multiple inferences defined by batch size). First inference call includes model initialization, and takes few additional seconds.

