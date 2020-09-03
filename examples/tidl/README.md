**NOTE:** These examples can be tested only on **TI's AM5729/49 SoC** using Processor Linux SDK! 

More information can be found at: http://software-dl.ti.com/processor-sdk-linux/esd/docs/latest/linux/index.html, a Linux distro for AM5729/49 coming with all the necessary frameworks, libraries and drivers preinstalled.

- Before running this on AM5729/49 SoC, TVM compiled models need to be copied to the target filesystem.
  Models can be compiled according to instructions at: http://software-dl.ti.com/processor-sdk-linux/esd/docs/latest/linux/Foundational_Components/Machine_Learning/neo.html#compiling-network-models-to-run-with-dlr.

- Batch size must be the same as batch size used for code generation (4 is min required to utilize 4 EVEs). Bigger batch size improves throughput, but also increases latency.

- NEO-AI-DLR inference can be executed using helper scripts:

  - ``do_mobilenet4_CLIP.sh``: Decodes video clip preloaded on PLSDK filesystem, does image preprocessing, DLR inference and live display with the inference results overlaid, e.g. ``/do_mobilenet4_CLIP.sh <artifacts_folder>``

  - ``do_mobilenet4_CAM.sh``: Captures live video input, does image preprocessing, DLR inference and live display with inference results overlaid, e.g. ``./do_mobilenet4_CAM.sh <artifacts_folder>``

  - ``do_tidl4.sh``: Uses ``dog.npy`` image, replicates as many times as batch size is, and does file-to-stdout NEO-AI-DLR inference.
    Inference call is executed twice and execution time reported for both iterations (each iteration does multiple inferences defined by batch size). First inference call includes model initialization, and takes few additional seconds.
