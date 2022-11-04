# TI's fork of DLR

DLR is a compact, common runtime for deep learning models and decision tree models compiled by [TVM](https://github.com/neo-ai/tvm), [AWS SageMaker Neo](https://aws.amazon.com/sagemaker/neo/) or [Treelite](https://treelite.readthedocs.io/en/latest/install.html). DLR uses the TVM runtime for models compiled by TVM. DLR provides unified Python/C++ APIs for loading and running compiled models on various devices.

TI's DLR fork adds the capability of loading and running models compiled by [TI's TVM fork](https://github.com/TexasInstruments/tvm/tree/tidl-j7).

## Branches

  * tidl-j7 - This is the release branch
    (in sync with TI's TVM tidl-j7 branch)


## Tags/Releases

TI's DLR releases are synchronized with TI's [Processor SDK RTOS](https://www.ti.com/tool/download/PROCESSOR-SDK-RTOS-J721E) releases. The following table lists the PSDK RTOS release and the corresponding DLR tag that is compatible with it.

| PSDK release | TVM+TIDL release tag | Key features                                              |
|--------------|----------------------|-----------------------------------------------------------|
| 8.4          | TIDL\_PSDK\_8.4      | Update tvm                                                |
| 8.2          | TIDL\_PSDK\_8.2      | Merged with neo-ai-dlr v1.10.0                            |
| 8.1          | TIDL\_PSDK\_8.0      |                                                           |
| 8.0          | TIDL\_PSDK\_8.0      |                                                           |
| 7.3          | TIDL\_PSDK\_7.3      | Merged with neo-ai-dlr v1.8.0                             |


License
-------
TVM is licensed under the [Apache-2.0](LICENSE) license.

Getting Started
---------------
Refer to the [TI TVM User's Guide](https://software-dl.ti.com/codegen/docs/tvm/building.html).

Release Notes
-------------
Refer to [Release Notes](https://software-dl.ti.com/codegen/docs/tvm/release-notes.html) in the TI TVM User's Guide.
