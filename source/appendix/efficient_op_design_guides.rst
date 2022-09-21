=================================================================
高效算子设计指南(ONNX篇)
=================================================================

当算子设计范围与硬件支持范围相契合时, 可以更充分地挖掘硬件性能, 提高模型推理速度.
本章节针对在 ``AX620`` 硬件平台上如何实现高效设计算子进行说明.

----------------------------------
Convolution
----------------------------------

.. note::

    The convolution operator consumes an input tensor and a filter, and computes the output.

**Conv** 支持的 ``OpSet Version``: ``1``, ``11-13``

.. csv-table::
   :file: ../csv/Conv_OP.csv
   :header-rows: 1

.. hint::

  ``input/output_channel``
      - ``input_channel`` 为 ``16`` 的倍数, ``output_channel`` 为 ``8`` 的倍数时效率最高
      - 不满足倍数限制时浪费补到对应倍数的计算量

----------------------------------
ConvTranspose
----------------------------------

``ConvTranspose`` 对以下三种情况支持最高效.

* kernel_size 为 ``2 x 2``, stride 为 ``2``, pad 取 ``0``
* kernel_size 为 ``4 x 4``, stride 为 ``2``, pad 取 ``1``
* kernel_size 为 ``4 x 4``, stride 为 ``4``, pad 取 ``0``

.. attention::

    ``ConvTranspose`` 的效率略低于实现同样上采样功能的 ``resize`` 算子.

----------------------------------
Linear
----------------------------------

推荐 ``channels`` 为 ``16`` 的倍数.

----------------------------------
Activation
----------------------------------

- ``ReLU`` 支持最高效
- ``LeakyReLU``, ``HardSwish``, ``Swish``, ``Mish`` 也能高效支持(但弱于 ``ReLU``)
- ``PReLU`` 支持效率较低

----------------------------------
Transpose/Reshape
----------------------------------

.. attention::

    实现效率较低, 尽量避免使用.

----------------------------------
Pool
----------------------------------

.. list-table::
    :widths: 10 60
    :header-rows: 1

    * - 算子
      - 高效建议

    * - MaxPool
      - 高效支持 ``kernel_size <= 2`` 且 ``kernel_size == stride`` 的的情况, 建议尽量 ``kernel_size <= 3``
    
    * - AvgPool
      - ``kernel_size`` 为 ``2`` 的幂最高效, 建议最大不超过 ``32``

----------------------------------
Resize
----------------------------------

- ``scale`` 仅支持二的幂, 建议 [1/16, 1/8, 1/4, 1/2, 2, 4, 8, 16] 范围内
- ``mode`` 仅支持 ``nearest``, ``bilinear`` 和 ``area``.
  