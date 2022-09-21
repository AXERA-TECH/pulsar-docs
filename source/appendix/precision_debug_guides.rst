=========================================
精度损失排查与精度调优建议
=========================================

****************************
精度损失排查
****************************

.. attention::

    转换后的模型出现精度损失时, 请按照以下推荐方式排查出现问题的 ``stage`` 或 ``layer``

.. _checklists:

------------------------
CheckLists
------------------------

.. data:: 第一步, 明确发生精度损失的硬件平台

    1. 仅发生在 ``AX`` 平台上

        ::

            请继续往下排查.
    
    2. 其他平台均有掉点问题发生

        ::

            共性问题, 需要用户自行斟酌是否要训练一个更好的模型然后再重新量化;
            确定其他平台的使用 INT8 还是 INT16 量化, 亦或者是混合量化.

.. data:: 第二步, 确定发生精度损失的阶段

    1. ``pulsar run`` 对分精度低 (``cos-sim < 98%``)

        ::

            请按照 [第三步] 建议继续往下排查

    2. 上板接入用户 ``后处理`` 程序, 解析之后精度很低

        ::

            请按照 [第四步] 建议继续往下排查

.. _three_step:

.. data:: 第三步, cos-sim 低于 98%, 排查建议

    1. ``pulsar run`` 所需的 ``output_config.prototxt`` 文件 **必须** 由 ``pulsar build`` 自动生成
    2. 检查 ``pulsar build`` 配置文件 ``config.prototxt`` 中的 ``color space`` 和 ``mean/std`` 配置是否正确
    3. 使用 ``pulsar run`` 比较 ``model.lava_joint`` 和 ``model.onnx`` 之间的 ``cos-sim`` 值, 观察是否发生精度损失
    4. 使用逐层对分查看发生精度损失的 ``layer``

.. _four_step:

.. data:: 第四步, 上板精度低, 排查建议

    1. 执行 ``run_joint`` 指令时会打印 ``joint`` 模型的部分信息, 用户需要检查 ``后处理程序`` 是否正确解析输出数据
    2. 如果其他平台不掉点, 但是在 ``AX`` 平台上报错的 ``BadCase``, 具体见 :ref:`上板精度丢失排查方法 <precision_loss_on_board>`

.. data:: 第五步, 寻求 Axera 帮助
    
    当用户通过前四步排查建议依然无法解决问题时, 请将相关 ``log`` 以及 ``结论`` 发送给 ``FAE`` 同学, 以便 ``AX`` 工程师定位问题

-----------------------
模型编译后掉点
-----------------------

本节对 :ref:`CheckLists <checklists>` 中 :ref:`第三步 <three_step>` 进行详细说明.

.. hint::

    ``pulsar run`` 为 ``SuperPulsar`` 工具链中用于仿真和对分的集成工具, 详情可以参考 :ref:`x86平台下仿真与对分 <pulsar_run_sim>`

如果原始 ``onnx`` 模型编译为 ``joint`` 模型后, ``pulsar run`` 的 ``cos-sim`` 很低, 说明转换后的模型发生了精度损失, 需要排查问题出现的具体位置.

.. data:: config 配置

    ``pulsar run`` 所需要的 ``config`` 是从 ``pulsar build`` 中自动生成的.

    .. code-block:: python
        :linenos:

        # 注意, 以下命令并不完整
        pulsar build --input model.onnx --config config.prototxt --output_config output_config.prototxt  ...
        pulsar run model.onnx model.joint --config output_config.prototxt  ...

.. data:: csc & mean/std

    ``color space convert, csc`` 配置之后需要按照通道顺序配置 ``mean/std``.

    .. code-block:: python
        :linenos:

        # 配置编译后模型的输入数据 color space 为 BGR
        dst_input_tensors {
            color_space: TENSOR_COLOR_SPACE_BGR
        }

        # mean/std 需要按照 BGR 的顺序填写
        input_normalization {
            mean: [0.485, 0.456, 0.406]  # 均值
            std: [0.229, 0.224, 0.255]   # 方差
        }

    ``dst_input_tensors`` 中 ``color_space`` 为 ``BGR`` 代表编译时是按照 ``BGR`` 格式来读取的校正图片数据, 从而 ``mean/std`` 也要按 ``BGR`` 顺序设置.

.. data:: 查看模型量化阶段是否已经丢失精度

    ``pulsar build`` 编译过程中, 会生成一个调试用的中间文件 ``model.lava_joint``, 通过

    .. code-block:: python
        :linenos:
        
        # 注意, 以下命令均不完整
        pulsar run model.onnx model.lava_joint --input ...

    可以验证在量化阶段是否发生了精度损失.

.. data:: 模型量化阶段丢失精度解决方法

    1. 增加量化数据集

        .. code-block:: python
            :linenos:

            dataset_conf_calibration {
                path: "imagenet-1k-images.tar"
                type: DATASET_TYPE_TAR
                size: 256       # 编译过程中校准所需的实际数据个数
                batch_size: 32  # 默认为 32, 可以修改为其他值
            }

    2. 调整量化策略和量化方法

        - 量化策略, ``CALIB_STRATEGY_PER_CHANNEL`` 和 ``CALIB_STRATEGY_PER_TENSOR``
        - 量化方法, ``OBSVR_METHOD_MIN_MAX`` 和 ``OBSVR_METHOD_MSE_CLIPPING``
        - 量化策略和量化方法可以 **两两组合**, 其中 ``CALIB_STRATEGY_PER_CHANNEL`` 可能会出现掉点
        - 推荐 ``PER_TENSOR/MIN_MAX`` 或 ``PER_TENSOR/MSE_CLIPPING`` 的组合方式

        .. code-block:: python
            :linenos:

            dataset_conf_calibration {
                path: "magenet-1k-images.tar"  # 量化数据集
                type: DATASET_TYPE_TAR
                size: 256       # 编译过程中校准所需的实际数据个数
                batch_size: 32  # 默认为 32, 可以修改为其他值

                calibration_strategy: CALIB_STRATEGY_PER_TENSOR  # 量化策略
                observer_method: OBSVR_METHOD_MSE_CLIPPING       # 量化方法
            }

    3. 采用 ``INT16`` 量化

        - 具体参考 :ref:`16bit 量化 <Q16bit>`
    
    4. 打开 ``dataset_conf_error_measurement``, 用于编译过程中误差测试

        .. code-block:: python
            :linenos:

            dataset_conf_error_measurement {
                path: "imagenet-1k-images.tar"
                type: DATASET_TYPE_TAR
                size: 32
                batch_size: 8
            }

.. data:: 逐层对分

    具体功能参考 :ref:`逐层对分使用说明 <layer_wise_compare>`

.. data:: pulsar debug
    
    ``pulsar debug`` 功能后续将补充介绍

.. _precision_loss_on_board:

-----------------------
模型上板掉点
-----------------------

本节对 :ref:`CheckLists <checklists>` 中 :ref:`第四步 <four_step>` 进行详细说明.

.. data:: 判断后处理程序是否有误

    在 ``AX`` 开发板上使用 ``run_joint`` 指令可以实现板端推理, 然后使用用户自己的后处理程序对推理结果进行解析.

    为了验证的用户后处理程序是否存在错误, 可以比较在同一输入的条件下, ``pulsar run`` 的输出结果与 ``run_joint`` 的输出结果之间是否存在差异, 
    
    具体参考 :ref:`gt 文件夹对分说明 <pulsar_run_gt_compare>`, 如果对分成功, 说明 **用户的** 后处理程序 ``可能`` 存在错误.

.. data:: 后处理程序正确, 但精度依然很低

    原因的可能
        * ``npu simulator`` 生成的指令和 ``cmode`` 跑出的结果不一致
        * ``run_joint.so`` 和 ``npu drive`` 出错
    
    这类问题需要提交相关日志, 以便于快速修复.

.. data:: BadCase 处理

    其他平台没问题, 仅在 ``AX`` 平台精度很低, 对于这类 ``BadCase``, 先用 ``pulsar run`` 查一下 ``cos-sim``, 如果没有掉点严重(低于 98%), 
    
    那么将这个 ``BadCase`` 送到板子上用 ``run_joint`` 跑一遍,

    看下结果是否和 ``pulsar run`` 一致, 如果不一致, 说明是上板有问题, 需要 ``AX`` 工程师修复.

------------------------
其他注意事项
------------------------

如果需要 ``AX`` 工程师排查问题, 请提供详细的日志信息以及相关实验结论. 

>>> 注: 如果能够提供最小复现集可以提高解决问题的效率.

.. note::

    在某些情况下 ``SILU`` 函数会导致检测模型的 ``mAP`` 很低, 将其替换为 ``ReLU`` 函数后可以解决问题.

.. note::

    如果 ``量化数据集`` 与 ``训练数据集`` 差别很大, 会导致精度大幅度降低.

    判断 ``calibration`` 选择是否合理, 可以从 ``calibration`` 数据集中选取一张进行 ``pulsar run`` 对分.

****************************
精度调优建议
****************************

对于量化后的精度误差, 建议用户采用以下 ``2`` 种方式进行优化, 且均需要在 ``config.prototxt`` 文件配置后重新转换模型.

--------------------
calibration 设置
--------------------

* 量化策略与量化方案两两组合
* 尝试使用其他量化数据集
* 适当增加或减少数据量

--------------------
QAT 训练
--------------------

当使用各种调优手段依然无法提高模型精度时, 那么该模型可能是 ``PTQ`` 方案中的 ``corner case``, 这个时候可以尝试使用 ``QAT`` 训练.

.. attention::
    
    更多调优建议将会逐步更新.