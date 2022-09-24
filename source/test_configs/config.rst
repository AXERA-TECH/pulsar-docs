.. _config_details:

============================
配置文件详细说明
============================

本节将对 ``SuperPulsar Config`` 部分进行介绍.

------------------------------------
config.prototxt 概述
------------------------------------

- ``SuperPulsar`` 是一个功能强大且复杂的工具集, 它在编译、仿真、调试模型中往往需要获得必要的配置参数, 才能精确地按照预想的方式工作
- 当前, 所有的参数都可以通过配置文件传递给 ``SuperPulsar`` , :ref:`少数参数 <some_params_called_by_cmdline>` 可以通过命令行参数接口临时指定或修改
- 配置文件除了给编译器传递配置参数, 它还有一个重要的作用是可以指导编译器完成复杂的编译过程

.. mermaid::

  graph LR
    config_prototxt[配置文件]
    command_line[命令行参数]
    input_model[输入模型1<br/>输入模型2<br/>...<br/>输入模型N] ----> super_pulsar[SuperPulsar]
    super_pulsar ----> output_model[输出模型]
    config_prototxt -.-> |传递编译参数| super_pulsar
    command_line -.-> |设置且覆盖编译参数| super_pulsar

.. attention::

    通过命令行参数接口传递的编译参数, 将覆盖配置文件提供的参数

**内容格式**

- 当前配置文件格式是一种叫 ``prototxt`` 的文本格式, 其内容可以直接借助文本编辑器进行阅读和修改
- ``prototxt`` 格式的配置文件内部可以使用注释, 注释以 ``#`` 开头
- 版本为 ``proto3``

**名词约定**

- 由于配置文件结构复杂, 且配置参数层级较深, 为了方便描述, 尽可能减少因用词不当造成理解上的偏差, 在此约定一些常用名词的含义. 如果您在阅读本系列使用文档中发现有任何表达不清、用词错误等等问题或者有更好的建议, 欢迎批评指正

**参数路径**

- **参数路径** 用于表达某一个配置参数在多层结构体参数中的位置
- 当一个配置参数位于其它具有多个嵌套层级的结构体中时, 将使用各级结构体参数的名称加点号 ``.`` 组成的字符串来表达当前正在介绍的参数在配置文件中的位置, 例如用字符串 ``pulsar_conf.batch_size`` 代表的参数在配置文件中的位置如下:

  .. code-block:: sh
    :name: input_conf_items
    :linenos:
    
    pulsar_conf { # 编译器相关配置
      batch_size: 1
    }

.. note::

  | 这里 **把配置文件本身当做一个匿名的结构体** , 它内部的一级参数的路径就是参数名称本身
  | 当介绍 **通用数据类型** 时不标注参数路径, 因为它们可能出现在配置文件的多个地方
  | 有些地方会使用参数完整路径或相对路径来表达一个参数名称

**编译过程**

- 编译过程一般用于代指将一种格式的模型编译成另一种格式. 如把一个 ``onnx`` 格式的模型编译成 ``joint`` 格式

**编译步骤**

- 编译步骤一般用于一个编译过程可以明确地分出几个步骤的情况中. 例如将两个 ``onnx`` 模型先分别编译成 ``joint`` 格式, 然后再将两个 ``joint`` 模型融合得到一个 ``joint`` 模型
- 在描述配置文件的时候, 可能会说 “整个编译过程分为三个编译步骤, 每一个 **编译步骤** 的配置参数”
- 但是当我们要具体描述一个 **编译过程** 中的某一个 **编译步骤** 时, 又可能在一个小节内会将 **编译步骤** 说成 **编译过程** , 这个时候这两个词多代指的对象是一样的. 注意结合上下文区分

**编译参数**

- 编译参数用于代指 **编译过程** 或 **编译步骤** 所需要配置的参数

-----------------------------------
config 内部结构概览
-----------------------------------

``config.prototxt`` 由以下六个部分组成, 包括:

- :ref:`输入输出配置 <input_and_output_config>`
- :ref:`选择硬件平台 <select_hardware>`
- :ref:`CPU子图设置 <cpu_backend_settings>`
- :ref:`Tensor的特殊处理 <tensor_conf>`
- :ref:`Neuwizard的配置 <neuwizard_conf>`
- :ref:`Pulsar的配置 <pulsar_conf>`

config 内部结构示例

.. code-block:: sh
  :name: config.prototxt outline
  :linenos:
  :emphasize-lines: 13-14, 17, 19
  
  # config.outline.prototxt

  # 基本的输入输出配置
  input_path:  # 输入模型的相对路径
  input_type:  # 输入模型类型, 缺省等同于 INPUT_TYPE_AUTO, 编译器将按照模型文件名自动推断, 但有时候推断结果并不是预期的
  output_path: # 输出模型的相对路径
  output_type: # 输出模型类型, 不显示指定时, 按模型文件后缀名自动识别, 缺省等同于 OUTPUT_TYPE_AUTO

  # 硬件选择
  target_hardware:       # 目前可选 AX620, AX630
  
  # Tensor的特殊处理(旧版本), 称为 tensor_conf, 推荐使用新版本, 可以定制更复杂的功能
  input_tensors      {}
  output_tensors     {}

  # Tensor的特殊处理(新版本)
  src_input_tensors  {}  # 用于描述输入模型的输入 tensor 的属性, 与 input_tensors 等效
  src_output_tensors {}  # 用于描述输入模型的输出 tensor 的属性
  dst_input_tensors  {}  # 用于修改输出模型的输入 tensor 的属性, 与 output_tensors 等效
  dst_output_tensors {}  # 用于修改输出模型的输出 tensor 的属性

  # cpu 子图后端处理引擎: ONNX OR AXE
  cpu_backend_settings {}

  # neuwizard 参数配置
  neuwizard_conf {               # 用于指导 Neuwizard 将 onnx 模型编译成 lava_joint 格式
    operator_conf            {}  # 用于配制各种盖帽算子
    dataset_conf_calibration {}  # 用于描述编译过程中的校准数据集
  }

  # pulsar compiler 配置
  pulsar_conf {
    # 用于指导 pulsar_compiler 将 lava_joint 或 lava 格式的模型编译成 joint 或者 neu 格式的模型
    ...
  }

在 ``config.prototxt`` 需要按照以上结构合理配置.

.. attention::

  保留 ``input_tensors``, ``output_tensors`` 选项是为了兼容旧版本工具链, 而 ``src_input_tensors`` 和 ``dst_input_tensors`` 等价于 ``input_tensors`` 和 ``output_tensors``, 推荐使用新版本的 :ref:`tensor_conf <tensor_conf>`.

--------------------------------------
配置文件不同模块的详细说明
--------------------------------------

本节分别对 ``config.prototxt`` 中的各个 ``sub_config`` 做详细说明.

.. _input_and_output_config:

~~~~~~~~~~~~~~~~~~~~~~
输入输出配置
~~~~~~~~~~~~~~~~~~~~~~

.. _input_path:

^^^^^^^^^^^^^^^^^^^^^^^
input_path
^^^^^^^^^^^^^^^^^^^^^^^

属性说明

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数路径
      - ``input_path``
    - - 参数作用
      - 指定输入模型的路径
    - - 参数类型
      - String
    - - 可选列表
      - /
    - - 注意事项  
      - 1.路径是配置文件所在目录的相对路径

        2.参数值字符串需要用半角双引号 "" 包裹

代码示例

.. code-block:: sh
  :linenos:

  # input_path example
  input_path: "./model.onnx"

^^^^^^^^^^^^^^^^^^^^^^^
input_type
^^^^^^^^^^^^^^^^^^^^^^^

属性说明

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数路径
      - ``input_type``
    - - 参数作用
      - | 明示输入模型的类型
        | 缺省时编译器将按模型文件名称的后缀名自动推断. 有的时候推断结果可能不是期望的
    - - 参数类型
      - Enum
    - - 可选列表
      - ``INPUT_TYPE_ONNX``
    - - 注意事项  
      - 注意枚举参数值不需要带引号

代码示例

.. code-block:: sh
  :linenos:

  # input_type example

  input_type: INPUT_TYPE_ONNX

.. _output_path:

^^^^^^^^^^^^^^^^^^^^^^^^
output_path
^^^^^^^^^^^^^^^^^^^^^^^^

属性说明

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数路径
      - ``output_path``
    - - 参数作用
      - 指定输出模型的路径
    - - 参数类型
      - String
    - - 可选列表
      - /
    - - 注意事项  
      - 同 :ref:`input_path <input_path>`

代码示例

.. code-block:: sh
  :linenos:

  # output_path example

  output_path: "./compiled.joint"

^^^^^^^^^^^^^^^^^^^^^^^^
output_type
^^^^^^^^^^^^^^^^^^^^^^^^

属性说明

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数路径
      - ``output_type``
    - - 参数作用
      - 指定输出模型的类型
    - - 参数类型
      - Enum
    - - 可选列表
      - ``OUTPUT_TYPE_JOINT``
    - - 注意事项  
      - 注意枚举参数值不需要带引号

代码示例

.. code-block:: sh
  :linenos:

  # output_type example

  output_type: OUTPUT_TYPE_JOINT

.. _select_hardware:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
target_hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

属性说明

.. list-table::
    :widths: 15 40
    :header-rows: 1

    * - 属性
      - 描述
    * - 参数路径
      - ``target_hardware``
    * - 参数作用
      - 指定编译输出模型所适用的硬件平台
    * - 参数类型
      - Enum
    * - 可选列表
      - | ``TARGET_HARDWARE_AX630``
        | ``TARGET_HARDWARE_AX620``
    * - 注意事项
      - 无


代码示例

.. code-block:: sh
  :linenos:

  # target_hardware example

  target_hardware: TARGET_HARDWARE_AX630

.. tip::

  推荐在命令行参数中指定硬件平台, 避免因为硬件平台的原因导致模型转换报错.

.. _tensor_conf:

~~~~~~~~~~~~~~~~~~~~~~~~~~
tensor_conf
~~~~~~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^
概述
^^^^^^^^^^^^^^^^^^^^^

.. note::

  ``SuperPulsar`` 工具链具备调整输出模型的输入/输出 ``tensor`` 的属性的能力, 
  即允许输出模型(如 ``joint`` 模型)跟原始输入模型(如 ``onnx`` 模型)的输入输出数据属性(比如图像尺寸、颜色空间等)不一致.

**tensor_conf** 配置包括 ``src_input_tensors`` , ``src_output_tensors`` , ``dst_input_tensors`` , ``dst_output_tensors``. 

属性说明

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数路径
      - ``config_name`` 自身, 例如 ``src_input_tensors``
    - - 参数作用
      - | ``src_input_tensors`` 用于 ``描述(说明)`` 输入模型的 ``输入 tensor`` 属性
        | ``src_output_tensors`` 用于 ``描述(说明)`` 输入模型的 ``输出 tensor`` 属性
        | ``dst_input_tensors`` 用于 ``修改`` 输出模型的 ``输入 tensor`` 属性
        | ``dst_output_tensors`` 用于 ``修改`` 输出模型的 ``输出 tensor`` 属性
    - - 参数类型
      - Struct
    - - 可选列表
      - /
    - - 注意事项
      - 无

^^^^^^^^^^^^^^^^^^^^^
可选列表
^^^^^^^^^^^^^^^^^^^^^

""""""""""""""""""""""
tensor_name
""""""""""""""""""""""

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数名
      - ``tensor_name``
    - - 参数作用
      - 指定当前结构体所描述输入模型的 ``tensor`` 或所作用的输出模型的 ``tensor`` 的名称
    - - 参数类型
      - String
    - - 可选列表
      - /
    - - 注意事项
      - 对于 ``src_input_tensors``、 ``src_output_tensors``、 ``dst_input_tensors`` 和 ``dst_output_tensors`` 等每一个数组, 
        若其中有任何一个 ``item`` 结构体中的 ``tensor_name`` 字段是缺省的, 那么该 ``item`` 的内容将覆盖所在数组中的其它 ``item`` 的内容

.. _color_space:

""""""""""""""""""""""
color_space
""""""""""""""""""""""

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数名
      - ``color_space``
    - - 参数作用
      - 用于描述输入模型的 ``tensor`` 的颜色空间, 或指定输出模型的 ``tensor`` 的颜色空间
    - - 参数类型
      - Enum
    - - 可选列表
      - | ``TENSOR_COLOR_SPACE_BGR``
        | ``TENSOR_COLOR_SPACE_RGB``
        | ``TENSOR_COLOR_SPACE_GRAY``
        | ``TENSOR_COLOR_SPACE_NV12``
        | ``TENSOR_COLOR_SPACE_NV21``
        | ``TENSOR_COLOR_SPACE_BGR0``
        | ``TENSOR_COLOR_SPACE_AUTO``
        | **DEFAULT:** ``TENSOR_COLOR_SPACE_AUTO`` , 根据模型输入 channel 数自动识别: 3-channel: BGR; 1-channel: GRAY
    - - 注意事项
      - 无

.. _data_type:

""""""""""""""""""""""
data_type
""""""""""""""""""""""

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数名
      - ``data_type``
    - - 参数作用
      - 指定输入输出 ``tensor`` 的数据类型
    - - 参数类型
      - Enum
    - - 可选列表
      - | ``DATA_TYPE_UNKNOWN``
        | ``UINT2``
        | ``INT2``
        | ``MINT2``
        | ``UINT4``
        | ``MINT4``
        | ``UINT8``
        | ``INT8``
        | ``MINT8``
        | ``UINT16``
        | ``INT16``
        | ``FLOAT32``
        | **DEFAULT:** ``UINT8`` 为输入 ``tensor`` 的默认值 , ``FLOAT32`` 为输出 ``tensor`` 的默认值
    - - 注意事项
      - 无

.. _QValue:

""""""""""""""""""""""""""""""""""""""""""""
quantization_value
""""""""""""""""""""""""""""""""""""""""""""

一个整数, 通常被称为 ``Q`` 值. 配置正数时生效, 或者满足以下条件之一时也会以推荐值生效

  - 源模型输出实型, 目标模型输出整型
  - 源模型输入实型, 目标模型输入整型

代码示例

.. code-block:: sh

  # 配置 Q 值
  dst_output_tensors {
    data_type: INT16
    quantization_value: 256  # 不配置时为动态Q值
  }

.. hint::

  ``Q`` 值可以理解为一种特殊的 ``affine`` 操作. ``Q`` 值实际上代表了一个 ``scale`` , 可以通过把实数域的输出除以 ``sclae`` 后
  转换成规定的定点数值域.

.. note::

  ``Q`` 值分两种:
    * 动态 ``Q`` 值通过 ``calibration`` 数据集中的最大最小范围动态计算出 ``scale`` 值.
    * 静态 ``Q`` 值通常是用户根据先验信息手动指定了 ``scale`` 值.

.. hint::

  ``joint`` 模型中包含了 ``Q`` 值信息, 在 ``run_joint`` 时会打印出具体的 ``Q`` 值.

.. attention::
  
  ``AX630`` 上使用 ``Q`` 值, 可以省一步 ``cpu affine`` 操作, 因此可以实现加速. 而 ``AX620`` 支持 ``float`` 输出, 所以即使是用了 ``Q`` 值也不能提速.

""""""""""""""""""""""
color_standard
""""""""""""""""""""""

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - 属性
      - 描述
    - - 参数名
      - ``color_standard``
    - - 参数作用
      - 用于设置色彩空间标准
    - - 参数类型
      - Enum
    - - 可选列表
      - | ``CSC_LEGACY``
        | ``CSS_ITU_BT601_STUDIO_SWING``
        | ``CSS_ITU_BT601_FULL_SWING``
        | **DEFAULT:** ``CSC_LEGACY``
    - - 注意事项
      - 无

""""""""""""""""""""""
tensor_layout
""""""""""""""""""""""

.. list-table::
  :widths: 10 60
  :header-rows: 1

  - - 属性
    - 描述
  - - 参数名
    - ``tensor_layout``
  - - 参数作用
    - 用于修改数据排布形式
  - - 参数类型
    - Enum
  - - 可选列表
    - | ``NHWC``
      | ``NCHW``
      | ``NATIVE`` 默认项, 不推荐
  - - 注意事项
    - 无

代码示例

.. code-block:: sh
  :linenos:

  # target_hardware example

  src_input_tensors {
    color_space: TENSOR_COLOR_SPACE_AUTO
  }
  dst_output_tensors {
    color_space: TENSOR_COLOR_SPACE_NV12
  }

.. _cpu_backend_settings:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU子图的设置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  ``AXEngine`` 是 ``AXera`` 自研的推理库, 可以在某种程度上提升模型的 ``FPS`` , 本质上是将 ``ONNX`` 的 ``CPU`` 子图替换为了 ``AXE`` 子图, 在内存使用方面, ``AXE`` 子图在某些模型上的内存使用量也会大幅度降低, 最差情况下也是和原来 ``ONNX`` 持平.

.. list-table::
    :widths: 15 40
    :header-rows: 1

    * - 属性
      - 描述
    * - 参数路径
      - ``cpu_backend_settings``
    * - 参数作用
      - 控制编译后模型采用的 ``CPU`` 后端模式, 目前有 ``ONNX`` 与 ``AXEngine`` 可选
    * - 参数类型
      - Struct
    * - 可选列表
      - /
    * - 注意事项  
      - 如果需要使带 ``AXEngine`` 后端的 ``joint`` 模型可以在某一个旧版不支持 ``AXEngine`` 后端的 ``BSP`` 上运行时, 需要同时开启 ``onnx_setting.mode`` 与 ``axe_setting.mode`` 为 ``ENABLE``
    
代码示例

.. code-block:: sh
  :linenos:

  cpu_backend_settings {
    onnx_setting {
      mode: ENABLED
    }
    axe_setting {
      mode: ENABLED
      axe_param {
        optimize_slim_model: true
      }
    }
  }

字段说明

.. list-table::
    :header-rows: 1

    * - 字段名
      - 参数路径
      - 参数类型
      - 参数作用
      - model
      - 注意事项
    * - ``onnx_setting``
      - cpu_backend_settings.onnx_setting
      - Struct
      - 控制 ``ONNX`` 后端是否开启
      - DEFAULT / ENABLED / DISABLED, 默认为 DEFAULT
      - ONNX 的 DEFAULT 与 ENABLED 等价
    * - ``axe_setting``
      - cpu_backend_settings.axe_setting
      - Struct
      - 控制 ``AXEngine`` 后端是否开启
      - DEFAULT / ENABLED / DISABLED, 默认为 DEFAULT
      - AXEngine 的 DEFAULT 与 DISABLED 等价
    * - ``optimize_slim_model``
      - cpu_backend_settings.axe_setting.axe_param.optimize_slim_model
      - Bool
      - 表示是否开启优化模式
      - 无
      - 当网络输出特征图较小时建议开启, 否则不建议

.. important::

  推荐用户更多地使用 ``AXE`` 的 ``CPU`` 后端（模型 ``initial`` 更快，速度优化也更好），目前的 ``ONNX`` 后端支持是为了兼容旧版本工具链, 在未来的版本中将会逐步废弃.

.. _neuwizard_conf:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
neuwizard_conf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``neuwizard_conf`` 中包含多种配置信息, 可以通过合理配置其中选项以满足多种需求.

^^^^^^^^^^^^^^^^^^^^^^^^^^
operator_conf
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  ``operator_conf`` 中可以为输入输出配置盖帽运算, 附加的盖帽算子对现有算子的输入或输出的 ``tensor`` 附加一次运算; 在配置文件中, 添加盖帽算子的过程是通过给现有算子的输入或输出 ``tensor`` 扩充或修改属性的过程来实现的.

输入输出盖帽算子可以实现 ``tensor`` 的前处理和后处理

.. list-table::
  :widths: 10 20 50
  :header-rows: 1

  - - 算子列表
    - 类型
    - 描述
  - - ``input_conf_items``
    - Struct
    - 前处理算子, 用于为模型的输入数据做前处理
  - - ``output_conf_items``
    - Struct
    - 后处理算子, 用于对输出数据做后处理

代码示例

.. code-block::
  :name: gm_opr
  :linenos:

  # 示例代码, 不能直接拷贝使用
  neuwizard_conf {
    operator_conf {
      input_conf_items {
        selector {
          ...
        }
        attributes {
          # 前处理算子数组
          ...
        }
      }
      output_conf_items {
        selector {
          ...
        }
        attributes {
          # 后处理算子数组
          ...
        }
      }
    }
  }

""""""""""""""""""""""""""""""""
前处理与前处理算子
""""""""""""""""""""""""""""""""

参数路径

- ``neuwizard_conf.operator_conf.input_conf_items``

示例代码

.. code-block:: sh
  :name: input_conf_items.pre
  :linenos:

  # 注意按参数路径, 将以下内容放入配置文件中合适的位置
  input_conf_items {
      # selector 用于指示附加的前处理算子将要作用的输入 tensor
      selector {
          op_name: "inp" # 输入 tensor 的名称
      }
      # attributes 用于包裹作用于 "inp" 的盖帽算子
      attributes {
          input_modifications {
              # 对输入数据做一个 affine 操作, 用于改变编译后模型的输入数据类型, 既将输入数据类型由浮点数域 [0, 1) 类型改为 uint8
              affine_preprocess {
                  slope: 1
                  slope_divisor: 255
                  bias: 0
              }
          }
      }
  }

.. attention::

  ``affine`` 本质上是一个 ``* k + b`` 操作. 
  ``affine_preprocess`` 中的 ``affine`` 操作与直觉相反, 例如将浮点数域 [0, 1) 类型改为 UINT8 [0, 255] 是需要除以 ``255`` 而不是乘 ``255``, 
  而将 [0, 255] 转换为浮点域 [0, 1], 需要乘以 ``255`` (配置 slope_divisor 为 ``0.00392156862745098``).

.. _input_conf_items_selector:

``input_conf_items.selector`` 属性说明

.. list-table::
  :widths: 10 60
  :header-rows: 1

  - - 属性
    - 描述
  - - 参数名
    - ``selector``
  - - 参数路径
    - :file:`neuwizard_conf.operator_conf.input_conf_items.selector`
  - - 参数作用
    - 用于指示附加的前处理算子将要作用的输入 tensor 的名称
  - - 字段说明
    - | ``op_name`` 指定输入 tensor 的完整名称. 如 "inp"
      | ``op_name_regex`` 指定一个正则表达式, 用于适配多个 tensor. 相应的 attributes 结构体中的盖帽算子将作用于所有被适配的 tensor

代码示例

.. code-block:: sh
  :name: input_conf_items.selector
  :linenos:

  # input_conf_items.selector 示例
  selector {
    op_name: "inp"
  }

.. _input_conf_items_attribute:

``input_conf_items.attributes`` 属性说明

.. list-table::
  :widths: 10 60
  :header-rows: 1

  * - 属性
    - 描述
  * - 参数名
    - ``attributes``
  * - 参数路径
    - :file:`neuwizard_conf.operator_conf.input_conf_items.attributes`
  * - 参数类型
    - Struct
  * - 参数作用
    - 用于描述对输入 ``tensor`` 的属性的更改, 目标输入 ``tensor`` 由 ``input_conf_items.selector`` 所指定
  * - 字段说明
    - | ``type`` : 明示或修改输入 ``tensor`` 的数据类型. 枚举类型, 默认值 ``DATA_TYPE_UNKNOWN``
      | ``input_modifications`` : 前处理算子数组, 对输入 tensor 添加的盖帽算子. 有多种, 可以同时指定多个

其中, ``type`` 为枚举类型, :ref:`点击这里 <data_type>` 查看支持的类型. ``input_modifications`` 具体说明如下:

.. list-table::
  :widths: 10 60
  :header-rows: 1

  * - 属性
    - 描述
  * - 字段名
    - ``input_modifications``
  * - 类型
    - Struct
  * - 作用
    - 作用于某一个输入 ``tensor`` 的 **前处理算子** 所组成的数组
  * - 注意事项
    - 在前处理算子数组中的所有算子依次执行, 排在数组中第二位的算子以前一个算子的输出为输入, 依次类推
    
**前处理算子**

前处理算子包括 ``input_normalization`` 和 ``affine_preprocess``.

.. list-table::
  :widths: 10 60
  :header-rows: 1
  :name: 前处理算子[input_normalization]

  * - 算子名称
    - ``input_normalization``
  * - 参数路径
    - neuwizard_conf.operator_conf.input_conf_items.attributes.input_modifications.input_normalization
  * - 字段说明
    - ``mean`` : 浮点数数组
      ``std`` : 浮点数数组
  * - 作用
    - 实现 :math:`y = (x - mean) / std` .
  * - 注意事项:
    - | 这里 ``mean/std`` 的顺序与输入 ``tensor`` 的 :ref:`颜色空间 <color_space>` 有关
      | 如果上述变量等于 ``TENSOR_COLOR_SPACE_AUTO`` / ``TENSOR_COLOR_SPACE_BGR`` 则 ``mean/std`` 的顺序为 ``BGR``
      | 如果上述变量等于 ``TENSOR_COLOR_SPACE_RGB`` 则 ``mean/std`` 的顺序就是 ``RGB``

.. _pre_affine_preprocess:

.. list-table::
  :widths: 10 60
  :header-rows: 1
  :name: 前处理算子[affine_preprocess]

  * - 算子名称
    - ``affine_preprocess``
  * - 参数路径
    - neuwizard_conf.operator_conf.input_conf_items.attributes.input_modifications.affine_preprocess
  * - 字段说明
    - | ``slope`` : 浮点数数组
      | ``slope_divisor`` : 浮点数数组
      | ``bias`` : 浮点数数组. 数组长度同 ``slope``
      | ``bias_divisor`` : 浮点数数组. 数组长度同 ``slope``
  * - 作用
    - 实现 :math:`y = x * (slope / slope\_divisor) + (bias / bias\_divisor)` .
  * - 注意事项:
    - 无

代码示例

.. code-block:: sh
  :name: input_conf_items.attributes.input_modifications.affine_preprocess
  :linenos:

  # 将输入数据类型由数域 {k / 255}(k=0, 1, ..., 255) 改为整数域 [0, 255], 希望编译后的模型输入数据类型为 uint8
  affine_preprocess {
    slope: 1
    slope_divisor: 255
    bias: 0
  }

""""""""""""""""""""""""""""""""
后处理与后处理算子
""""""""""""""""""""""""""""""""

参数路径

- ``neuwizard_conf.operator_conf.output_conf_items``

代码示例

.. code-block:: sh
  :name: output_conf_items.post
  :linenos:

  # 注意按参数路径, 将以下内容放入配置文件中合适的位置
  output_conf_items {
      # selector 用于指示输出 tensor
      selector {
          op_name: "oup" # 输出 tensor 的名称
      }
      # attributes 用于包裹作用于 "oup" 的盖帽算子
      attributes {
          output_modifications {
              # 对输出数据做一个 affine 操作, 用于改变编译后模型的输出数据类型, 既将输出数据类型由浮点数 [0, 1) 类型改为 uint8
              affine_preprocess {
                  slope: 1
                  slope_divisor: 255
                  bias: 0
              }
          }
      }
  }

``output_conf_items.selector`` 同 :ref:`input_conf_items.selector <input_conf_items_selector>` , ``output_conf_items.attributes`` 同 :ref:`input_conf_items.attribute <input_conf_items_attribute>` .

**后处理算子**

后处理算子 ``affine_preprocess``.

.. list-table::
  :widths: 10 60
  :header-rows: 1
  :name: 后处理算子[affine_preprocess]

  * - 算子名称
    - 算子说明
  * - affine_preprocess
    - 对输出 ``tensor`` 做 ``affine`` 操作

其余同 :ref:`input_modifications.affine_preprocess <pre_affine_preprocess>`


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dataset_conf_calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _calibration:

.. list-table::
  :widths: 10 60
  :header-rows: 1
  :name: dataset_conf_calibration

  * - 算子名称
    - ``dataset_conf_calibration``
  * - 参数路径
    - neuwizard_conf.dataset_conf_calibration
  * - 作用
    - 用于描述校准过程中需要的数据集
  * - 注意事项:
    - 默认的 ``batch_size`` 为 ``32``, 如果出现 ``Out Of Memory, OOM`` 的错误, 可以尝试调小 ``batch_size``

代码示例

.. code-block:: sh
  :name: output_conf_items
  :linenos:

  dataset_conf_calibration {
    path: "../imagenet-1k-images.tar"  # 需要换成自己使用的量化数据
    type: DATASET_TYPE_TAR             # 类型是 tar
    size: 256                          # 一个整数, 用于表示数据集大小, 会从全集里随机采样
    batch_size: 32                     # 一个整数, 用于转模型过程中, 内部参数训练、校准或误差检测时所使用数据的 batch_size, 默认值为 32
  }

.. _pulsar_conf:

~~~~~~~~~~~~~~~~~~~~~~~~~~
pulsar_conf
~~~~~~~~~~~~~~~~~~~~~~~~~~

属性说明

.. list-table::
    :widths: 15 40
    :header-rows: 1

    * - 属性
      - 描述
    * - 参数路径
      - ``pulsar_conf``
    * - 参数作用
      - 编译器子工具 ``pulsar_compiler`` 的配置参数

        用于指导 ``pulsar_compiler`` 将 ``lava_joint`` 或 ``lava`` 格式的模型编译成 ``joint`` 或 ``neu`` 格式的模型
    * - 参数类型
      - Struct
    * - 可选列表
      - /
    * - 注意事项  
      - 注意按照参数路径放入到配置文件的正确位置

代码示例

.. code-block:: sh
  :name: config.pulsar_conf
  :linenos:

  pulsar_conf {
    ax620_virtual_npu: AX620_VIRTUAL_NPU_MODE_111 # 编译后模型使用 ax620 虚拟 NPU 1+1 模式的 1 号虚拟核
    batch_size_option: BSO_DYNAMIC                # 编译后的模型支持动态 batch
    batch_size: 1
    batch_size: 2
    batch_size: 4                                 # 最大 batch_size 为 4; 要求 batch_size 为 1 2 或 4 时推理保持较高性能
  }

结构体字段说明

.. list-table::
    :header-rows: 1

    * - 字段名
      - 参数路径
      - 参数类型
      - 参数作用
      - 可选列表
      - 注意事项
    * - ``virtual_npu``
      - pulsar_conf.virtual_npu
      - Enum
      - 指定目标模型所使用的 ``AX630A`` 虚拟 ``NPU`` 核
      - | ``VIRTUAL_NPU_MODE_AUTO``
        | ``VIRTUAL_NPU_MODE_0``
        | ``VIRTUAL_NPU_MODE_311``
        | ``VIRTUAL_NPU_MODE_312``
        | ``VIRTUAL_NPU_MODE_221``
        | ``VIRTUAL_NPU_MODE_222``
        | **DEFAULT:** ``VIRTUAL_NPU_MODE_AUTO``
      - | MODE_0表示不使用虚拟NPU
        | 此配置项需要在 ``SuperPulsarConfiguration.target_hardware`` 被指定为 ``TARGET_HARDWARE_AX630`` 的前提下使用
        | 此配置项跟 ``ax620_virtual_npu`` 二选一使用
    * - ``ax620_virtual_npu``
      - pulsar_conf.ax620_virtual_npu
      - Enum
      - 指定目标模型所使用的 ``AX620A`` 虚拟 ``NPU`` 核
      - | ``AX620_VIRTUAL_NPU_MODE_AUTO``
        | ``AX620_VIRTUAL_NPU_MODE_0``
        | ``AX620_VIRTUAL_NPU_MODE_111``
        | ``AX620_VIRTUAL_NPU_MODE_112``
      - | MODE_0表示不使用虚拟NPU
        | 此配置项需要在 ``SuperPulsarConfiguration.target_hardware`` 被指定为 ``TARGET_HARDWARE_AX620`` 的前提下使用
        | 此配置项跟 virtual_npu 二选一使用
    * - ``batch_size_option``
      - pulsar_conf.batch_size_option
      - Enum
      - 设置 ``joint`` 格式模型所支持的 ``batch`` 类型
      - | ``BSO_AUTO``
        | ``BSO_STATIC``  # 静态 ``batch``, 推理时固定 ``batch_size``, 性能最优
        | ``BSO_DYNAMIC`` # 动态 ``batch``, 推理时支持不超过最大值的任意 ``batch_size``, 使用较灵活
        | **DEFAULT:** ``BSO_AUTO`` , 默认为静态 ``batch``
      - 无
    * - ``batch_size``
      - pulsar_conf.batch_size
      - IntArray
      - 设置 ``joint`` 格式模型所支持的 ``batch size`` , 默认为 1
      - /
      - | 当指定了 ``batch_size_option`` 为 ``BSO_STATIC`` 时, ``batch_size`` 表示 ``joint`` 格式模型推理时能用的唯一 ``batch size``
        | 当指定了 ``batch_size_option`` 为 ``BSO_DYNAMIC`` 时, ``batch_size`` 表示 ``joint`` 格式模型推理时所能使用的最大 ``batch size``
        | 当生成支持动态 ``batch`` 的 ``joint`` 格式模型时, 可配置多个值, 以提高使用不超过这些值的 ``batch size`` 进行推理时的性能
        | 当指定多个 ``batch size`` 时会增加 ``joint`` 格式模型文件的大小
        | 当配置多个 ``batch_size`` 时, ``batch_size_option`` 将默认采用 ``BSO_DYNAMIC``

.. _some_params_called_by_cmdline:

----------------------------------------------
可以通过命令行传递的参数
----------------------------------------------

.. hint::

  命令行参数会 override 配置文件中的某些对应配置, 命令行参数只起到辅助作用, 通过配置文件可以实现更复杂的功能.

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - 参数
      - 说明
    - - input
      - 输入模型路径
    - - output
      - 输出模型路径
    - - calibration_batch_size
      - 校准数据集的 batch_size
    - - batch_size_option
      - {BSO_AUTO,BSO_STATIC,BSO_DYNAMIC}
    - - output_dir
      - 指定输出目录
    - - virtual_npu
      - 指定虚拟NPU
    - - input_tensor_color
      - {auto,rgb,bgr,gray,nv12,nv21}
    - - output_tensor_layout
      - {native,nchw,nhwc}
    - - color_std
      - {studio,full} only support nv12/nv21 now
    - - target_hardware 
      - {AX630,AX620,AX170} target hardware to compile
    - - enable_progress_bar
      - 是否打印进度条, 默认不开启


----------------------------------------------
config.prototxt 最简配置
----------------------------------------------

simplest_config.prototxt 示例, 可以直接复制到文件中运行.

.. code-block::
  :name: simplest_config.prototxt
  :linenos:

  # simplest_config.prototxt 示例, 可以直接复制到文件中运行
  input_type: INPUT_TYPE_ONNX     # 指明输入模型的类型为 onnx, 如果此字段被省略, 编译器将按模型文件后缀名自动推断, 然而有时推断结果可能不是期望的
  output_type: OUTPUT_TYPE_JOINT  # 指定输出模型的类型为Joint
  src_input_tensors {                     # 用于描述输入模型的输入 tensor 的属性
    color_space: TENSOR_COLOR_SPACE_AUTO  # 由编译器自行判断颜色空间
  }
  dst_input_tensors {                     # 用于修改输出模型的输入 tensor 的属性
    color_space: TENSOR_COLOR_SPACE_AUTO  # 由编译器自行判断颜色空间
  }
  neuwizard_conf {       # neuwizard 参数配置
    operator_conf {      # 输入输出盖帽配置: 附加的输入输出盖帽算子对现有算子的输入或输出的 tensor 附加一次运算；在配置文件中, 添加盖帽算子的过程是通过给现有算子的输入或输出 tensor 扩充或修改属性的过程来实现的
      input_conf_items { # 用于为模型的输入数据做前处理
        attributes {     # 用于描述对输入 tensor 的属性的更改, 目标输入 tensor 由 input_conf_items.selector 所指定, 不指定, 默认为 ? 
          input_modifications {   # 前处理算子数组, 对输入 tensor 添加的盖帽算子, 有多种, 可以同时指定多个, 在前处理算子数组中的所有算子依次执行, 排在数组中第二位的算子以前一个算子的输出为输入, 依次类推
            affine_preprocess {   # 对输入数据做一个 affine (i.e. x * k + b)操作, 用于改变编译后模型的输入数据类型, 可将输入数据类型由浮点数 [0, 1) 类型改为 uint8
              slope: 1            # 浮点数数组. 数组长度等于 1 或者数据的 channel 数. 当长度为 1 时, 编译工具会自动复制 channel 次
              slope_divisor: 255  # 浮点数数组. 数组长度同 slope
              bias: 0             # 浮点数数组. 数组长度同 slope
                                  # 实际效果等同于: y = x * (slope / slope_divisor) + (bias / bias_divisor)
            }
          }
        }
      }
    }
    dataset_conf_calibration {
      path: "./imagenet-1k-images.tar"  # 一个具有 1000 张图片的 tar 包, 用于编译过程中对模型校准
      type: DATASET_TYPE_TAR            # 类型为 tar
      size: 256                         # 表示数据集大小, 会从全集里随机采样, batch_size 默认为 32
    }
  }
  pulsar_conf {    # pulsar compiler 参数配置
    batch_size: 1  # 设置 joint 格式模型所支持的 batch size, 默认为 1
  }
