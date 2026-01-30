# 模型转换

## GPQT 量化 (可选)

**环境配置**

```sh
$ conda create -n gptq python=3.13 -y
$ conda activate gptq
```

当前使用的 `pytorch` 版本为 `2.9.1`, `torchvision` 版本为 `0.24.1`. 参考 [ModelCloud GPTQModel](https://github.com/ModelCloud/GPTQModel) 安装 `GPTQ` 量化支持.

```sh
# clone repo
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel

# python3-dev is required, ninja is to speed up compile, need to upgrade to latest `setuptools` to avoid errors
apt install python3-dev ninja setuptools -U

# pip: compile and install
# You can install optional modules like  vllm, sglang, bitblas.
# Example: pip install -v --no-build-isolation .[vllm,sglang,bitblas]
pip install -v . --no-build-isolation
```

**量化**

> 该脚本会将 LLM 部分单独量化, 不会影响视觉部分.

```sh
CUDA_VISIBLE_DEVICES=0 python3 convert_to_gptq.py \
    --model_id HY-MT1.5-1.8B \
    --out_dir ./HY-MT1.5-1.8B_GPTQ_INT4 \
    --bits 4
```

## LLM build

在 `AX650N` 上可以使用总上下文长度为 `2k` 的模型, 其中 `1k` 为 `prefill`, `1k` 是输出.

```bash
# 编译上下文 2k, 最大 prefill 为 1k 的模型
pulsar2 llm_build --input_path ../python/HY-MT1.5-1.8B_GPTQ_INT4  --output_path ../python/HY-MT1.5-1.8B_GPTQ_INT4_axmodel  --hidden_state_type bf16 --prefill_len 128 --kv_cache_len 2047 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512 --last_kv_cache_len 640 --last_kv_cache_len 768 --last_kv_cache_len 896 --last_kv_cache_len 1024  --chip AX650 -c 1 --parallel 8
```

使用上述命令编译大语言模型, 注意**自行修改**模型输入输出路径.

当编译目标平台为 `AX650N` 时, 设置 `FLOAT_MATMUL_USE_CONV_EU=1` 环境变量可以大幅度提高模型 `TTFT` 时间 (`AX620E` 无效).

---

当编译目标平台为 `AX620E` 时, 使用下面的命令编译上下文总长度为 `1k` 的模型, 其中 `512` 为 `prefill`, `512` 为输出.

```sh
pulsar2 llm_build --input_path HY-MT1.5-1.8B_GPTQ_INT4   --output_path HY-MT1.5-1.8B_GPTQ_INT4_axmodel_ax620e  --hidden_state_type bf16 -w s4 --prefill_len 128 --kv_cache_len 2047 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512 --last_kv_cache_len 640 --last_kv_cache_len 768 --last_kv_cache_len 896 --last_kv_cache_len 1024  --chip AX620E -c 1 --parallel 32
```

如果你在编译中遇到了下面的错误, 不用担心, 这并不是来自模型编译中的错误, 当日志中出现 `build llm model done!` 则意味着模型已经编译完毕, 后面的错误来自于模型对分需要特定的环境, 而这个环境通常不会包含在工具链的镜像中, 因此可以忽略这个错误.

```sh
2026-01-21 11:04:41.387 | SUCCESS  | yamain.command.llm_build:llm_build:364 - build llm model done!
`torch_dtype` is deprecated! Use `dtype` instead!
Traceback (most recent call last):
  File "/home/baiyongqiang/local_space/npu-codebase/yamain/common/error.py", line 59, in guard_context
    yield
  File "/home/baiyongqiang/local_space/npu-codebase/yamain/command/llm_build.py", line 447, in llm_build
    model = AutoModelForCausalLM.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/miniforge-pypy3/envs/npu/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 604, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/miniforge-pypy3/envs/npu/lib/python3.12/site-packages/transformers/modeling_utils.py", line 277, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/miniforge-pypy3/envs/npu/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4884, in from_pretrained
    hf_quantizer, config, dtype, device_map = get_hf_quantizer(
                                              ^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/miniforge-pypy3/envs/npu/lib/python3.12/site-packages/transformers/quantizers/auto.py", line 319, in get_hf_quantizer
    hf_quantizer.validate_environment(
  File "/home/baiyongqiang/miniforge-pypy3/envs/npu/lib/python3.12/site-packages/transformers/quantizers/quantizer_gptq.py", line 67, in validate_environment
    raise ImportError(
ImportError: Loading a GPTQ quantized model requires gptqmodel (`pip install gptqmodel`) or auto-gptq (`pip install auto-gptq`) library.
```
