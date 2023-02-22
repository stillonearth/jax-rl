# MiniJAXRL

A verifiable implementation of some RL algorithms in JAX.

Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) by [@vwxyzjn](https://github.com/vwxyzjn).

Jax is better than pytorch in terms of performance, but it is not as easy to use as pytorch. This repo is an attempt to make it easier to use.

**This is work in progress**

## Soft-Actor-Critic

A jax port of clean-rl SAC.

![sac v3](https://spinningup.openai.com/en/latest/_images/math/c01f4994ae4aacf299a6b3ceceedfe0a14d4b874.svg)

- Implementation: [sac.py](./sac.py)
- Usage: `python sac.py`

### JAX Environmental Variables

JAX might try to preallocate too much memory on GPU, so you need to set this env variable:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Dreamer

A one-file mini implementation of [Dreamer (v1)](https://arxiv.org/pdf/1912.01603.pdf) algorithm in JAX.

![dreamer v1](https://user-images.githubusercontent.com/97428129/220439512-d2ce48cb-09cc-439f-b502-10ac1fba370a.png)

### JAX Environmental Variables

Depending on JAX version and system configuration these variables might be needed.

```bash
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Implementations

- Implementation: [dreamer_v1.py](./sac.py)
- Usage: `python dreamer_v1.py`
