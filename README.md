# jax-rl

A verifiable implementation of some RL algorithms in JAX.

Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) by [@vwxyzjn](https://github.com/vwxyzjn).

## Soft-Actor-Critic

A jax port of clean-rl SAC.

![sac v3](https://spinningup.openai.com/en/latest/_images/math/c01f4994ae4aacf299a6b3ceceedfe0a14d4b874.svg)

- Implementation: [sac.py](./sac.py)
- Usage: `python sac.py`

## Notes

JAX would try to preallocate too much memory on GPU, so you need to set this env variable:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```
