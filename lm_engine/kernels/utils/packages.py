# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

try:
    import cutlass.cute

    _IS_CUTE_DSL_AVAILABLE = True
except:
    _IS_CUTE_DSL_AVAILABLE = False


def is_cute_dsl_available() -> bool:
    return _IS_CUTE_DSL_AVAILABLE


try:
    import torch

    _IS_TORCH_AVAILABLE = True
except ImportError:
    _IS_TORCH_AVAILABLE = False


def is_torch_available() -> bool:
    return _IS_TORCH_AVAILABLE


try:
    import torch_neuronx

    _IS_TORCH_NEURONX_AVAILABLE = True
except:
    _IS_TORCH_NEURONX_AVAILABLE = False


def is_torch_neuronx_available() -> bool:
    return _IS_TORCH_NEURONX_AVAILABLE


_IS_JAX_AVAILABLE = None


def is_jax_available() -> bool:
    # deferred to the first call (instead of running at `import xma` time) so that merely importing xma
    # never pays JAX's import cost (nor touches the XLA/PJRT runtime) in torch-only processes.
    global _IS_JAX_AVAILABLE

    if _IS_JAX_AVAILABLE is None:
        try:
            import jax

            _IS_JAX_AVAILABLE = True
        except ImportError:
            _IS_JAX_AVAILABLE = False

    return _IS_JAX_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except:
    _IS_TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE


try:
    import causal_conv1d

    _IS_CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    _IS_CAUSAL_CONV1D_AVAILABLE = False


def is_causal_conv1d_available() -> bool:
    return _IS_CAUSAL_CONV1D_AVAILABLE
