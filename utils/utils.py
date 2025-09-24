import jax
import jax.numpy as jnp
import flax.linen as nn

class L1Norm(nn.Module):
    eps: float = 1e-8
    axis: int = -1
    keepdims: bool = True

    @nn.compact
    def __call__(self, x):
        abs_mean = jnp.mean(jnp.abs(x), axis=self.axis, keepdims=self.keepdims)
        norm_factor = jnp.maximum(abs_mean, self.eps)
        return x / norm_factor

class L2Norm(nn.Module):
    eps: float = 1e-8
    axis: int = -1

    @nn.compact
    def __call__(self, x):
        norm = jnp.linalg.norm(x, axis=self.axis, keepdims=True)
        return x / (norm + self.eps)