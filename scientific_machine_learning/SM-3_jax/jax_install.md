

```
pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 버전 지정하고 싶을 때
pip install --upgrade "jax[cuda]"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

```python
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```

```
> pip install flax
```

- To upgrade to the latest version of JAX and Flax, you can use:

```
> pip install --upgrade pip jax jaxlib
> pip install --upgrade git+https://github.com/google/flax.git
```