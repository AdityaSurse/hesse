"""hesse: Live Loss-Landscape Curvature Tracking & Edge-of-Stability Diagnostics in PyTorch."""

from hesse.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    TrackerConfig,
)
from hesse.data import (
    DatasetBundle,
    get_cifar10_subset,
    get_mnist_subset,
    get_synthetic_regression,
    load_dataset,
)
from hesse.hessian import (
    Curvature,
    HVPOperator,
    LanczosResult,
    PowerIterationResult,
    PreconditionedResult,
    TraceResult,
    hutchinson_trace,
    hvp,
    lanczos,
    power_iteration,
    preconditioned_sharpness,
    spectral_density,
    stochastic_lanczos_quadrature,
)
from hesse.models import (
    MLP,
    MODEL_REGISTRY,
    SmallCNN,
    TinyTransformer,
    get_model,
)
from hesse.optim import (
    Lion,
    get_optimizer,
    get_stability_threshold,
)
from hesse.tracker import Tracker
from hesse.train import train
from hesse.utils import (
    count_params,
    flatten_grads,
    flatten_params,
    get_device,
    set_seed,
    unflatten_params,
)
from hesse.viz import plot_run

__version__ = "1.0.0"

__all__ = [
    "Curvature",
    "DataConfig",
    "DatasetBundle",
    "HVPOperator",
    "LanczosResult",
    "Lion",
    "MODEL_REGISTRY",
    "MLP",
    "ModelConfig",
    "OptimConfig",
    "PowerIterationResult",
    "PreconditionedResult",
    "RunConfig",
    "SmallCNN",
    "TinyTransformer",
    "TraceResult",
    "Tracker",
    "TrackerConfig",
    "count_params",
    "flatten_grads",
    "flatten_params",
    "get_cifar10_subset",
    "get_device",
    "get_mnist_subset",
    "get_model",
    "get_optimizer",
    "get_stability_threshold",
    "get_synthetic_regression",
    "hutchinson_trace",
    "hvp",
    "lanczos",
    "load_dataset",
    "plot_run",
    "power_iteration",
    "preconditioned_sharpness",
    "set_seed",
    "spectral_density",
    "stochastic_lanczos_quadrature",
    "train",
    "unflatten_params",
]
