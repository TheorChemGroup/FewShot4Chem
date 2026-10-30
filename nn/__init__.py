"""Contains neural networks and layers.

Every model class below is imported through a try/except ImportError guard rather than
unconditionally. Restricted pixi environments (e.g. "chemeleon", pinned to Python>=3.11 for
chemprop, which does not carry transformers/pytorch-fast-transformers/clamp/sentence-transformers)
would otherwise fail on `import nn` entirely for lacking ONE optional dependency, breaking every
model, not just the one that needed it. A class is None where its dependency is absent —
MODEL_REGISTRY simply won't have that entry in that environment — never faked or stubbed.
"""

from nn._register_model import MODEL_REGISTRY

try:
    from nn.ae import LightningAE
except ImportError:
    LightningAE = None

try:
    from nn.chemberta import Chemberta
except ImportError:
    Chemberta = None

try:
    from nn.chemfm import LightningChemFM
except ImportError:
    LightningChemFM = None

try:
    from nn.descriptors import LightningDescriptors
except ImportError:
    LightningDescriptors = None

try:
    from nn.fingerprint_creator import FingerprintCreator
except ImportError:
    FingerprintCreator = None

try:
    from nn.gin_pretrained import LightningGINPretrained
except ImportError:
    LightningGINPretrained = None

try:
    from nn.gnn import LightningMultilayerGATv2
except ImportError:
    LightningMultilayerGATv2 = None

try:
    from nn.grover_model import LightningGrover
except ImportError:
    LightningGrover = None

try:
    from nn.lit_clamp import LightningCLAMP
except ImportError:
    LightningCLAMP = None

try:
    from nn.mean_embeds import MeanEmbeds
except ImportError:
    MeanEmbeds = None

try:
    from nn.mole import LightningMolE
except ImportError:
    LightningMolE = None

try:
    from nn.molformer import LightningMoLFormer, LightningMoLFormerHF
except ImportError:
    LightningMoLFormer = None
    LightningMoLFormerHF = None

try:
    from nn.selformer import LightningSELFormer
except ImportError:
    LightningSELFormer = None

try:
    from nn.simpleconv import LightningSimpleConv
except ImportError:
    LightningSimpleConv = None

try:
    from nn.text_embedder import TextEmbedder
except ImportError:
    TextEmbedder = None

try:
    # chemprop>=2.2.0 requires Python>=3.11; not every pixi environment has it installed
    # (see models/configs/chemeleon/chemeleon.yaml). Registration is skipped, not faked,
    # when it is absent — MODEL_REGISTRY simply will not have "chemeleon" in that env.
    from nn.chemeleon import CheMeleon
except ImportError:
    CheMeleon = None

__all__ = [
    "MODEL_REGISTRY",
    "CheMeleon",
    "Chemberta",
    "FingerprintCreator",
    "LightningAE",
    "LightningCLAMP",
    "LightningChemFM",
    "LightningDescriptors",
    "LightningGINPretrained",
    "LightningGrover",
    "LightningMolE",
    "LightningMoLFormer",
    "LightningMoLFormerHF",
    "LightningMultilayerGATv2",
    "LightningSELFormer",
    "LightningSimpleConv",
    "MeanEmbeds",
    "TextEmbedder",
]
