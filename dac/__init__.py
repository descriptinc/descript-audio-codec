__version__ = "0.0.4"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["dac.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model
