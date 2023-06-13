__version__ = "0.1.0"
__model_version__ = "0.0.1"
import audiotools

audiotools.ml.BaseModel.INTERN += ["dac.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model
