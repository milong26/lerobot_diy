import os
IS_LOCAL = os.environ.get("ENV", "") == "local"
if IS_LOCAL:
    from . import server
from . import tools
from . import obj_dection


