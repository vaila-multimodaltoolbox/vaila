import mokka
from util import *

#
# Internal
#


class _Internal:
    def __init__(self):

        # Set attribute 'mokka.app'
        mokka.app = _qMokkaCoreApplicationInstance
        # Set attribute 'mokka.data'
        mokka.data = _qMokkaCoreDataManagerInstance


_internalInstance = _Internal()
