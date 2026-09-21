from .belem import BElem, register
from .bint import BInt


class BBool(BInt):
    def __init__(self, value):
        super().__init__(value)

    @classmethod
    def ident(self):
        return "BOOL"



register(BBool)
