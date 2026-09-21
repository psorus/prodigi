from .belem import BElem, register, classes, read_object, BMatchingError, BManifestError
from .helper import tabify, search_clause, iterate_blocks
from .helper import copy_dic, copy_tokens
from .arraylike import store_value, get_value

class BComment(BElem):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return ""

    def __repr__(self):
        return "BComment()"

    @classmethod
    def ident(self):
        return 'Comment'

    def to_tokens(self):
        return []

    @classmethod
    def from_tokens(self, tokens, *args):
        return BComment()

    def _match(self, tokens, dic=None):
        if dic is None:
            dic={}
        return dic, tokens

    def manifest(self, dic):
        return []

    def template(self, dic=None):
        return []



register(BComment)
