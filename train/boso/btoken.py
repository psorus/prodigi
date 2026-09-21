from .belem import BElem, register, BMatchingError


class BToken(BElem):
    def __init__(self, key):
        self.key = key
        super().__init__()

    def __str__(self):
        return "TOKEN "+str(self.key)

    def __repr__(self):
        return f"BToken({repr(self.key)})"

    @classmethod
    def ident(self):
        return 'TOKEN'

    def to_tokens(self):
        return [("TOKEN",self.key)]

    @classmethod
    def from_tokens(self, tokens, *args):
        assert len(tokens)==1
        token=tokens[0]
        assert len(token)==2
        typ, value=token
        assert typ.lower()=="token"
        return BToken(value)

    def _match(self, tokens, dic=None):
        if dic is None:dic={}
        if len(tokens)==0:raise BMatchingError("too few tokens left")
        if tokens[0][0].lower()!="token":raise BMatchingError("expected token, got "+str(tokens[0]))
        if tokens[0][1].lower()!=self.key.lower():raise BMatchingError("expected token "+str(self.key)+", got "+str(tokens[0][1]))
        return dic, tokens[1:]

    def manifest(self, dic):
        return self.to_tokens()

    def template(self, dic=None):
        return self.to_tokens()
        

    #def can_match(self, tokens, dic=None):
    #    if dic is None:dic={}
    #    return len(tokens)>0 and tokens[0][0].lower()=="token" and tokens[0][1].lower()==self.key.lower(), tokens[1:], dic


register(BToken)
