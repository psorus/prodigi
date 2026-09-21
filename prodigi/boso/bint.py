from .belem import BElem, register, BMatchingError


class BInt(BElem):
    def __init__(self, value):
        self.value = value
        super().__init__()

    def __str__(self):
        return self.ident()+" "+str(self.value)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.value)})"

    @classmethod
    def ident(self):
        return 'INT'

    def to_tokens(self):
        return [(self.ident(),self.value)]

    @classmethod
    def from_tokens(self, tokens, *args):
        assert len(tokens)==1
        token=tokens[0]
        assert len(token)==2
        typ, value=token
        assert typ.lower()==self.ident().lower()
        return BInt(value)

    def _match(self, tokens, dic=None):
        if dic is None:dic={}
        if len(tokens)==0:raise BMatchingError("too few tokens left")
        if tokens[0][0].lower()!=self.ident().lower():raise BMatchingError(f"different identify found", tokens[0][0], self.ident(), tokens[0][1], self.value)
        value=self.value
        if value in dic:
            value=dic[value].value()
        if str(tokens[0][1])!=str(value): raise BMatchingError(f"wrong value found (mine {value}!={tokens[0][1]}), types {type(value)},{type(tokens[0][1])}")
        return dic, tokens[1:]

    def manifest(self, dic):
        value=self.value
        if value in dic:
            #print("updating", value, "to", dic[value])
            value=dic[value]
            if hasattr(value, "value"):
                value=value.value()
        return [(self.ident(), value)]
    def template(self, dic=None):
        if dic is None:dic={}
        return self.manifest(dic)

    #def can_match(self, tokens, dic=None):
    #    if dic is None: dic={}
    #    return len(tokens)>0 and tokens[0][0].lower()==self.ident().lower() and tokens[0][1]==self.value, tokens[1:], dic


register(BInt)
