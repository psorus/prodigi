from .belem import BElem, register, BMatchingError, BManifestError
from .arraylike import Arraylike, split_indice, store_value, get_value

class BVar(BElem):
    def __init__(self, name, typ,*args, value=None):
        self.name=name
        self.typ=typ.lower()
        self.has_value=value is not None
        self.value=value
        

    def __str__(self):
        if self.has_value:
            return f"{self.name}({self.typ})={str(self.value)}"
        else:
            return f"{self.name}({self.typ})"

    def __repr__(self):
        if self.has_value:
            return f"BVar({self.name}, {self.typ}, {self.value})"
        else:
            return f"BVar({self.name}, {self.typ})"

    @classmethod
    def ident(self):
        return "VAR"

    def to_tokens(self):
        if self.has_value:
            return self.value.to_tokens()
        else:
            return [(self.typ, "?"+self.name)]


    @classmethod
    def from_tokens(self, tokens, *args):
        raise Exception(f"Helper class {self.__class__} should never be instantiated from tokens")

    def _match(self, tokens, dic=None):
        if dic is None:dic={}
        if len(tokens)==0:
            raise BMatchingError(f"no more tokens to match, expected {self.typ}")
        current=tokens[0]
        typ, value=current[:2]
        typ=typ.lower()
        if typ!=self.typ.lower():
            raise BMatchingError(f"types dont match, expected {self.typ}, got {typ}, variable name {self.name}")
        #assert typ==self.typ.lower(),f"types dont match, expected {self.typ}, got {typ}"
        dic=store_value(dic, self.name, value)
        #variable_name, variable_indice= split_indice(self.name, dic)
        #if not variable_name in dic:
        #    dic[variable_name]=Arraylike()
        #dic[variable_name].register(variable_indice, value)
        #dic[self.name]=value
        return dic, tokens[1:]

    def manifest(self, dic):
        variable_name, variable_indice=split_indice(self.name, dic)
        if not variable_name in dic:
            raise BManifestError(f"variable {variable_name} not found in manifest")
        value=dic[variable_name]
        if len(variable_indice)>0:value=value[variable_indice]
        return [(self.typ, value)]

    def template(self, dic=None):
        if dic is None:dic={}
        try:
            value=get_value(dic, self.name)
            return [(self.typ.upper(), value)]
        except KeyError:
            typ=self.typ
        #variable_name, variable_indice=split_indice(self.name, dic)
        #if variable_name in dic:
        #    value=dic[variable_name]
        #    return [(self.typ.upper(), value)]
        #else:
        #    typ=self.typ

        return [("FILL", typ)]


    #def can_match(self, tokens, dic=None):
    #    if dic is None:dic={}
    #    if not tokens:
    #        return False, tokens, dic
    #    current=tokens[0]
    #    typ, value=current[:2]
    #    typ=typ.lower()
    #    variable_name, variable_indice= split_indice(self.name, dic)
    #    if not variable_name in dic:
    #        dic[variable_name]=Arraylike()
    #    dic[variable_name].register(variable_indice, value)
    #    return typ==self.typ.lower(), tokens[1:], dic


register(BVar)
