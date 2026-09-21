from .helper import search_clause, read_tokens

class BElem():
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement __str__() method")

    def __repr__(self):
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement __repr__() method")

    @classmethod
    def ident(self):
        raise NotImplementedError(f"Class {self.__name__} must implement ident() method")

    def to_tokens(self) -> [()]:
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement to_tokens() method")

    @classmethod
    def parse(self, string):
        return self.from_tokens(read_tokens(string))

    @classmethod
    def from_tokens(self, tokens, *args):
        #inside=search_clause(tokens, "boso", "end_boso")
        return classes["boso"].from_tokens(tokens)

    def _match(self, tokens, dic=None):
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement _match() method")

    def match(self, tokens, dic=None):
        dic = {} if dic is None else dic
        dic, tokens=self._match(tokens, dic)
        if len(tokens)>0:
            raise BMatchingError(f"Unmatched tokens: {tokens}")
        return {key:value.to_numpy() for key, value in dic.items()}

    def manifest(self, dic):
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement manifest() method")

    def template(self, dic=None):
        raise NotImplementedError(f"Class {self.__class__.__name__} must implement template() method")

    #def can_match(self, tokens, dic=None):
    #    raise NotImplementedError(f"Class {self.__class__.__name__} must implement can_match() method")

    def can_match(self, tokens, dic=None):
        try:
            self.match(tokens, dic)
            return True
        except BMatchingError:
            return False




classes={}
def register(obj):
    global classes
    classes[obj.ident().lower()] = obj

#def read_object(typ, value_string, *args):
def read_object(tokens):
    if len(tokens)==1:
        typ, value_string=tokens[0][:2]
        args=tokens[0][2:]
        isvar=False
        if value_string.startswith("?"):
            isvar=True
            value_string=value_string[1:]
            return classes["var"](value_string, typ,*args)#currently: cant read values
        typ=typ.lower()
        assert typ in classes, f"Unknown type {typ}"
        return classes[typ].from_tokens([tuple([typ, value_string])],*list(args))
    else:
        typ, logic_type=tokens[0][:2]
        arguments=tokens[0][2:]
        #print("typ",typ,"logic_type",logic_type,"arguments",arguments)
        assert typ=="logic", f"Expected logic type, got {typ}"
        inner=tokens[1:-1]
        logic_type=logic_type.lower()
        assert logic_type in classes, f"Unknown logic type {logic_type}"
        return classes[logic_type].from_tokens(inner, *arguments)
        
        


class BMatchingError(Exception):
    pass
class BManifestError(Exception):
    pass

