from .belem import BElem, register, classes, read_object, BMatchingError, BManifestError
from .helper import tabify, search_clause, iterate_blocks
from .helper import copy_dic, copy_tokens
from .arraylike import store_value, get_value

class BOptional(BElem):
    def __init__(self, child, ident=None):
        self.child=child
        self.ident=ident
        self.has_ident=ident is not None
        super().__init__()

    def __str__(self):
        if self.has_ident:
            return f"Optional({self.ident})\n"+tabify(str(self.child))
        else:
            return "Optional\n"+tabify(str(self.child))

    def __repr__(self):
        if self.has_ident:
            return f"BOptional({repr(self.child)}, ident={repr(self.ident)})"
        return f"BOptional({repr(self.child)})"

    @classmethod
    def ident(self):
        return 'Optional'

    def to_tokens(self):
        ret=[]
        if self.has_ident:
            ret.append(("LOGIC","optional", self.ident))
        else:
            ret.append(("LOGIC","optional"))
        ret.extend(self.child.to_tokens())
        ret.append(("LOGIC", "end_optional"))
        return ret

    @classmethod
    def from_tokens(self, tokens, *args):
        ident=None
        if len(args)>0:
            ident=args[0]
        return BOptional(classes["boso"].from_tokens(tokens),ident=ident)

    def _match(self, tokens, dic=None):
        if dic is None:
            dic={}
        #print("prematch", dic)
        try:
            new_dic, new_tokens=self.child._match(copy_tokens(tokens), copy_dic(dic))
            if self.has_ident:
                new_dic=store_value(new_dic, self.ident, True)
            return new_dic, new_tokens
        except BMatchingError:
            if self.has_ident:
                dic=store_value(dic, self.ident, False)
            return dic, tokens

    def manifest(self, dic):
        if self.has_ident:
            try:
                shall=bool(get_value(dic, self.ident))
                if shall:
                    return self.child.manifest(dic)
                else:
                    return []
            except KeyError:
                pass
        try:
            return self.child.manifest(dic)
        except BManifestError:
            return []
    def template(self, dic=None):
        if dic is None:
            dic={}
        if self.has_ident:
            try:
                shall=bool(get_value(dic, self.ident))
                if shall:
                    return self.child.template(dic)
                else:
                    return []
            except KeyError:
                pass
        try:
            return self.child.template(dic)
        except BManifestError:
            return []
    

    #def can_match(self, tokens, dic=None):
    #    if dic is None:dic={}
    #    valid, new_tokens, dic=self.child.can_match(tokens, dic)
    #    if valid:
    #        #print("can_match optional: valid")
    #        return True, new_tokens, dic
    #    #print("can_match optional: not valid, but optional, so valid")
    #    return True, tokens,dic




register(BOptional)
