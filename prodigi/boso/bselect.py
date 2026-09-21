from .belem import BElem, register, classes, read_object, BMatchingError, BManifestError
from .helper import tabify, search_clause, iterate_blocks
from .helper import copy_dic, copy_tokens
from .arraylike import store_value, get_value


class BOption(BElem):
    def __init__(self, child, ident, value):
        self.child=child
        self.ident=ident
        self.value=value
        super().__init__()

    def __str__(self):
        return f"Option({self.ident}={self.value})\n"+tabify(str(self.child))

    def __repr__(self):
        return f"BOption({repr(self.child)}, ident={repr(self.ident)}, value={repr(self.value)})"

    @classmethod
    def ident(self):
        return 'Option'

    def to_tokens(self):
        ret=[]
        ret.append(("LOGIC","option", self.ident, self.value))
        ret.extend(self.child.to_tokens())
        ret.append(("LOGIC", "end_option"))
        return ret

    @classmethod
    def from_tokens(self, tokens, *args):
        ident, value=args[:2]
        return BOption(classes["boso"].from_tokens(tokens), ident=ident, value=value)

    def _match(self, tokens, dic=None):
        if dic is None:
            dic={}
        new_dic, new_tokens=self.child._match(tokens, dic)
        return new_dic, new_tokens

    def manifest(self, dic):
        return self.child.manifest(dic)

    def template(self, dic=None):
        if dic is None:
            dic={}
        return self.child.template(dic)


class BSelect(BElem):
    def __init__(self, children, ident):
        self.children={c.value:c for c in children}
        self.ident=ident
        super().__init__()

    def __str__(self):
        return f"Select({self.ident})\n"+tabify("\n".join(str(zw) for zw in self.children.values()))


    def __repr__(self):
        return f"BSelect({repr(list(self.children.values()))}, ident={repr(self.ident)})"

    @classmethod
    def ident(self):
        return 'Select'

    def to_tokens(self):
        ret=[]
        ret.append(("LOGIC","select", self.ident))
        for child in self.children.values():
            ret.extend(child.to_tokens())
        ret.append(("LOGIC", "end_select"))
        return ret

    @classmethod
    def from_tokens(self, tokens, *args):
        ident=args[0]
        children=[]
        for block in iterate_blocks(tokens):
            args=block[0][2:]
            block=block[1:-1]
            children.append(BOption.from_tokens(block, *args))
        return BSelect(children, ident=ident)

    def _match(self, tokens, dic=None):
        if dic is None:
            dic={}
        #print("prematch", dic)
        last_error=None
        for value, child in self.children.items():
            try:
                new_dic, new_tokens=child._match(copy_tokens(tokens), copy_dic(dic))
                new_dic=store_value(new_dic, self.ident, value)
                return new_dic, new_tokens
            except BMatchingError as e:
                last_error=e
                continue
        raise BMatchingError("No option matched"+str(last_error))

    def manifest(self, dic):
        value=get_value(dic, self.ident)
        return self.children[value].manifest(dic)

    def template(self, dic=None):
        if dic is None:
            dic={}
        value=get_value(dic, self.ident)
        return self.children[value].template(dic)

    #def can_match(self, tokens, dic=None):
    #    if dic is None:dic={}
    #    valid, new_tokens, dic=self.child.can_match(tokens, dic)
    #    if valid:
    #        #print("can_match optional: valid")
    #        return True, new_tokens, dic
    #    #print("can_match optional: not valid, but optional, so valid")
    #    return True, tokens,dic




register(BSelect)
