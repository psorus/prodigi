from .belem import BElem, register, classes, read_object
from .helper import tabify, search_clause, iterate_blocks
from .arraylike import array_from_number, set_index

class BLoop(BElem):
    def __init__(self, child, counter="i", minimum=None, maximum=None, condition=None):
        self.child=child
        self.counter=counter
        if minimum is None and maximum is None:
            raise ValueError("Loop must have at least a minimum or maximum")
        elif maximum is None:
            maximum=minimum
            minimum=0
        elif minimum is None:
            minimum=0

        if hasattr(minimum, "isdigit") and minimum.isdigit():
            minimum=int(minimum)
        if hasattr(maximum, "isdigit") and maximum.isdigit():
            maximum=int(maximum)

        self.minimum=minimum
        self.maximum=maximum
        self.has_condition=condition is not None
        self.condition=condition
        super().__init__()

    def __str__(self):
        if self.has_condition:
            return f"Loop({self.counter}={self.minimum}=>{self.maximum} if {self.condition})\n"+tabify(str(self.child))
        return f"Loop({self.counter}={self.minimum}=>{self.maximum})\n"+tabify(str(self.child))

    def __repr__(self):
        if self.has_condition:
            return f"BLoop({repr(self.child)}, {repr(self.counter)}, {repr(self.minimum)}, {repr(self.maximum)}, {repr(self.condition)})"
        return f"BLoop({repr(self.child)}, {repr(self.counter)}, {repr(self.minimum)}, {repr(self.maximum)})"

    @classmethod
    def ident(self):
        return 'Loop'

    def to_tokens(self):
        ret=[]
        if self.has_condition:
            ret.append(("LOGIC","loop", str(self.counter), str(self.minimum), str(self.maximum), *self.condition))
        else:
            ret.append(("LOGIC","loop", str(self.counter), str(self.minimum), str(self.maximum)))
        ret.extend(self.child.to_tokens())
        ret.append(("LOGIC", "end_loop"))
        return ret

    @classmethod
    def from_tokens(self, tokens, *args):
        #print("trying to reconstruct from tokens", tokens)
        #print("got args", *args)
        #print("loop got args", *args)
        child=classes["boso"].from_tokens(tokens)
        return BLoop(child, *args)

    def _check_condition(self,index, dic, condition):
        ctyp, var=condition[:1],condition[1:]
        value=dic[var].value()
        if ctyp=="b":
            return index<value
        elif ctyp=="o":
            return index>value
        elif ctyp=="e":
            return index==value
        elif ctyp=="d":
            return index!=value
        else:
            raise ValueError(f"Invalid condition type {ctyp} in loop")


    def _match(self, tokens, dic=None):
        if dic is None:
            dic={}
        minimum=self.minimum
        maximum=self.maximum
        if minimum in dic:
            minimum=dic[minimum].value()
        if maximum in dic:
            maximum=dic[maximum].value()
        for i in range(minimum, maximum):
            if self.has_condition:
                if not self._check_condition(i,dic, self.condition):
                    continue
            dic=set_index(dic, self.counter, i)
            dic, tokens=self.child._match(tokens, dic)
        if self.counter in dic:del dic[self.counter]
        return dic, tokens

    def manifest(self, dic):
        ret=[]
        minimum=self.minimum
        maximum=self.maximum
        if minimum in dic:
            minimum=dic[minimum]#.value()
        if maximum in dic:
            maximum=dic[maximum]#.value()
        for i in range(minimum, maximum):
            if self.has_condition:
                if not self._check_condition(i,dic, self.condition):
                    continue
            dic=set_index(dic, self.counter, i)
            ret.extend(self.child.manifest(dic))
        if self.counter in dic:del dic[self.counter]
        return ret
    def template(self, dic=None):
        if dic is None:
            dic={}
        ret=[]
        minimum=self.minimum
        maximum=self.maximum
        if minimum in dic:
            minimum=dic[minimum]#.value()
        if maximum in dic:
            maximum=dic[maximum]#.value()
        for i in range(minimum, maximum):
            if self.has_condition:
                if not self._check_condition(i,dic, self.condition):
                    continue
            dic=set_index(dic, self.counter, i)
            ret.extend(self.child.template(dic))
        if self.counter in dic:del dic[self.counter]
        return ret
        


    #def can_match(self, tokens, dic=None):
    #    if dic is None: dic={}
    #    if self.minimum in dic:
    #        self.minimum=dic[self.minimum].value()
    #    if self.maximum in dic:
    #        self.maximum=dic[self.maximum].value()
    #
    #    for i in range(self.minimum, self.maximum):
    #        valid, ntokens, dic=self.child.can_match(tokens)
    #        tokens=ntokens
    #        if not valid:return False, tokens, dic
    #    return True, tokens, dic




register(BLoop)
