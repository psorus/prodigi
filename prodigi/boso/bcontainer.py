from .belem import BElem, register, classes, read_object
from .helper import tabify, search_clause, iterate_blocks

class BContainer(BElem):
    def __init__(self, children):
        self.children=children
        super().__init__()

    def __str__(self):
        inner=[str(zw) for zw in self.children]
        inner=[zw for zw in inner if zw.strip()!=""]
        return "Boso\n"+tabify("\n".join(inner))

    def __repr__(self):
        return f"BContainer({','.join([repr(zw) for zw in self.children])})"

    @classmethod
    def ident(self):
        return 'BOSO'

    def to_tokens(self):
        ret=[]
        ret.append(("LOGIC","boso"))
        for zw in self.children:
            ret.extend(zw.to_tokens())
        ret.append(("LOGIC", "end_boso"))
        return ret

    @classmethod
    def from_tokens(self, tokens, *args):
        #tokens=search_clause(tokens, "boso", "end_boso")
        #print(tokens)
        #exit()
        children=[]
        for token in iterate_blocks(tokens):
            #print(token)
            #continue

            obj=read_object(token)
            #typ=token[0].lower()
            #obj=read_object(typ, token[1], *token[2:])
            children.append(obj)
        #exit()
        if len(children)==1:
            return children[0]
        return BContainer(children)

    def _match(self, tokens, dic=None):
        if dic is None:
            dic={}
        #if tokens[0][0].lower()=="token" and tokens[0][1].lower()=="boso":
        #    tokens=tokens[1:]
        #if tokens[-1][0].lower()=="token" and tokens[-1][1].lower()=="end_boso":
        #    tokens=tokens[:-1]

        for child in self.children:
            
            dic, tokens=child._match(tokens, dic)
        return dic, tokens

    def manifest(self, dic):
        ret=[]
        for child in self.children:
            ret.extend(child.manifest(dic))
        return ret
    def template(self, dic=None):
        if dic is None:dic={}
        ret=[]
        for child in self.children:
            ret.extend(child.template(dic))
        return ret


    #def can_match(self, tokens, dic=None):
    #    if dic is None:
    #        dic={}
    #    for child in self.children:
    #        works, ntokens, dic=child.can_match(tokens, dic)
    #        #print(f"matching in container{len(self.children)}", repr(child), tokens, ":", works)
    #        tokens=ntokens
    #        if not works:
    #            return False, tokens, dic
    #    return True, tokens, dic




register(BContainer)
