
def search_clause(token_list, clause, end_clause):
    #list is like [(token, type), (token, type), ...]
    #we are searching for a block that is contained in [(token, clause) --- > (token, end_clause)]
    #in case the whole block is missing: return everything
    has_start=any([zw[1]==clause for zw in token_list if zw[0].lower()=="token" and len(zw)>1])
    if not has_start:return token_list
    inside=False
    block=[]
    for zw in token_list:
        if len(zw)>=2:
            typ, token=zw[:2]
            typ=typ.lower()
            #if typ!="token":continue
            if typ=="logic" and token==clause:
                inside=True
                continue
            if typ=="logic" and token==end_clause:
                inside=False
                return block
        if inside:
            block.append(zw)
    return block

def read_tokens(string):
    #searches for all occurences of <?? ?? ??> in a string. returns them as a list in order 
    tokens=[]
    while "<" in string:
        string=string[string.index("<")+1:]
        if ">" not in string:break
        current=string[:string.index(">")]
        string=string[string.index(">")+1:]
        current=current.strip()
        current=tuple(current.split(" "))
        tokens.append(current)
    return tokens

def tabify(string, spaces=2):
    #adds tabs to the beginning of each line in a string
    addon=" "*spaces
    lines=string.split("\n")
    string="\n".join([addon+line for line in lines])
    return string

def find_end_index(tokens, clause, end_clause):
    #old style fails because of nesting
    #end_index=[i for i, zw in enumerate(tokens) if len(zw)>1 and zw[0].lower()=="logic" and zw[1]==end_clause]
    #end_index=end_index[0] if len(end_index)>0 else len(tokens)-1
    depth=0
    for i, token in enumerate(tokens):
        if len(token)>1 and token[0].lower()=="logic":
            current=token[1]
            if current==clause:
                depth+=1
            elif current==end_clause:
                depth-=1
                if depth==0:
                    return i
    return len(tokens)-1



def iterate_blocks(tokens):
    while len(tokens)>0:
        if len(tokens[0])>0 and tokens[0][0].lower()=="logic":
            logic_elem=tokens[0][1]
            end_clause="end_"+logic_elem
            end_index=find_end_index(tokens, logic_elem, end_clause)
            current=tokens[:end_index+1]
            #print("initalizing logic block", current)
            yield current
            tokens=tokens[end_index+1:]
        else:
            yield [tokens[0]]
            tokens=tokens[1:]

    #for zw in tokens:
    #    yield [zw]

def copy_dic(dic):
    new_dic={}
    for key, value in dic.items():
        new_dic[key]=value.copy()
    return new_dic
def copy_tokens(tokens):
    new_tokens=[]
    for zw in tokens:
        new_tokens.append(tuple(zw))
    return new_tokens






