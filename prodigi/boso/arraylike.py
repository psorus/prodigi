import numpy as np

class Arraylike():
    #works like a list, but for higher dimensional indexing and without requiring to add elements in order
    def __init__(self):
        self.storage={}
        self.dim=None

    def __str__(self):
        return f"Arraylike(dim={self.dim}, storage={self.storage})"

    def copy(self):
        new_array=Arraylike()
        for key, value in self.storage.items():
            new_array.storage[key]=value
        new_array.dim=self.dim
        return new_array

    def register(self, indice, value):
        if self.dim is None:
            self.dim=len(indice)
        else:
            assert self.dim==len(indice), "All indices must have the same dimension"
        self.storage[tuple(indice)]=value

    def get_index(self, indice):
        return self.storage[tuple(indice)]

    def is_value(self):
        return self.dim==0 and len(self.storage)==1

    def value(self):
        #only works when dim==0 and storage has only one element
        assert self.dim==0, "cant convert to value if dim is not 0"
        assert len(self.storage)==1, "cant convert to value if there is more than one element"
        return list(self.storage.values())[0]

    def to_numpy(self):
        if self.dim==0:
            return self.storage[()]
        max_indice=np.max(list(self.storage.keys()), axis=0)
        example=list(self.storage.values())[0]
        dtype=type(example)
        stringify=False
        if dtype==str or dtype==np.str_:
            dtype=object
            stringify=True
        arr=np.zeros(max_indice+1, dtype=dtype)
        for indice, value in self.storage.items():
            arr[indice]=value
        if stringify:
            arr=arr.astype(str)
        return arr

def array_from_number(number):
    arr=Arraylike()
    arr.register((), number)
    return arr

def set_index(dic, name, value):
    if name in dic and dic[name].is_value():
        dic[name].register((), value)
    else:
        dic[name]=array_from_number(value)
    return dic

def store_value(dic, name, value):
    variable_name, variable_indice=split_indice(name, dic)
    if not variable_name in dic:
        dic[variable_name]=Arraylike()
    dic[variable_name].register(variable_indice, value)
    return dic

def get_value(dic, name):
    variable_name, variable_indice=split_indice(name, dic)
    if not variable_name in dic:
        raise KeyError(f"{variable_name} not found in dictionary")
    if hasattr(dic[variable_name],"get_index"):
        return dic[variable_name].get_index(variable_indice)
    else:
        curr=dic[variable_name]
        if len(variable_indice)>0:
            curr=curr[variable_indice]
        return curr




def split_indice(stri, dic):
    #given a string like arr[0][1], returns arr, (0,1)
    if not "[" in stri:
        return stri, ()
    name=stri[:stri.index("[")]
    indices=[]
    while True:
        if not "[" in stri:
            break
        stri=stri[stri.index("[")+1:]
        indice=stri[:stri.index("]")]
        indice=indice.strip()
        if indice.isdigit():
            indice=int(indice)
        else:
            indice=dic[indice].value()
        indices.append(int(indice))
    return name, tuple(indices)
    























