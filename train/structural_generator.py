import numpy as np



def generate_structure(nodes=5, max_parents=2,num_hidden=0, max_allowed_noise=0.1, maximum_depth=None):
    current_elders=[]
    data=[]
    all_parents=set()
    elder_depths={}

    nodes+=num_hidden

    had_max_par=False

    min_adam_count=min([max_parents//2,nodes])
    if min_adam_count<2:min_adam_count=2
    max_adam_count=max([min_adam_count,int(np.sqrt(nodes))])
    adam_count=np.random.randint(min_adam_count,max_adam_count+1)
    if adam_count>nodes//2:
        adam_count=nodes//2
    if adam_count==0:
        adam_count=1
    for i in range(adam_count):
        data.append({"id":i,"parents":[], "type":"adam"})
        current_elders.append(i)
        elder_depths[i]=0
    zoey_count=np.random.randint(0,int(np.ceil(max_allowed_noise*nodes))+1)
    for i in range(zoey_count):
        data.append({"id":i+adam_count,"parents":[], "type":"zoey"})
    remaining_nodes=nodes-(adam_count+zoey_count)
    for i in range(remaining_nodes):
        node_id=i+adam_count+zoey_count
        if max_parents<1:
            max_parents=1
        available_elders=[index for index, depth in elder_depths.items() if maximum_depth is None or depth<maximum_depth]
        #print("available elders", available_elders)
        #print("available depths",[elder_depths[zw] for zw in available_elders])
        #print("all depths",elder_depths)
        parent_count=np.random.randint(1,max_parents+1)
        parent_count=min(parent_count,len(available_elders))
        if len(available_elders)>=max_parents and not had_max_par:
            had_max_par=True
            parent_count=max_parents
        #if parent_count==0:
        #    print("found empty parent count", available_elders, elder_depths, len(data))
            #guarantee that the template fits. Technically slightly restricts what kind of scms can be generated. As there is always a max_par node that contains all adams. But I think that is fine

        parents=sorted(np.random.choice(available_elders,parent_count,replace=False).tolist())
        for par in parents:
            all_parents.add(par)
        data.append({"id":node_id,"parents":parents, "type":"child"})
        current_elders.append(node_id)
        current_depth=max([elder_depths[par] for par in parents])+1
        elder_depths[node_id]=current_depth

    all_parents=list(all_parents)
    hidden_dex=np.random.choice(all_parents,num_hidden,replace=False)
    for i in range(num_hidden):
        index=hidden_dex[i]
        data[index]["type"]="hidden"

    reorder=[int(zw) for zw in np.random.permutation(len(data))]
    #I want to make sure all hidden nodes are at the end
    for i in range(num_hidden):
        index=hidden_dex[i]
        currently=reorder[index]
        who_has=[j for j in range(len(data)) if reorder[j]==len(data)-1-i][0]
        reorder[index]=len(data)-1-i
        reorder[who_has]=currently
    
    data2=[]
    for dic in data:
        data2.append({"id":reorder[dic["id"]],"parents":sorted([reorder[p] for p in dic["parents"]]),"type":dic["type"]})
    data2.sort(key=lambda x:x["id"])


    return data2



def to_adjacency(data):
    n=len(data)
    adjacency_matrix=np.zeros((n,n),dtype=int)
    for dic in data:
        for parent in dic["parents"]:
            adjacency_matrix[parent,dic["id"]]=1
    return adjacency_matrix

def find_max_max_parents(dim, MAX_MAX_PARENTS=10):
    #technically dim-1
    #but thats so boring
    #how about 2*sqrt(dim)
    ret= min(2*int(np.floor(np.sqrt(dim))),MAX_MAX_PARENTS, dim-1)
    if ret<1:ret=1

    #print("dim",dim,"max_max_parents",ret, MAX_MAX_PARENTS)
    return ret

def find_depth(struc):
    depths=[None for zw in struc]
    for i, dic in enumerate(struc):
        if len(dic["parents"])==0:
            depths[i]=0

    while any([zw is None for zw in depths]):
        for i, dic in enumerate(struc):
            if depths[i] is None:
                if all([depths[p] is not None for p in dic["parents"]]):
                    depths[i]=max([depths[p] for p in dic["parents"]])+1
    print(depths)

    return max(depths)

if __name__=="__main__":
    struc=generate_structure(5, 2, 0, maximum_depth=3)
    #print(find_depth(struc))
    #exit()
    maxpar=0
    for dic in struc:
        print(dic)
        if len(dic["parents"])>maxpar:
            maxpar=len(dic["parents"])
    print("maxpar",maxpar)
    #draw the structure with networkx
    from networkx import DiGraph, draw
    from plt import plt

    plt.figure(figsize=(10,6))

    G=DiGraph()
    G.add_nodes_from([dic["id"] for dic in struc])
    G.add_edges_from([(parent,dic["id"]) for dic in struc for parent in dic["parents"]])
    draw(G, with_labels=True)
    plt.savefig("last.png",dpi=300)
    plt.show()


