'''
@author: Yifan Zhu
@e-mail: mayz0571@qq.com
'''
import networkx as nx
from networkx.algorithms import bipartite
import csv
import random
import pandas as pd
import numpy as np
import os
import datetime
import time
import copy

#构建初始海运网络
def foundnetwork():
    df = pd.read_csv("data/edge capacity distance.csv", encoding='gbk')
    G2015 = nx.Graph()
    for index, row in df.iterrows():
        G2015.add_edge(row[0], row[1], weight=row[4])

    return G2015

#构建初始二分网络
def foundbipartitenetwork(df,ports):
    #df = pd.read_csv("data/All routes of ports 2015.csv", encoding='gbk')
    #ports = pd.read_csv("data/port_capacity_bc_wbc.csv", encoding='gbk')
    G2015 = nx.Graph()
    G2015.add_nodes_from(range(1,1623), bipartite=0)
    G2015.add_nodes_from(ports['port'], bipartite=1)
    for index, row in df.iterrows():
        G2015.add_edge(row[0], row[13], weight=row[14])

    Gm = bipartite.project(G2015, ports['port'])
    Gl = bipartite.project(G2015, range(1,1623))
    print(Gm.number_of_nodes())
    print(Gl.number_of_nodes())
    return G2015,Gm

#计算初始负载（不变量）
def load_index(G2015):

    #effe=nx.global_efficiency(G2015)
    bc1 = nx.betweenness_centrality(G2015)
    print(bc1)
    bc2 = nx.betweenness_centrality(G2015, weight='weight')
    print(bc2)
    load = nx.degree(G2015, weight='weight')
    print(load)

    bc = pd.DataFrame.from_dict(bc1, orient='index', columns=['bc'])
    bc = bc.reset_index().rename(columns={'index': 'port'})
    wbc = pd.DataFrame.from_dict(bc2, orient='index', columns=['wbc'])
    wbc = wbc.reset_index().rename(columns={'index': 'port'})
    newload = pd.DataFrame(load, columns=['port', 'load'])
    bc.to_csv("output/edge bc.csv", encoding='gbk')
    wbc.to_csv("output/edge wbc.csv", encoding='gbk')
    newload.to_csv("output/edge load.csv", encoding='gbk')

#计算容量,b为可调参数β
def calculata_volume(port_capacity,b):
    port_capacity['volume']=port_capacity['total capacity'].values*(1+b)
    return port_capacity

def restore_replace(original_port_of_replace,replaceport,originalport,routeID,weight):
    if (replaceport in original_port_of_replace):
        dict1=original_port_of_replace[replaceport]
        if(routeID in dict1):
            dict2=dict1[routeID]
            if(originalport not in dict2):
                dict2.setdefault('originalport',weight)
        else:
            dict1.setdefault(routeID, {originalport:weight})
    else:
        original_port_of_replace.setdefault(replaceport,{routeID:{originalport:weight}})
    if(replaceport=='Dalian' and routeID=='166'):
        print(original_port_of_replace[replaceport])

#负载重分配
def cascading(choose_replaceport,replaceport_replaceport,df,G2015,node,p,original_port_of_replace):
    lines=df[df['port']==node]
    for routeID in lines['Route ID'].values:
        probability=np.random.randint(0,100)/100 #np.random.randint方法在[0,101)均匀随机抽取整数 该语句可验证：print(Counter([np.random.randint(0,101) <25 for _ in range(1000000)]))
        if( probability < p):
            weight=G2015.get_edge_data(routeID,node)['weight']
            cr=choose_replaceport[(choose_replaceport['Route ID'] == routeID)]
            cr=cr[ cr['port'] == node]
            if(len(cr)==0):
                #cr=replaceport_replaceport[replaceport_replaceport['Route ID'] == routeID ]
                #cr=cr[cr['port'] == node]
                print(node,int(routeID),original_port_of_replace[replaceport])
                print(original_port_of_replace[replaceport][str(int(routeID))])

            r=list(cr['replace port'])
            replace=eval(r[0])
            top_nodes = {n for n, d in G2015.nodes(data=True) if d["bipartite"] == 1}

            replace=list(set(replace)&top_nodes)
            replace.sort()
            #print('replace port', replace)
            if(len(replace)==0): #没有替代港就跳港
                weight = G2015.degree(weight='weight')[routeID]
                edge = G2015.edges(routeID)
                #print(edge)
                if (len(edge) == 1):
                    G2015.remove_node(routeID)
                else:
                    G2015.add_edges_from(edge, weight=weight / (len(edge) - 1))
                    df.at[df[df['Route ID'] == routeID].index, ['average capacity']] = weight / (len(edge) - 1)
            else:
                replaceport=random.choice(replace)#选择的替代港
                if(nx.has_path(G2015,replaceport,routeID)):
                    if(nx.shortest_path_length(G2015,replaceport,routeID)==1):
                        weight=weight+(G2015.get_edge_data(replaceport,routeID)['weight'])
                        df.at[df[(df['Route ID'] == routeID)&(df['port']== replaceport)].index, ['average capacity']] = weight
                    else:
                        df.at[str(routeID) + replaceport] = [routeID, '', '', '', '', '', '', '', '', '', '', '', '',
                                                              replaceport, weight]
                G2015.add_edge(replaceport,routeID,weight=weight)
                restore_replace(original_port_of_replace,replaceport,node,str(int(routeID)),weight)
                if(node=='Dalian'):
                    print(original_port_of_replace['Dalian'])

        elif(probability>=p):
            weight=G2015.degree(weight='weight')[routeID]
            edge=G2015.edges(routeID)
            #print(edge)
            if (len(edge) == 1):
                G2015.remove_node(routeID)
            else:
                G2015.add_edges_from(edge,weight=weight/(len(edge)-1))
                df.at[df[df['Route ID'] == routeID].index, ['average capacity']] = weight/(len(edge)-1)

    G2015.remove_node(node)
    df=df[df['port']!=node]

    top_nodes = {n for n, d in G2015.nodes(data=True) if d["bipartite"] == 1}
    if(len(top_nodes)==0):
        return G2015, df,original_port_of_replace, 0
    else:
        return G2015,df,original_port_of_replace,1

#重新计算负载
def new_load_index(firstload,G2015):
    load=nx.degree(G2015, weight='weight')
    newload = pd.DataFrame(load, columns=['port','load'])
    result = pd.merge(firstload, newload, how='right', on='port')
    return result


def nodeefficiency(G):
    sumeff = 0 #图论中最短路径计算的网络效率
    dict = {}
    sumweighteff = 0  # 真实距离为权重的网络效率
    dictweight={}
    sumgraph_weighteff = 0  # 先按图论距离排，再按真实距离为权重找最短路径的网络效率
    dictgraph_weight = {}
    for u in G.nodes():  # 遍历图F的每个点
        eff = 0
        path = nx.shortest_path_length(G,source=u)  # 在网络G中计算从u开始到其他所有节点（注意包含自身）的最短路径长度。如果两个点之间没有路径，那path里也不会存储这个目标节点（比前面的代码又省了判断是否has_path的过程）
        for v in path.keys():  # path是一个字典，里面存了所有目的地节点到u的最短路径长度
            if u != v:  # 如果起终点不同才累加计算效率
                eff += 1 / path[v]
                sumeff += 1 / path[v]
                allShortestPath=nx.all_shortest_paths(G, source=u, target=v)
                tmp=0
                for sp in allShortestPath:
                    dis=0
                    for i in range(len(sp)-1):
                        dis+=G.get_edge_data(sp[i],sp[i+1])['weight']
                    if(tmp!=0 and tmp<dis):
                        tmp=tmp
                    else:
                        tmp=dis
                sumgraph_weighteff+=1/tmp
        if (G.number_of_nodes() - 1 > 0):
            dict[u]=eff/(G.number_of_nodes()-1)
        else:
            dict[u] = eff

        eff = 0
        path_weight = nx.shortest_path_length(G, source=u, weight='weight')  # 加上真实海运距离的最短路径
        for v in path_weight.keys():  # path是一个字典，里面存了所有目的地节点到u的最短路径长度
            if u != v:  # 如果起终点不同才累加计算效率
                eff += 1 / path_weight[v]
                sumweighteff += 1 / path_weight[v]
                #nx.all_simple_paths(G, source=2, target=4, cutoff=path[v])
        if (G.number_of_nodes() - 1 > 0):
            dictweight[u]=eff/(G.number_of_nodes()-1)
        else:
            dictweight[u] = eff

    if((G.number_of_nodes() * (G.number_of_nodes() - 1))!=0):
        result = (1 / (G.number_of_nodes() * (G.number_of_nodes() - 1))) * sumeff #
        resultweight= (1 / (G.number_of_nodes() * (G.number_of_nodes() - 1))) * sumweighteff #
        resultgraph_weight=(1 / (G.number_of_nodes() * (G.number_of_nodes() - 1))) * sumgraph_weighteff #
    else:
        resultweight=sumweighteff
        result = sumeff
        resultgraph_weight=sumgraph_weighteff
    result977=(1 / 977 / 976) * sumeff #N取977下的网络效率

    return result,result977,resultweight,resultgraph_weight


def mkdir(path):
  # 去除首位空格
  path=path.strip()
  # 去除尾部 \ 符号
  path=path.rstrip("\\")

  # 判断路径是否存在
  # 存在     True
  # 不存在   False
  isExists=os.path.exists(path)

  # 判断结果
  if not isExists:
    # 如果不存在则创建目录
    print(path+' 创建成功')
    # 创建目录操作函数
    os.makedirs(path)
    return True
  else:
    # 如果目录存在则不创建，并提示目录已存在
    print(path+' 目录已存在')
    return False


def start_up():
    starttime = datetime.datetime.now()
    df = pd.read_csv('data/All routes of ports 2015.csv', encoding='gbk')
    port_capacity = pd.read_csv("data/port_capacity_bc_wbc.csv", encoding='gbk')
    choose_replaceport = pd.read_csv("data/each route's replace ports set of each port.csv",
                                     encoding='gbk')
    replaceport_replaceport = pd.read_csv("data/each route's replace ports set of each replace port.csv",
                                          encoding='gbk')
    allports = pd.read_csv("data/port_capacity_bc_wbc.csv", encoding='gbk')  # 读取所有港口
    df0 = copy.deepcopy(df)
    G, Gm = foundbipartitenetwork(df, port_capacity)  # 航线和港口二分网络,单模网络
    G0 = copy.deepcopy(G)  # 复制初始网络，之后G会变化

    ps = [0.2]
    bs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    seed = 30
    for p in ps:
        for b in bs:
            firstload = calculata_volume(port_capacity, b)
            # eff0,eff9770,effweight0 = nodeefficiency(Gm)
            # largest_components0 = max(nx.connected_components(Gm), key=len) #已知数据
            ######## begin 生成带表头的记录文件
            '''with open("expected output/results/result seed=" + str(seed) + " p=" + str(p) + " b=" + str(b) + ".csv",
                      'a+', encoding='utf-8', newline='') as fw2:
                writer = csv.writer(fw2)
                writer.writerow(
                    ['port', 'Initial efficiency', 'Initial efficiency(No.977)', 'Initial efficiency(SR)',
                     'Initial len(largest_components)',
                     'Termination len(largest_components)', 'Initial BN No.nodes',
                     'Termination BN No.nodes', 'Initial BN No.edges', 'Termination BN No.edges', 'Initial MN No.nodes',
                     'Termination MN No.nodes', 'Initial MN No.edges', 'Termination MN No.edges'])'''
            ########## end
            random.seed(seed)
            np.random.seed(seed)  # 指定种子
            # 模拟每一个港口失效后的级联失效过程
            i=0
            for rows in allports['port'].values:
                original_port_of_replace = {}
                i+=1
                if(i>=0 and rows=='Shanghai'):
                    print(rows, '###############################')
                    G = G0.copy()
                    df = df0.copy()
                    G, df, original_port_of_replace,flag = cascading(choose_replaceport, replaceport_replaceport, df, G, rows, p,original_port_of_replace)  # 第一次失效
                    data = new_load_index(firstload, G)  # 计算重分配后的负载
                    # print(data)
                    nodes = data[data['load'] > data['volume']]  # 找出失效的节点集合
                    # print(nodes)
                    while (len(nodes) > 0):
                        for row in nodes['port'].values:
                            G, df, original_port_of_replace,flag = cascading(choose_replaceport, replaceport_replaceport, df, G, row, p,original_port_of_replace)
                        if (flag == 0):
                            print("网络崩溃@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                            break
                        data = new_load_index(firstload, G)  # 计算重分配后的负载
                        nodes = data[data['load'] > data['volume']]  # 找出失效的节点集合
                    print(original_port_of_replace)
                    print(1/0)
                    #print("nodes:", G.nodes())  # 输出全部的节点
                    #print("edges:", G.edges())  # 输出全部的边
                    #print("number of nodes:", G.number_of_nodes())  # 输出边的数量
                    #print("number of edges:", G.number_of_edges())  # 输出边的数量
                    #print(nx.degree(G, weight='weight'))
                    top_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
                    Gm = bipartite.project(G, top_nodes)
                    if (Gm.number_of_nodes() != 0):
                        largest_components = max(nx.connected_components(Gm), key=len)
                    else:
                        largest_components = []
                    print(len(largest_components))
                    #####################输   出##########################
                    with open(
                            "expected output/results/result seed=" + str(seed) + " p=" + str(p) + " b=" + str(b) + ".csv",
                            'a+', encoding='utf-8', newline='') as fw2:
                        writer = csv.writer(fw2)
                        writer.writerow(
                            [rows, 0.4086980748443965, 0.4086980748443965, 0.00017681820805288644, 977,
                             len(largest_components), G0.number_of_nodes(),
                             G.number_of_nodes(), G0.number_of_edges(), G.number_of_edges(), 977,
                             Gm.number_of_nodes(), 16680, Gm.number_of_edges()])
                    mkdir("expected output/results/seed=" + str(seed) + " p=" + str(p) + " b=" + str(b))
                    terminal = pd.DataFrame(nx.get_edge_attributes(G, 'weight'), index=['weight'])
                    terminal.T.to_csv(
                        "expected output/results/seed=" + str(seed) + " p=" + str(p) + " b=" + str(b) + "/b " + rows + ".csv")
                    terminal = pd.DataFrame(nx.get_edge_attributes(Gm, 'weight'), index=['weight'])
                    terminal.T.to_csv("expected output/results/seed=" + str(seed) + " p=" + str(p) + " b=" + str(b) + "/m " + rows + ".csv")

    endtime = datetime.datetime.now()
    print(starttime,endtime)

start_up()