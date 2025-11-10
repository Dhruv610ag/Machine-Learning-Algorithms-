"""Adjacency matrix representation of graph"""
# class Graph:
#     def _init_(self,n:int):
#         self.graph=[]
#         for i in range(n+1):
#             temp=[0 for _ in range(n+1)]
#             self.graph.append(temp)
    
#     def add_edge(self,u:int,v:int,w:int):
#         self.graph[u][v]=w
#         self.graph[v][u]=w
        
#     #edge representation
#     def add_edge_list(self,u:int,v:int):
#         self.graph.append((u,v))    

#     def print_graph(self):
#         print(self.graph)

# if _name_ == "_main_":
#     g=Graph(5)
#     g.add_edge(1,2,5)
#     g.add_edge(1,3,10)
#     g.add_edge(2,4,6)
#     g.add_edge(3,5,8)
#     g.print_graph()    

# if _name_ == "_main_":
#     g=Graph(5)
#     g.add_edge_list(1,2)
#     g.add_edge_list(1,3)
#     g.add_edge_list(2,4)
#     g.add_edge_list(3,5)
#     g.print_graph()

"""Adajacency list representation of graph"""
# class Graph:
#     def _init_(self,n:int):
#         self.graph=[]
#         for i in range(n):
#             self.graph.append([])
    
#     def add_edge(self,u:int,v:int,w:int):
#         self.graph[u].append((v,w))
#         self.graph[v].append((u,w))
    
#     def print_graph(self):
#         print(self.graph)

# if _name_ == "_main_":
#     g=Graph(5)
#     g.add_edge(0,1,5)
#     g.add_edge(0,2,10)
#     g.add_edge(1,3,6)
#     g.add_edge(2,4,8)
#     g.print_graph()

"""Graph Traversal Techniques"""
#Graph Traversal using DFS
# class Graph:
#     def __init__(self,n):
#         self.g=[]
#         self.visited=[]
#         for i in range(n+1):
#             self.g.append([])
#             self.visited.append(False)
        
#     def add_edge(self,u,v):
#         self.g[u].append(v)
#         self.g[v].append(u)
    
#     def dfs(self,s):
#         print(f"Visited:{s}")
#         for v in self.g[s]:
#             if not self.visited[v]:
#                 self.visited[v]=True
#                 self.dfs(v)
# g=Graph(5)
# g.add_edge(1,2) 
# g.add_edge(1,3)
# g.add_edge(2,4)
# g.add_edge(3,5)
# g.visited[1]=True
# g.dfs(1)

#Graph Traversal using BFS
# from collections import deque
# class Graph:
#     def __init__(self,n):
#         self.g=[]
#         self.visited=[]
#         for i in range(n+1):
#             self.g.append([])
#             self.visited.append(False)
        
#     def add_edge(self,u,v):
#         self.g[u].append(v)
#         self.g[v].append(u)
    
#     def bfs(self,s):
#         queue=deque()
#         queue.append(s)
#         self.visited[s]=True
#         while queue:
#             u=queue.popleft()
#             print(f"Visited:{u}")
#             for v in self.g[u]:
#                 if not self.visited[v]:
#                     queue.append(v)
#                     self.visited[v]=True
# g=Graph(5)
# g.add_edge(1,2) 
# g.add_edge(1,3)
# g.add_edge(2,4)
# g.add_edge(3,5)
# g.bfs(1)

