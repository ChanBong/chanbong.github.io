---
layout: post
title: 'Graph Theory'
date: '2022-02-22 15:04'
excerpt: >-
  Notes of MAN-204 till lecture 13
comments: true
tags: [general, acads, spring_2022, graph]
---

# MAN-204

## Notes
### Lecture 1-2
- What is a graph ?
- What problems can we solve ?

```DEFINATION
Multiple Edges
Loop
Simple Graph
Adjacent Vertices
Complement of a Graph
Clique
Independent set
Bipartite
```
- Clique : Set of pairwise adjacent vertices
- Independent Set : Set of pairwise nonadjacent vertices
- Clique Number : Size of the largest clique
- Independence Number :  Size of the largest independent set of G
- Bipartite : Two color colouring. Formally $V(G) = X_1 \cup X_2 \text{ and } X_1 \cap X_2 = \phi$  

### Lecture 3
```DEFINATION
Path
Cycle
Subgraph 
Connected Graph
Complete Graph
Complete Bipartite Graph 
Girth of a graph
```
- Notation 
	- $P_n$ : A path with n vertices
	- $C_n$ : A cycle with n edges
	- $K_n$ : Complete graph with n vertices
	- $K_{m, n}$  : Complete bipartite graph 
- Girth : Length of the smallest cycle present in the graph, infinite if the graph has no cycle
- Walk : A list of vertices and edges $v_0, e_1, v_1, e_2, v_2 ... v_{k-1}, e_k, v_k$ where $e_i$ has endpoints $v_i$ and $v_{i-1}$
- Trail : A walk where edges are not repeated
- Path : A walk where edges and vertices both are not repeated  
- The length of a walk, trail, path is the number of edges that are present in the walk/trail/path. Hense $P_n$ has length n-1 and $C_n$ has length n.

### Lecture 4-5
- Lemma : Every u,v walk contains a u,v path 
	- Proof: By principal of strong induction 
- A subgraph H is maximal connected gubgraphs when 
	- H is connected
	- H is not contained in any larger connected subgraph of G 
- Components are pairwise disjoint. No two share a vertex
- Deleting or adding a vertex decreases or increases the bumber of components by 0 or 1 repectively
- Proposition : Every graph with n vertices ans k edges has atleast n-k components
	- Proof : Since if it has 0 edges it has n components and adding an edge decreases it by at max 1 you will have atleast n-k components
- Cut-edge/vertex : If the deletion of some edge/vertex increases the number of components in the graph then it is called cut-edge/vertex
- Theorem : An edge is a cut-edge iff it belongs to no cycle 
	- SInce e only affects the component it is in, say H, it suffices to prove that H-e is connected iff e belongs to a cycle
	- So let e connect x and y. Since H-e is connected it contains a x, y path. This path along with e completes a cycle
	- Now say e lies in a cycle C. If path between any two vertices u, v doesn't pass through e. Then this path is in H-e, otherwise you can construct a u-x path from P, x-y path using e and y-v path along P. Hence this also belongs in H-e. H-e is connected
- Lemma : Every closed odd walk contains an odd cycle 
	- Proof : Strong induction on length
- A closed even walk may not contain a cycle
	- If an edge e appears exactly once in a closed even walk W, then W does contain a cycle through e.
- Theoram :  A graph is bipartite iff it has no odd cycle
	- If it is bipartite you can only have even cycles. 
	- If no odd cycles then you can use f(v) : Minimu length from u to v. 
	- X = {f(v) is odd } ; Y = {f(v) is even}. Then they partiition the set. And for two elements belonging to the same set if they are connected then you will have a n odd cycle

### Lecture 6-7
- Adjecency matrix A(G) is matrix where $a[i][j]$ is the number of edges i and j have in common
	- A(G) is symmetric
	- If graph is simple A(G) will only contain 0s and 1s, with 0s on the diagonal
	- Degree of v is the sum of the entries in the row for v in either A(G) or M(G)
- Incidence Matrix M(G) ia matrix where $m[i][j]$ is 1 is $v_i$ is an endpoint of $e_j$ 
- Isomorphism : From a simple graph G to a simple graph H is a bijection $f : V(G) \to V(H)$ such that $u,v \in E(G) \iff f(u)f(v) \in E(H)$ . We say that G is isomorphic to H. $G \cong H$  
- Relation of isomorpism is an equivalance relation. We can have an isomorphism class
- Isomorphism class of $G = \{G' : G \cong G'\}$ 
- Two graphs are isomorphic iff their complements are isomorphic
- With n vertices we can have $2^{n \choose 2}$ simple graph
	- These are only nonisomorphic graphs with 4 vertices
- A decomposition of a graph is a list of subgraph such that each edge appears in exactly one of the subgraphs in the list. Basically breaking a graph into subgraphs that are mutually exclusive and disjoint
- A graph is self-complementary if it is isomorphic to its complement
	- A n-vertex graph H is self completementry iff $K_n$ has a decomposition cosisting of two copies of H
- Let $G_1, G_2, ... G_k$ be graphs. Then their union G s.t $$V(G) = \bigcup_{i=1}^{k}V(G_i)$$ $$E(G) = \bigcup_{i=1}^{k}E(G_i)$$
- Difference between union and decompostion is that the edge can repeat itself

### Lecture 8
- Petersen Graph : Each vertex is a 2 subset of set of {1, 2, 3, 4, 5}. Each edge connects two disjoint 2 sets. 
	- Each vertex has degree 3 as there are 3 possibilities to pick disjoint subsets from a 2set
	- If two vertices are non-adjacent in the petersen graph then they have exactly one common neighbour
	- The girth is 5. There can't be a cycle of 3(By defination) and 4 (Only one common neighbour)
- Degree : Number of edges incient to v. Each loop counts twice.
	- $\delta(G)$  = minimum degree in G
	- $\Delta(G)$ = maximum degree in G
- If $\delta(G) = \Delta(G) \implies$ G is a regular graph 
- Neighbour of $v \in V(G)$ ; $N(v) = \{x \in V(G) : \text{x is incident to v}\}$  
- Theorem : If G is a graph, then $\sum_{v \in V(G)} d(v) = 2*e(G)$
	- Each edge contributes 2 to degree
- The average degree in G $$\delta(a) \leq \frac{\sum_{v \in V(u)} d(v)}{n(G)}=\frac{2 \cdot e(u)}{n(G)} \leq \Delta(G)$$
- No graph can have odd number of vertices with odd degree
- A k-regular graph with n-vertices has $(n*k)/2$ edges  

### Lecture 9
- K dimensional cube or hypercube $Q_k$ 
	- $\mid{V(Q_K)}\mid$  = $2^k$
	- Every verte has degree k. Grph is k-regular 
	- By degree sum formula $e(Q_k) = k*2^{k-1}$  
	- $Q_k$ is bipartite
		- X = {parity is even if it has even number of ones}
		- Y = {parity is odd if it has odd number of ones}
		- Clearly X and Y is a bipartition
- Proposition : A k-regular graph which is bipartite has the same number of verticesin each set
	- $e(G) = k * \mid X \mid = k* \mid Y \mid$ 
- Vertex-deleted subgraph : Graph obtaimed by deleting one vertex form the Graph G. denoted by G-v
- Proposition : For a simple graph G with vertices $v_1, v_2, ... v_n$ and $n \geq 3$ , $$e(G) = \frac{\sum e(G-v_i)}{n-2} \text{ and } d_G(v_j) = \frac{\sum e(G-v_i)}{n-2} - e(G-v_j)$$
	- If e connects u and v then in the sum it is counted n-2 times (not counted only in G-u and G-v)
	- For the second one use first and $e(G) - e(G-v_j) = d_G(v_j)$
- Extremal Problems
- Prob 1 : Minimum number of edges in a connected graph with n vertices is n-1
	- Contradiction : If it has n-2 edges, then since graph with n vertices and k edges has $\geq$ n-k components i.e $\geq$ 2. And hence it is disconnected
- A path $P_n$ has exactly n-1 edges

### Lecture 10
- Proposition : If G is a simple n-vertex graph with $\delta(G) \geq \frac{n-1}{2}$ then G is connected
	- Proof : Let $u, v \in V(G)$. If u and v are connected then done. Else it is enough to prove that they have a common neighbour. 
	- $\mid N(u)\mid  = d(u) \geq \frac{n-1}{2}$ , $\mid N(u)\mid  = d(u) \geq \frac{n-1}{2}$
	- $\mid N(u) \cup N(v)\mid   \leq n-2$
	- $\mid N(u) \cap N(v)\mid   = 1$
	- This inequality is sharp
- Theorem : Every loopless graph G has a bipartite subgraph with atleast $\frac{e(G)}{2}$ edges
	- Let X, Y be any partition of the vertex set V(G). Consider subraph H with V(H) = V(G) and only those edges of G which have one endpoint in X and other in Y. So H is bipartite. If $e(H) \geq \frac{(G)}{2}$ then we are done.  Else, let u be any vertex in H. If $d_H(u) < \frac{d_G(u)}{2}$ , it means that it has more adjacent vertices on one side than the other. If we transfer all those edges to the other side then we get $d_H(u) \geq \frac{d_G(u)}{2}$ .
	- Since this process increases the number of edges in H, it has to stop sometime. When it stops, $d_H(u) \geq \frac{d_G(u)}{2} \forall u \in V(G)$  . Sum both sides $2e(H) \geq e(G)$ 
- The degree sequence of a graph is the list of vertex degreees, usually written in decreasing order as $d_1 \geq d_2 \geq ... \geq d_n$ 
- Proposition : The non-negative integers $d_1, d_2, ... d_n$ are the vertex degrees of some graph iff $\sum d_i$ is even
	- If $d_1, d_2, ... d_n$ are the degrees of a graph G, then by degree sum this is even
	- If $\sum d_i$ is even. Number of odd $d_i$'s must be even. TODO

### Lecture 11
- Graphic Sequence : List of non-negative integers that is the degree sequence of some sinple graph 
- Theorem : Havel and Hakimi : A list d is a degree sequence of a simple graph iff d' is the degree sequence of a simple graph, where d' is constructed by sorting the list in decreasing fashion and then subtracting lasrgest number times 1 form the elements from begining. If in the end all that remain is 0's then it is possible to realise this graph 
- Proof : TODO

### Lecture 12
- 
