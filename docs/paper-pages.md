# Paper by page — Mei & Zhao 2024

Toàn văn `docs/paper.pdf` được trích xuất theo từng trang, dùng để **tra cứu nhanh khi vấn đáp**.

- **Citation:** Mei, P. & Zhao, Y. "Dynamic network link prediction with node representation learning from graph convolutional networks." *Scientific Reports* 14:538 (2024).
- **DOI:** 10.1038/s41598-023-50977-6
- **Tổng số trang:** 12
- **Index theo chủ đề:** xem `docs/paper-notes.md` và memory `reference_original_paper`

---

## Trang 1 — Title, Abstract, Introduction (mở đầu)

```
1
Vol.:(0123456789)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports
Dynamic network link prediction 
with node representation learning 
from graph convolutional networks
Peng Mei  & Yu hong Zhao *
Dynamic network link prediction is extensively applicable in various scenarios, and it has progressively 
emerged as a focal point in data mining research. The comprehensive and accurate extraction of node 
information, as well as a deeper understanding of the temporal evolution pattern, are particularly 
crucial in the investigation of link prediction in dynamic networks. To address this issue, this paper 
introduces a node representation learning framework based on Graph Convolutional Networks (GCN), 
referred to as GCN_MA. This framework effectively combines GCN, Recurrent Neural Networks (RNN), 
and multi-head attention to achieve comprehensive and accurate representations of node embedding 
vectors. It aggregates network structural features and node features through GCN and incorporates 
an RNN with multi-head attention mechanisms to capture the temporal evolution patterns of dynamic 
networks from both global and local perspectives. Additionally, a node representation algorithm 
based on the node aggregation effect (NRNAE) is proposed, which synthesizes information including 
node aggregation and temporal evolution to comprehensively represent the structural characteristics 
of the network. The effectiveness of the proposed method for link prediction is validated through 
experiments conducted on six distinct datasets. The experimental outcomes demonstrate that the 
proposed approach yields satisfactory results in comparison to state-of-the-art baseline methods.
The objective of link prediction for dynamic networks is to evaluate the probability of future connections between 
nodes. Owing to the rapid advancement of communication networks, the Internet, and the big data era, dynamic 
network analysis has emerged as a crucial research problem, attracting the attention of experts from various fields 
towards dynamic network link prediction. In biology, dynamic network link prediction has many potential appli-
cations in protein network modeling, including: Protein interaction  prediction1, evolution of metabolic  network2, 
and conversion mechanism of signal transduction between proteins provide conditions for guiding the control of 
 diseases3 and signal transduction  complexes4, as well as revealing the interaction relationship between molecules, 
which provides potential value for drug  design5, disease understanding and gene  regulation6. These applications 
demonstrate the important role of dynamic network link prediction in protein network modeling, which helps to 
understand the structure and function of protein interaction networks from a dynamic perspective, and provides 
new insights and methods for biological research. In the social  domain7, we focus on the network evolution of 
social users, which is of great significance for social media marketing, information  dissemination8 research, and 
social dynamic analysis. Moreover, dynamic network link prediction is not limited to social media and can be 
applied to various fields such as finance, transportation networks, and environmental science. It aids in better 
understanding crucial changes, event evolution, and node behavior within the network, thereby offering support 
for diverse application scenarios.
The existing works on dynamic network link prediction face two primary challenges: First, employing a single 
neighbor information to represent nodes overlooks the influence of node clustering, neighbor relationship, and 
time evolution in the network. Second, when constructing temporal attribute models, the spotlight is often nar-
rowed down to the evolution pattern of the global time step, neglecting the impact of short-term connections 
and feature changes between nodes and their neighbors in a single time step, which can affect the accuracy of 
node embedding vectors. To address the aforementioned issues, this study presents a GCN_MA framework. In 
this paper, a NRNAE algorithm is proposed to enrich the node information representation by using node degree, 
clustering coefficient and neighbor relationship, so that GCN can aggregate the above multi-dimensional features 
to learn node embedding vectors more comprehensively. RNN and multi-head attention are used to model the 
time attributes from the global and local perspectives respectively, so as to complete the link prediction task more 
accurately and comprehensively. In terms of global time attribute modeling, the whole time step is considered, 
OPEN
School of Information Engineering, Inner Mongolia University of Science and Technology, Baotou  014010, China. 
*email: zhaoyuhong35@163.com
```

---

## Trang 2 — Introduction (tiếp), đóng góp, Related work, Notations

```
2
Vol:.(1234567890)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
and the parameters W of GCN are continuously updated through Long Short-T erm Memory (LSTM) network, 
which can dynamically adjust the aggregation of multi-dimensional features of GCN in each time step to learn 
the global time evolution information. Local temporal attribute modeling focuses on a specific time step, cap -
tures the changes of links and features around the node and its neighbors through multi-head attention, and 
makes adaptive adjustments to the node embedding vector, so as to obtain high-quality node embedding vector 
representation. The contributions of this paper are as follows:
1) This study presents a GCN-based node representation learning framework that captures temporal attributes 
by examining global and local information fluctuations, thereby offering a more comprehensive depiction 
of the temporal evolution pattern of dynamic networks.
2) A novel NRNAE algorithm is proposed to enrich the structural features of the network and ensure the 
effectiveness of the aggregated information learned by GCN at each node.
3) Comprehensive experiments were conducted to validate the efficacy of the GCN_MA framework in dynamic 
network link prediction and to compare it to several advanced baseline methods.
Related research
In recent years, a multitude of dynamic network link prediction techniques have been proposed by researchers. 
Among these, the method grounded in similarity measurement is deemed to be the most straightforward and 
potent. This approach posits that the higher the similarity between nodes, the greater the likelihood of their 
forming  connections9. In recent years, a multitude of dynamic network link prediction techniques have been 
proposed by researchers. Among these, the method grounded in similarity assessment is deemed as the most 
straightforward and potent. In this approach, the higher the degree of similarity between nodes, the greater the 
likelihood of establishing  links9. Chen et al.10 introduced NCC and NCCP , two similarity measures founded on 
the clustering coefficient of neighboring nodes. Wu et al.11 advanced a technique capable of dynamically predict-
ing the similarity of future node pairs, and calculated the similarity measure for nodes via an algorithm based 
on node ranking. Zhang et al.12 proposed a method integrating node centrality with time series to appraise the 
impact of common neighbors in dynamic networks, and to capture the evolving pattern of node centrality over 
time. The proficiency of graph convolutional networks in learning intricate systems has prompted some research-
ers to develop representation learning techniques based on this approach. Cui et al.13 introduced DyGCN, a vari-
ant of GCN that caters to dynamic networks by updating node embeddings to propagate embedding information. 
Chami et al.14 merged GCN with hyperbolic geometry, designing hyperbolic graph convolutional neural networks 
(HGCN) to acquire inductive node representations for hierarchical and scale-free graphs.
With the ongoing advancements in deep learning technology research, an increasing number of studies have 
emerged that focus on applying deep learning techniques for dynamic network link prediction. Xian et al. 15 
proposed a link prediction model, GraphLP , based on network reconstruction theory, which leverages the fea-
ture learning capability of deep learning models to automatically extract structural patterns from graphs for 
improved link prediction. In an effort to capture the evolution patterns of time, Zhou et al. 16 employed the 
concept of triadic closure as a guiding principle to capture the evolution patterns of various snapshots. Goyal 
et al.17 introduced DynGEM, a model based on deep autoencoder, which progressively updates node embeddings 
from the initialization of the previous step. However, these methods struggle with capturing long-term dynam-
ics, resulting in limitations to their accuracy. To address this issue, Chen et al. 18 proposed a novel end-to-end 
model, GC-LSTM, which effectively combines LSTM and GCN to capture the local structural attributes of nodes 
and their interrelationships for enhanced link prediction in dynamic networks. The above-mentioned methods 
primarily rely on global time information for modeling temporal attributes but do not take into account the 
more intricate changes in local information of nodes within dynamic networks. Therefore, this study needs to 
consider modeling temporal attributes from the perspective of local information alterations to comprehensively 
capture temporal evolution patterns.
Definitions and methods
Definition of a dynamic network
A dynamic network can be represented as a sequence of discrete snapshots, denoted as G ={ G1, G2, ..., GT } , 
where Gt = (V , Et , At ) (t ∈[ 1, T]) represents the t-th time network snapshot. Let V be the set of all nodes and 
Et denote the set of edges within a fixed time interval [t − τ , t] . At denotes the adjacency matrix of Gt , where 
At (i, j) = 1 if there is a link between nodes i and j , and At (i, j) = 0 otherwise.
Definition of link prediction in dynamic networks
The link prediction in dynamic  networks19 aims to forecast the adjacency matrix ˆAT+1 corresponding to the next 
time step snapshot GT+1 at time step T + 1 , given the prior T historical network snapshots G ={ G1, G2, ..., GT } 
along with their respective adjacency matrices {A1, A2, ..., AT } , as illustrated in Eq. (1):
The model to be constructed is represented by f (·) , while ˆAT+1 ∈ RN×N  denotes the predicted value. The 
topology of a dynamic network evolves over time, and its progression can be exemplified by the alterations in 
the adjacency matrix across different time intervals, as illustrated in Fig.  1. The objective of dynamic network 
link prediction is to identify the links that are most likely to emerge or vanish in the subsequent time window. 
In the AT context, yellow denotes the newly established connection, while red signifies the disintegrated link.
(1)ˆAT+1 = f (A1, A2, ... , AT )
```

---

## Trang 3 — Framework overview (Fig. 2), NRNAE algorithm — phần đầu

```
3
Vol.:(0123456789)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
Node representation learning framework based on graph convolutional network
In this study, a node representation learning architecture grounded on GCN, coined GCN_MA, is introduced. 
The design of this architecture seeks to integrate the multi-dimensional features of node degree, clustering coef-
ficient, and time evolution to comprehensively and accurately represent node information, while simultaneously 
modeling the time attribute from both global and local perspectives. This enables a more comprehensive learn-
ing of the time evolution pattern of dynamic networks, ultimately achieving the objective of link prediction. 
In the following, the main components of the proposed framework are detailed, including node information 
representation, learning of node representation vectors, and temporal property modeling, as illustrated in Fig. 2.
Firstly, the NRNAE algorithm was initially employed to mine network information, thereby enriching 
the network structure features {˜A1, ˜A2, ..., ˜AT } . The node degree matrix was utilized as the node features 
{X1, X2, ..., XT } to describe node information from two perspectives. The learning of node representation vectors 
involves aggregating multi-dimensional features via GCN to acquire each node embedding vector representation 
{H1
T , H2
T , ..., HN
T } , where N signifies the number of nodes. In this study, an improved LSTM was incorporated to 
continuously update the parameter W of GCN. This enables GCN to dynamically adjust the aggregation of multi-
dimensional features across different time steps, effectively modeling the time attribute of dynamic graphs from 
a global perspective. To further explore the dynamic characteristics of dynamic networks, a multi-head attention 
mechanism was introduced. This mechanism adaptively assigns reasonable weights to each node embedding 
vector, enabling the learning of local structure information for each time snapshot from a local perspective. 
Ultimately, it generates a new node embedding vector representation ZT ={ Z1
T , Z2
T , ...Z N
T } . The synergy of 
these two mechanisms allows the framework to capture the nuances of time evolution with higher accuracy. To 
complete the link prediction task for dynamic networks, the probability value of chain edges between nodes is 
calculated based on a Multi-Layer Perceptron (MLP).
Node information representation
The evolution of dynamic networks over time involves assessing not only the impact of individual nodes on the 
network’s overall structure, but also the influence on neighboring nodes. This includes the tendency of neigh -
boring nodes to cluster and the level of interaction between neighbors and other nodes. Consequently, this 
study defines the concept of "node aggregation effect," which quantifies the influence of the clustering tendency 
between a node’s attributes and its neighbors on the connection strength between the node and its neighboring 
nodes at different time steps. The node clustering effect encompasses multidimensional features such as degree 
and clustering coefficient. To harness the network information for node representation and enable GCN to 
1
2
4
5 6
3
1  2  3  4  5  6
1  2  3  4  5  6
1
2
4
5 6
3
1  2  3  4  5  6
1  2  3  4  5  6
GT-1
AT-1 GT
AT
Figure 1.  Dynamic network evolution and an illustration of its adjacency matrix. The network time evolves 
from time T − 1 to time T when E (2,6) and E (1,5) disappear and E (1,3) and E (4,6) appear and are filled with 
red and yellow colors in the corresponding adjacency matrix.
Ã T
XT
Multi- Head 
Attention
Z1
T
Z2
T
ZN
T
.  .  .
GT+1
GT
GCN
H1
T
H2
T
HN
T
.  .  .
LSTM
WT-1 WT
MLP
Figure 2.  Illustration of the overall framework of dynamic network link prediction based on GCN_MA. The 
NRNAE algorithm is used to represent the node information, GCN is used to aggregate multi-dimensional 
features to learn the embedding vector representation of the node, and LSTM and Multi-Head Attention are 
used to capture the global and local time evolution patterns respectively. Finally, MLP is used to calculate the 
probability value of the edge to complete the link prediction task.
```

---

## Trang 4 — NRNAE (tiếp): S(i,j), β; Global time attribute modeling với LSTM

```
4
Vol:.(1234567890)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
learn more accurate node embedding vectors by aggregating multi-dimensional features, in this paper, a Node 
Representation Algorithm Based on Node Aggregation Effect (NRNAE) is proposed.
CC(i) (clustering coefficient, CC) stands for clustering  coefficient20, which is used to describe the degree of 
clustering between a vertex i and other nodes in a graph; specifically, it captures the degree to which the neighbors 
of a target node are interconnected, within the range of CC(i) ∈[ 0, 1][0, 1] , as illustrated in Eq. (2):
where Ri denotes the number of triangles formed by node i and its first-order neighbor nodes, and Ki denotes 
the number of first-order neighbor nodes of i . In complex networks, the degree of a node quantifies the node’ s 
significance, which can be comprehended as the cumulative influence of edges surrounding the node. Addition-
ally, the clustering coefficient is employed to gauge the extent of clustering among nodes in the network. In an 
endeavor to scrutinize the traits of nodes more exhaustively, this study defines a novel concept, referred to as " 
Aggregation Strength ".
Aggregation Strength (AS)AS(i) is used to describe the probability of focusing on a node to form a cluster, 
and reflects the importance and influence of the node. The specific definition is illustrated in Eq. (3):
where deg ree(i) denotes the degree of node i and CC(i) denotes the clustering coefficient of node i.
The Node Aggregation Effect is utilized to describe the extent to which the attributes of a node and the pro-
pensity of neighboring nodes to cluster influence the connection strength between the node and its neighbors 
throughout the dynamic evolution process. The aggregation effect of nodes i and j is determined based on the 
Aggregation Strength. The specific definition is illustrated in Eq. (4).
where, N(i) denotes the set of first-order neighbor nodes of node i , and j ∈ N(i) . N(j) represents the set of first-
order neighbor nodes of node j . S(i, j) is normalized by S(i, j) ∈[ 0, 1] . ST ∈ RN×N  represents the matrix of node 
aggregation effects ( t ∈[ 1, T] ) constructed by the ST (i, j) . Furthermore, to examine the contribution of ST to 
the aggregated feature information of GCN, a weighting factor β ∈[ 0, 1] is incorporated. In this study, a node 
representation algorithm based on node aggregation effect is employed to create a new adjacency matrix ˜AT , 
which comprehensively represents node information. This enables GCN to aggregate multi-dimensional feature 
information and enhance the accuracy of the learned node embedding vector, as illustrated in Eq. ( 5):
where, AT ∈ RN×N  represents the adjacency matrix of snapshots at time step T.
Nodes represent the learning of vectors
Graph convolutional networks have powerful learning capabilities for complex systems and can effectively deal 
with graph data structures with non-Euclidean features. The core idea of Graph Convolutional Networks (GCN)21 
is to extend deep learning methods to graph-structured data in order to effectively learn and represent the 
relationships between nodes while preserving the topological structure of nodes on the graph. Compared with 
the graph convolutional network based on spatial domain, the graph convolutional network based on spectral 
domain is more suitable for dealing with various complex graph data, and can consider the global structure 
information of the whole graph. In this study, we utilize the graph convolutional network based on the spectral 
domain to calculate the embedding vector representation HT ={ H1
T , H2
T , ..., HN
T } for each node at a specific 
time step T, as illustrated in Eq. (6).
where,AT ∈ RN×N  and XT ∈ RN×f  represent the adjacency matrix and node characteristic matrix of GT , respec-
tively, and ˆDT ∈ RN×N  is the diagonal matrix generated by ˆDi,j = ∑
j ˆSi,j . ˆST = AT + βST + IN , with IN ∈ RN×N  
as the identity matrix. σ symbolizes the ReLU function that introduces nonlinearity, and WT ∈ Rf ×D is the weight 
parameter that can be learned. This paper proposes a new aggregation strategy for AT + βST + IN that is dif-
ferent from the traditional GCN method that uses AT + IN to guide node feature aggregation. In this paper, the 
node aggregation effect ST is added to further guide GCN to aggregate network structure information. Under 
the new aggregation strategy, the target node can identify the degree of influence of different neighbor nodes on 
its importance, and use this importance to guide GCN aggregation.
Global time attribute modeling based on LSTM
In this study, we establish a node representation matrix HT for a specific time step. Due to the independent learn-
ing of HT for each specific time step T , these matrices are mutually exclusive. These low-dimensional representa-
tion matrices solely capture the local network structure information at a specific time step, lacking the ability 
to encapsulate global structural information and dynamic network time evolution patterns. T o address these 
issues, we require an effective strategy to simulate the temporal properties across different time steps, enabling 
the node representation matrix HT to learn the global structural information of the graph from all time steps 
and resulting in high-quality node embedding vector representations.
(2)CC(i) = 2Ri
Ki(Ki − 1)
(3)AS(i) = deiree(i) ∗CC(i)
(4)S(i, j) =| N(i) ∩ N(j)|∗AS (i)
(5)˜AT = AT + βST
(6)HT = GCN(AT , ST , XT |WT ) = σ
(
ˆD
− 1
2
T ˆST ˆD
− 1
2
T XT WT
)
```

---

## Trang 5 — LSTM weight evolution chi tiết; bắt đầu Local time attribute (attention)

```
5
Vol.:(0123456789)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
The  LSTM22 architecture exhibits flexible nonlinear transformation capabilities while processing time series 
data, enabling it to uncover the temporal evolution patterns of dynamic networks. In an effort to enhance the 
model’s temporal expression capabilities, an improved LSTM will be employed to capture the time attribute 
of dynamic graphs. In this paper, all time steps are considered for the global time attribute modeling, and the 
parameters of GCN are continuously updated through LSTM, and the learning of these parameters is also passed 
to GCN after timely training and learning in LSTM, so that the aggregation of multi-dimensional features of 
GCN can be dynamically adjusted in each time step to learn the global time evolution information. As shown 
in Fig. 3, in this study, the weight matrix WT−1 generated by the GCN at the previous time step is utilized as the 
input to the LSTM to produce the weight matrix WT at the subsequent time step, as illustrated in Eq. (7):
Given ˜AT ∈ RN×N  , XT ∈ RN×f  , and WT−1 ∈ Rf ×D , the LSTM is extended to the matrix version by construct-
ing a node embedding vector matrix HT according to Eqs. (3) and (7). The update process, realized through Eqs. 
(8)–(13), is elucidated as follows:
where,Uψ ∈ Rf ×f  and Mψ ∈ Rf ×f  are weight matrices. bψ ∈ Rf ×D is a bias vector, ψ ∈{ F, I, O, C} . As shown 
in Fig. 3, FT ∈ Rf ×D represents the computation of the forget gate in the LSTM, which determines how much 
information needs to be retained in the cell state at the previous time step. IT ∈ Rf ×D is the computation of the 
input gate that determines which new temporal information will be added to the cell state. OT ∈ Rf ×D is the com-
putation of the output gate, which determines which time information can be output. ˜CT ∈ Rf ×D is the updat-
ing process of the cell state. In the context of this paper’s framework, the GCN weight parameter W  are solely 
updated by the enhanced LSTM, negating the need for further training and learning. Consequently, the number 
of parameters in the GCN does not escalate with the time step, thereby reducing both time and space complexity. 
Moreover, the framework continuously updates the GCN weight parameter W through the LSTM in each time 
step. This dynamic adjustment of the weight parameter W enables the GCN to better adapt to the time evolu -
tion and enhances the learning of node embedding vectors in graph data. Subsequent ablation experiments also 
confirmed that updating the GCN parameters through LSTM indeed improves the accuracy of link prediction.
(7)WT = LSTM(WT−1)
(8)FT = σ( MF WT−1 + UF WT−1 + bF )
(9)IT = σ( MI WT−1 + UI WT−1 + bI )
(10)OT = σ( MOWT−1 + UOWT−1 + bO)
(11)˜CT = tanh(MC WT−1 + UC WT−1 + bC )
(12)CT = FT ◦CT−1 + IT ◦˜CT
(13)WT = OT ◦tanh(CT )
Figure 3.  Update process of parameters in GCN by LSTM. The LSTM updates the weight parameter WT of the 
GCN at time step T based on the previous T − 1 time steps.
```

---

## Trang 6 — Multi-head self-attention; Link prediction & loss function

```
6
Vol:.(1234567890)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
Local time attribute modeling based on multi-head attention
In the field of dynamic network link prediction research, the alterations in local information within dynamic 
graphs are often overlooked, such as the rapid shifts in link status and node characteristics of a node and its 
neighboring nodes. Nevertheless, patterns that capture the evolution of local information can provide a more 
refined depiction of the dynamics within dynamic networks. The Multi-Head  attention21 model assumes a crucial 
role in this context, as it can effectively capture the interrelationships between nodes and neighbors, making it 
particularly suitable for modeling local information. By utilizing the multi-head attention model, it becomes 
feasible to focus on distinct neighbor nodes in each head, thereby facilitating a more comprehensive understand-
ing of the local environment of the node and further assisting in the capture of the temporal evolution pattern 
of the dynamic network.
In this study, the node embedding vector HT at time step T is employed as the input for the multi-head atten-
tion model in this study, where HT ={ H1
T , H2
T , ..., HN
T } . The scaled dot product attention, as adopted from  Ref23, 
is utilized to accelerate computation. For each time step, the temporal attention layer accepts HT as input and 
generates a novel sequence of node embedding vectors, denoted as ZT ={ Z1
T , Z2
T , ...Z N
T } . In this research, the 
multi-head attention model is utilized to learn features from diverse latent spaces, thereby enhancing the model’s 
representation  capacity24. Within each attention head, a linear transformation is applied to the input vector to 
produce QT , KT , and VT , as illustrated in Eqs. (14–16):
The attention weights are computed using QT and KT and are subsequently applied to VT to obtain the output 
Attention(QT , KT , VT ) , as illustrated in Eq. (17):
where QT , KT , and VT are three matrices, and QT ∈ RDK ×N  , KT ∈ RDK ×N  , VT ∈ RDV ×N .√dK  are scaled.
Figure 4 demonstrates the implementation of the multi-head attention mechanism, which involves process-
ing the original input sequence through multiple sets of self-attention operations. Subsequently, the results 
of each self-attention set are concatenated and undergo a linear transformation to produce the final output 
ZT ={ Z1
T , Z2
T , ... ZN
T } . The specific computation is delineated in Eqs. (18–19):
where ZT ∈ RN×D , WQ
i ∈ RDK ×N  , WK
i ∈ RDK ×N  , WV
i ∈ RDV ×N  and Wo ∈ RhDV ×N  are parameter matrices.
Link prediction and loss function
The link prediction problem is reformulated as a binary classification task in this study, with the Multilayer 
Perceptron (MLP)25 serving as the classifier. The embedding vector ZT ={ Z1
T , Z2
T , ... ZN
T } . of nodes, which is 
learned through the effective integration of GCN, LSTM, and multi-head attention models, is employed as the 
input to the MLP . By learning the nonlinear relationship between data through the hidden layer, a probability 
matrix PT is ultimately generated via the output layer, as illustrated in Eq. (20).
(14)QT = WqHT ∈ RDK ×N
(15)KT = WkHT ∈ RDK ×N
(16)VT = Wv HT ∈ RDV ×N
(17)Attention(QT , KT , VT ) = soft max( QT KT
T√dk
)VT
(18)headi = Attention(QWQ
i , KWK
i , VWV
i )
(19)ZT = Concat(head1, head2, ..., headh)Wo
 rotcev gniddebme edoN
                                                                                                                                                                                                                                                                                                                                                                                       
Wq
Wk
Wv
Q
KT
V
Attention factor
Attention Node 
embedding vector 
Multi-head 
Attention Node 
embedding vector 
Concat
Wo
Input
Output
Figure 4.  Data flow of head attention model. The input is the node embedding vector HT at time T , and the 
input is mapped to QT , KT , VT by linear variation. For each head, the attention weight coefficient is calculated, 
and it is applied to the corresponding VT to obtain the attention output headi of each head. Then, the output of 
multiple heads is concatenated, and the new node embedding vector ZT is finally output.
```

---

## Trang 7 — **Dataset descriptions, Table 1, Baseline methods, Evaluation metrics**

```
7
Vol.:(0123456789)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
MLP(·) represents a fully connected network endowed with parameters, which include an input layer, a 
hidden layer, and an output layer. ZT ∈ RN×D serves as the input to MLP(·) , while ReLU acts as the activation 
function in the hidden layer and σ as the activation function for the softmax () in the output layer. PT ∈ RN×N  
constitutes a probability matrix built from PT (i, j) . A higher value of PT (i, j) corresponds to a greater likelihood 
of an edge existing between nodes i and j. Conversely, PT (i, j) = 0 signifies the absence of edges between nodes.
The binary cross-entropy loss  function26, which evaluates the accuracy of model predictions and facilitates 
online parameter updates without the need for retraining the entire model, as illustrated in Eq. (21):
where N is the total number of samples, PT is the probability matrix, and YT ∈{ 0, 1} represents the labels of the 
target links indicating whether the links exist.
Results and analysis
Dataset
To assess the efficacy of the proposed framework, experiments are performed on six real-world datasets. The 
comprehensive statistics of the six datasets are delineated in Table 1.
CollegeMsg27: It represents message communication among users of the online community of students at the 
University of California, Irvine.  Mooc_action28: The MOOC user actions dataset represents actions performed 
by users on popular MOOC platforms.  Bitcoinotc29: The Who Trusts Who network of people who transact with 
Bitcoin on a platform called Bitcoin OTC. Email-eu-core-temporal (EUT) 30: This network is the institution 
generated using Email data from a large European study.  LastFM31: This dataset is based on the concept of the 
original Last.fm dataset and is based on the million-song dataset.  Wikipedia32: This public dataset is edited for 
a month by editors on Wikipedia pages. We have selected a variety of datasets from distinct domains, such as 
social interactions, communication patterns, financial transactions, and Wikipedia, among others. These datasets 
encompass both sparse and dense networks, which aids in evaluating the effectiveness of the proposed method 
more thoroughly, given the diversity of the data.
Baseline methods
DGCN33 introduced a novel dice similarity to overcome the problem of unclear directional neighbor influence, 
further guiding aggregation, and updating GCN’s weight parameters with Long Short-Term Memory (LSTM) 
to capture global structural information for all time steps in dynamic graphs.
HTGN34 maps temporal graphs into hyperbolic space and combines hyperbolic graph neural networks and 
hyperbolic gated recurrent neural networks to capture evolving behaviors while implicitly preserving hierarchical 
information. It successfully models temporal networks by employing the Hyperbolic Temporal Self-Attention 
(HTA) module to focus on historical states and the Hyperbolic Temporal Consistency (HTC) module to ensure 
stability and generalization.
DyGNN35 models dynamic information as the evolution of a graph. Specifically, the proposed framework 
continuously updates node information by consistently capturing the sequence information of edges (interac -
tions), as well as the time intervals between edges and information propagation.
EvolveGCN36 adjusts the GCN model along the time dimension without requiring assistance from node 
embeddings. The proposed method captures the dynamics of graph sequences by evolving GCN parameters using 
an RNN. In our comparison, we employ the improved Long Short-Term Memory (LSTM) network within RNN.
Evaluation index
The accuracy of the algorithm was measured by the link prediction metrics AUC (Area Under Curve) and AP 
(Average Precision). They have different emphases on measuring prediction accuracy: AUC measures the overall 
accuracy of an algorithm; AP only considers whether the prediction is accurate for the top-L ranked edges. AUC 
can be interpreted as comparing a randomly selected edge from the test set with a randomly selected edge that 
is not present, and adding 1 point if the score value of the edge in the test set is greater than the score value of 
the edge that is not present. If the scores are equal, add 0.5 points. For n independent comparisons, if there are 
(20)PT = σ( MLP(ZT ))
(21)Loss =− 1
N2
N2
∑
i=1
(YT log(PT )) + (1 − YT ) log(1 − PT )
Table 1.  Statistics of the dataset.
Data type Number of nodes Number of edges Time steps
CollegeMsg 1899 59,835 47
Mooc_actions 7047 411,749 72
Bitcoinotc 6005 35,592 62
EUT 1005 332,334 127
LastFM 1000 1,293,103 76
Wikipedia 5684 87,931 42
```

---

## Trang 8 — AUC/AP equations; **Table 2 — kết quả chính**; Analysis of results; Parametric analysis (β)

```
8
Vol:.(1234567890)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
n′ times when the score value of the edge in the test set is greater than the score of the edge that is not present, 
and n′′ times when the score value is equal, then the AUC is defined in Eq. (22):
Precision is defined as the proportion of accurate predictions among the top L predicted edges. If m predic-
tions are accurate, that is, m of the top L edges are in the test set, then Precision is defined in Eq. (23):
Analysis of experimental results
In this paper, we evaluate the results of the GCN_MA framework across different metrics and datasets. In addi-
tion, this paper investigates the impact of changing the parameter β in the NRNAE algorithm on the AUC and 
AP of the framework in link prediction, and validates the effectiveness of the proposed NRNAE algorithm for 
helping graph convolutional networks to aggregate node embedding vector representations. At the same time, 
this paper uses ablation experiments to verify the effectiveness of each component for the framework.
The experimental results presented in Table 2 demonstrate that GCN_MA yields optimal AUC and AP values 
across all six networks when compared to other baseline methods. Under the first four datasets, the AUC of 
GCN_MA surpasses 90%, reaching a remarkable 98.880% in the Mooc-action dataset. Although DGCN exhibits 
superior performance across six datasets, it falls short of the proposed method due to its inability to account for 
the influence of connectivity and time evolution between neighbors on node importance in dynamic networks. 
HTGN employs two hyperbolic spaces to capture historical state and time consistency. However, it overlooks 
the local and global characteristics of node features in dynamic networks, resulting in an AUC and AP of only 
71.51% and 70.88% on the LastFM dataset. DyGCN utilizes the continuous time property to capture edge order 
information and update node information based on the time interval between information propagation. However, 
it only considers local time evolution, failing to capture global time evolution like LSTM, resulting in subpar 
performance on the ETU, Wikipedia, and LastFM datasets. The AUC of GCN_MA on other datasets improves 
by 5.38%, 2.93%, and 3.51% respectively. EvolveGCN enhances the GCN model by combining GRU and LSTM 
RNNs to learn time feature information of dynamic networks, yet it does not integrate node feature information 
with time feature information for node embedding vector representation. Consequently, the AUC and AP on 
the Wikipedia dataset stand at 62.89% and 61.63%, respectively. In comparison, the proposed framework in this 
paper achieves a notable improvement of 24.53% and 24.12% in AUC and AP , respectively.
The results show that combining the long short-term memory network with multi-head attention mechanism 
to capture the time evolution pattern of the dynamic network can indeed obtain the structural information of 
the network more detailed and more comprehensive, so as to better promote the link prediction of the dynamic 
network.
Parametric analysis
To validate whether the proposed NRNAE algorithm can enhance the comprehensive fusion of node informa -
tion by Graph Convolutional Networks (GCN), and thereby effectively improve the accuracy and precision of 
link prediction in dynamic networks, a series of experiments are conducted. In these experiments, 80% of the 
dataset is employed as the training set, while the parameter β  in the algorithm is incrementally adjusted from 
0.0 to 1.0 to assess its impact on the results. As illustrated in Fig. 5, the AUC and AP values of the Mooc-action 
dataset exhibit a significantly superior performance compared to those of the other datasets. This exceptional 
performance can be attributed to the higher dataset density, simpler data structure, and inclusion of more time 
segments, which facilitate a more comprehensive utilization of network information and thereby enhance the 
accuracy of the learned node embedding vectors. Consequently, the results demonstrate that the AUC and AP 
values are less sensitive to the parameter β.
(22)AUC = n′+ 0.5n′′
n
(23)Pecision = m
L
Table 2.  Comparison of link prediction AUC and AP of each method in different datasets. Best performing 
values are in bold.
Metric Methods Mooc-action CollegeMsg ETU Bitcoinotc LastFM Wikipedia
AUC 
DyGNN 0.9242 0.8856 0.7527 0.8769 0.8034 0.8371
EvolveGCN 0.7794 0.7867 0.8494 0.7811 0.8593 0.6289
HTGN 0.9712 0.8491 0.8694 0.8814 0.7151 0.8414
DGCN 0.9720 0.8799 0.8947 0.9046 0.8201 0.8472
GCN_MA 0.9880 0.9149 0.9222 0.9120 0.8757 0.8742
AP
DyGNN 0.9179 0.8839 0.7519 0.8617 0.8067 0.8371
EvolveGCN 0.7602 0.7620 0.8453 0.7838 0.8566 0.6163
HTGN 0.9773 0.8813 0.8642 0.8752 0.7088 0.8624
DGCN 0.9657 0.8813 0.8788 0.8884 0.7781 0.8379
GCN_MA 0.9863 0.8926 0.9082 0.8943 0.8704 0.8575
```

---

## Trang 9 — Parametric analysis (tiếp); **Ablation experiment**

```
9
Vol.:(0123456789)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
In summary, when the value is between 0.7 and 0.9, the AUC and AP indices of GCN_MA concurrently 
achieve their optimal values, exhibiting an average increase of 0.52% and 0.45% in AUC and AP respectively, 
compared to the parameter value of 0, across six datasets. This finding suggests that the NRNAE algorithm, con-
structed considering the aggregation effect and aggregation strength, can enhance the information representation 
of nodes. Consequently, it aids GCN in integrating node embedding vectors more effectively, thereby enhancing 
the accuracy and precision of dynamic network link prediction.
Ablation experiment
In this section, we conduct ablation experiments to analyze the contributions of each component in the GCN_
MA model and the role played by each element in the model. Our study introduces three deep learning archi -
tectures in GCN_MA: graph convolutional network, long short-term memory network, and multi-head atten-
tion. To assess the impact of each component on the performance of GCN_MA, we sequentially remove one 
component and compare the results with the complete GCN_MA. The three ablation models are as follows: (1) 
GCN: This variant excludes the long short-term memory network and multi-head attention. (2) GCN_LSTM: 
This variant omits multi-head attention. (3) GCN_MultiAttention: This variant excludes the long short-term 
memory network.
The dynamic network ablation experiments on six real datasets are presented in Tables 3, 4, and Figs. 6, 7. The 
figure demonstrates that all three components significantly contribute to the overall performance of GCN_MA. 
Specifically, GCN_MA exhibits an average improvement of 1.69% and 0.9% in terms of AUC and AP indicators, 
respectively, compared with GCN_MultiAttention. This suggests that the long short-term memory network is 
instrumental in capturing the global time evolution pattern. Furthermore, GCN_MA outperforms GCN_LSTM 
by an average of 4.3% and 3.34% in AUC and AP metrics, respectively. This indicates that the multi-head attention 
model plays a crucial role in capturing the temporal evolution pattern of local information changes in dynamic 
networks. Regarding AUC and AP indicators, GCN_MA averages a 5.12% and 3.62% increase compared to GCN. 
Thus, it can be inferred that the combination of long short-term memory network and multi-head attention 
is effective in modeling the time properties of dynamic networks, allowing for a more comprehensive capture 
of the time evolution pattern. When compared to GCN, GCN_LSTM, and GCN_MultiAttention, the average 
growth rates of AUC and AP for GCN_MA on six datasets are 6.11%, 5.08%, 1.93%, and 4.36%, 4.00%, 1.02%, 
     







$8&
£
&ROOHJH0VJELWFRLQRWFPRRFBDFWLRQV(78ODVW)0ZLNLSHGLD
     







$3
£
Figure 5.  Curves of AUC and AP as a function of parameter β on six real data sets. The results show that when 
the value of β is between 0.7 and 0.9, the AUC and AP indicators of GCN_MA basically reach the optimal value 
at the same time.
Table 3.  AUC results of GCN_MA ablation experiments. Best performing values are in bold.
Methods
Dataset
lastFM Bitcoinotc ETU Wikipedia CollegeMsg Mooc_action
Ours 0.8757 0.9120 0.9222 0.8742 0.9149 0.9880
GCN_MultiAttention 0.8702 0.8916 0.9158 0.8407 0.8809 0.9865
GCN_LSTM 0.7999 0.8743 0.8880 0.8241 0.8694 0.9734
GCN 0.7774 0.8718 0.8755 0.8205 0.8722 0.9626
```

---

## Trang 10 — Ablation (tiếp); **Conclusion**; bắt đầu References

```
10
Vol:.(1234567890)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
respectively. Notably, GCN demonstrates the highest growth rate, indicating that it performs admirably in aggre-
gating node information and significantly contributes to the link prediction of dynamic networks.
The results in Table 3 indicate that the contribution of LSTM is indeed inferior to that of multi-head attention 
when compared with GCN_MultiAttention. This discrepancy arises because LSTM solely focuses on capturing 
the temporal evolution pattern within node embeddings. In contrast, multi-head attention not only captures the 
time evolution pattern but also adjusts the structural characteristics of the network by dynamically weighting 
neighbor nodes’ influence on the target node based on their changes over time. This adjustment plays a crucial 
role in enhancing the accuracy of final node embedding vectors, thereby rendering multi-head attention more 
impactful than LSTM alone. However, combining both mechanisms further enhances link prediction accuracy.
In conclusion, GCN_MA, a model that synchronously integrates multi-dimensional features and analyzes 
time attributes from both global and local perspectives, demonstrates appreciable growth rates in AUC and AP 
indicators. This further validates the effectiveness of the proposed method in dynamic network link prediction.
Conclusion
In this paper, we propose a node representation learning framework based on graph convolutional networks, 
called GCN_MA. The proposed framework aims to capture node information comprehensively and accurately, 
and model the temporal properties of dynamic networks from a global and local perspective to learn the temporal 
Table 4.  AP results of GCN_MA ablation experiments. Best performing values are in bold.
Methods
Dataset
lastFM Bitcoinotc ETU Wikipedia CollegeMsg Mooc_action
Ours 0.8704 0.8943 0.9082 0.8575 0.8926 0.9863
GCN_MultiAttention 0.8577 0.8842 0.8911 0.8511 0.8876 0.9838
GCN_LSTM 0.7872 0.8722 0.8780 0.8293 0.8692 0.9709
GCN 0.7782 0.8752 0.8711 0.8265 0.8754 0.9655
OODDVVWW))00 %%LLWWFFRRLLQQRRWWFF ((7788 ::LLNNLLSSHHGGLLDD &&RROOOOHHJJHH00VVJJ 00RRRRFFBBDDFFWWLLRRQQ






$8&
*&1*&1B/670*&1B0XOWL$WWHQWLRQ2XU
DD˄˄$$88&&˅˅
Figure 6.  Comparison of GCN_MA and different ablation models in AUC on six real datasets.
OODDVVWW))00 %%LLWWFFRRLLQQRRWWFF ((7788 ::LLNNLLSSHHGGLLDD &&RROOOOHHJJHH00VVJJ 00RRRRFFBBDDFFWWLLRRQQ






$3
EE˄˄$$33˅˅
Figure 7.  Comparison of GCN_MA and different ablation models in AP on six real datasets.
```

---

## Trang 11 — References (tiếp)

```
11
Vol.:(0123456789)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
evolution patterns of the network and achieve link prediction. In this paper, NRNAE algorithm is used to enrich 
the node information representation, and GCN is used to aggregate structural features and node features, so as 
to learn the embedding vector representation of each node. At the same time, this paper introduces a recurrent 
neural network with multi-head attention to model the dynamic network from the perspective of global and 
local information changes, respectively, to capture the evolution pattern of time. We experimentally compare 
our framework with four baseline methods on six different datasets. The focus of this paper is on studying dis -
crete dynamic networks with a homogeneous network type. However, in future work, we will shift our attention 
towards heterogeneous dynamic networks and time continuous dynamic networks in order to consider both 
structural similarity and feature-based similarity as measures for node similarity. This will enable us to obtain 
a higher quality low-dimensional representation that can effectively address the problem of dynamic network 
link prediction.
Data availability
The dataset used in this study is available at http:// snap. stanf ord. edu/ data/ act- mooc. html, http:// snap. stanf ord. 
edu/ data/ soc- sign- bitco in- otc. html, http:// snap. stanf ord. edu/ data/ email- Eu- core- tempo ral. html, https:// meta. 
wikim edia. org/ wiki/ Data_ dumps.
Received: 5 November 2023; Accepted: 28 December 2023
References
 1. Xu, F. et al. Specificity and competition of mRNAs dominate droplet pattern in protein phase separation. Phys. Rev. Res. 5(2), 
023159. https:// doi. org/ 10. 1103/ PhysRe- vRese arch.5. 023159 (2023).
 2. Sun, F ., Sun, J. & Zhao, Q. A deep learning method for predicting metabolite-disease associations via graph neural network. Brief 
Bioinform. 23(4), 266. https:// doi. org/ 10. 1093/ bib/ bbac2 66 (2022).
 3. Li, X. et al. Caspase-1 and Gasdermin D afford the optimal targets with distinct switching strategies in NLRP1b inflammasome-
induced cell death. Research https:// doi. org/ 10. 34133/ 2022/ 98383 41 (2022).
 4. Li, X. et al. RIP1-dependent linear and nonlinear recruitments of caspase-8 and RIP3 respectively to necrosome specify distinct 
cell death outcomes. Protein Cell 12(11), 858–876. https://  doi. org/ 10. 1007/ s13238- 020- 00810-x (2021).
 5. Wang, T., Sun, J. & Zhao, Q. Investigating cardiotoxicity related with hERG channel blockers using molecular fingerprints and 
graph attention mechanism. Comput. Biol. Med. 153, 106464. https:// doi. org/ 10. 1016/j. compb iomed. 2022. 106464 (2023).
 6. Liu, W . et al. NSCGRN: A network structure control method for gene regulatory network inference. Brief Bioinform. 23(5), 106464. 
https:// doi. org/ 10. 1093/ bib/ bbac1 56 (2023).
 7. Daud, N. N., Ab Hamid, S. H., Saadoon, M., Sahran, F . & Anuar, N. B. Applications of link prediction in social networks: A review. 
J. Netw. Comput. Appl. 166, 102716. https:// doi. org/ 10. 1016/j. jnca. 2020. 102716 (2022).
 8. Zhou, X., Liang, W ., Luo, Z. & Pan, Y . Periodic-aware intelligent prediction model for information diffusion in social networks. 
IEEE Trans. Netw. Sci. Eng. 8(2), 894–904. https:// doi. org/ 10. 1109/ TNSE. 2021. 30649 52 (2021).
 9. Li, C., Wei, W ., Feng, X. & Liu, J. Research of motif-based similarity for link prediction problem. IEEE Access 9, 66636–66645. 
https:// doi. org/ 10. 1109/ access. 2021. 30770 16 (2021).
 10. Chen, D., Yuan, Z., Huang, X. & Wang, D. Node similarity measurement and link prediction algorithm in temporal networks. J. 
Northeast. Univ. 41(1), 29–35. https:// doi. org/ 10. 12068/j. issn. 1005- 3026. 2020. 01. 006 (2020).
 11. Wu, X., Wu, J., Li, Y . & Zhang, Q. Link prediction of time-evolving network based on node ranking. Knowl. Based Syst. 195, 105740. 
https:// doi. org/ 10. 1016/j. knosys. 2020. 105740 (2020).
 12. Zhang, T., Zhang, K., Lv, L. & Li, X. Temporal link prediction using node centrality and time series. Int. J. Future Comput. Commun. 
https:// doi. org/ 10. 18178/ ijfcc. 2020.9. 3. 567 (2020).
 13. Cui, Z. et al. DyGCN: Efficient dynamic graph embedding with graph convolutional network. IEEE Trans. Neural Netw. Learn. 
Syst. https:// doi. org/ 10. 1109/ TNNLS. 2022. 31855 27 (2022).
 14. Chami, I., Ying, R., Ré, C. & Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. arXiv: Learning (2019).
 15. Xian, X. et al. Generative Graph Neural Networks for Link Prediction. https:// doi. org/ 10. 48550/ arXiv. 2301. 00169 (2022).
 16. Zhou, L., Y ang, Y ., Ren, X., Wu, F . & Zhuang, Y . Dynamic network embedding by modeling triadic closure process. Proc. AAAI 
Conf. Artif. Intell. https:// doi. org/ 10. 1609/ aaai. v32i1. 11257 (2018).
 17. Goyal, P ., Kamra, N., He, X. & Liu, Y . DynGEM: Deep Embedding Method for Dynamic Graphs. https:// doi. org/ 10. 48550/ arXiv. 
1805. 11273 (2018).
 18. Chen, J., Wang, X. & Xu, X. GC-LSTM: Graph convolution embedded LSTM for dynamic network link prediction. Appl. Intell.  
52, 7513–7528. https:// doi. org/ 10. 1007/ s10489- 021- 02518-9 (2022).
 19. Lei, K., Qin, M., Bai, B., Zhang, G. & Y ang, M. GCN-GAN: A non-linear temporal link prediction model for weighted dynamic 
networks. IEEE INFOCOM 2019-IEEE Conference on Computer Communications 388–396. https:// doi. org/ 10. 1109/ INFOC OM. 
2019. 87376 31 (2019).
 20. Wani, M. & Jabin, S. Mutual clustering coefficient-based suspicious-link detection approach for online social networks. J. King 
Saud Univ. Comput. Inf. Sci. 34(2), 218–231. https:// doi. org/ 10. 1016/j. jksuci. 2018. 10. 014 (2018).
 21. Zhang, S., Tong, H., Xu, J. & Maciejewski, R. Graph convolutional networks: A comprehensive review. Comput. Soc. Netw. 6(1), 
1–23. https:// doi. org/ 10. 1186/ s40649- 019- 0069-y (2019).
 22. Selvarajah, K., Ragunathan, K., Kobti, Z. & Kargar, M. Dynamic network link prediction by learning effective subgraphs using 
CNN-LSTM. In 2020 International Joint Conference on Neural Networks (IJCNN) 1–8. https://  doi. org/ 10. 1109/ ijcnn 48605. 2020. 
92073 01 (2020).
 23. Vaswani, A. et al. Attention is all you need. Neural Inf. Process. Syst. 30, 6000–6010 (2017).
 24. Voita, E., Talbot, D., Moiseev, F ., Sennrich, R. & Titov, I. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, 
the rest can be pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. https:// doi. org/ 
10. 18653/ v1/ p19- 1580 (2019).
 25. Lee, H., Sonthalia, R. & Foster, J. G. Dynamic Embedding-based Methods for Link Prediction in Machine Learning Semantic 
Network. In 2021 IEEE International Conference on Big Data (Big Data) 5801–5808. https:// doi. org/ 10. 1109/ BigDa ta525 89. 2021. 
96720 40 (2021).
 26. Ragunathan, K., Selvarajah, K. & Kobti, Z. Link prediction by analyzing common neighbors based subgraphs using convolutional 
neural network. Front. Artif. Intell. Appl. 325, 1906–1913. https:// doi. org/ 10. 3233/ FAIA2 00308 (2021).
 27. Kunegis, J. KONECT: The Koblenz Network Collection. In Proceedings of the 22nd International Conference on World Wide Web 
(Association for Computing Machinery, 2019) 1343–1350. https:// doi. org/ 10. 1145/ 24877 88. 24881 73 (2013).
```

---

## Trang 12 — References (kết thúc)

```
12
Vol:.(1234567890)Scientific Reports |          (2024) 14:538  | https://doi.org/10.1038/s41598-023-50977-6
www.nature.com/scientificreports/
 28. Kumar, S., Zhang, X. & Leskovec, J. Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of 
the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (Association for Computing Machinery, 
2019) 1269–1278. https:// doi. org/ 10. 1145/ 32925 00. 33308 95 (2019).
 29. Kumar, S. et al. REV2: Fraudulent user prediction in rating platforms. In Proceedings of the Eleventh ACM International Conference 
on Web Search and Data Mining (Association for Computing Machinery, 2018) 333–341. https:// doi. org/ 10. 1145/ 31596 52. 31597 29 
(2018).
 30. Paranjape, A., Benson, A. R. & Leskovec, J. Motifs in temporal networks. In Proceedings of the Tenth ACM International Conference 
on Web Search and Data Mining (Association for Computing Machinery, 2017). 601–610. https:// doi. org/ 10. 1145/ 30186 61. 30187 
31 (2017).
 31. Hidasi, B. & Tikk, D. Fast ALS-based tensor factorization for context-aware recommendation from implicit feedback. In Machine 
Learning and Knowledge Discovery in Databases. 67–82. https:// doi. org/ 10. 1007/ 978-3- 642- 33486-3_5 (2012).
 32. West, R. & Leskovec, J. Human Wayfinding in information networks. In Proceedings of the 21st International Conference on World 
Wide Web (Association for Computing Machinery, 2012) 619–628. https:// doi. org/ 10. 1145/ 21878 36. 21879 20 (2012).
 33. Gao, C., Zhu, J., Zhang, F ., Wang, Z. & Li, X. A novel representation learning for dynamic graphs based on graph convolutional 
networks. IEEE Trans. Cybern. 53(6), 3599–3612. https:// doi. org/ 10. 1109/ TCYB. 2022. 31596 61 (2023).
 34. Y ang, M., Zhou, M., Kalander, M., Huang, Z. & King, I. Discrete-time temporal network embedding via implicit hierarchical learn-
ing in hyperbolic space. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (Association 
for Computing Machinery, 2021) 1975–1985. https:// doi. org/ 10. 1145/ 34475 48. 34674 22 (2021).
 35. Ma, Y ., Guo, Z., Ren, Z., Tang, J. & Yin, D. Streaming graph neural networks. In Proceedings of the 43rd International ACM SIGIR 
Conference on Research and Development in Information Retrieval (Association for Computing Machinery, 2020) 719–728. https:// 
doi. org/ 10. 1145/ 33972 71. 34010 92 (2020).
 36. Pareja, A. et al. EvolveGCN: Evolving graph convolutional networks for dynamic graphs. Proc. AAAI Conf. Artif. Intell. 34(04), 
5363–5370. https:// doi. org/ 10. 1609/ aaai. v34i04. 5984 (2020).
Author contributions
P .M. and Y .Z. proposed graph convolutional network based node representation learning framework, GCN_MA 
and NRNAE algorithms. P .M. and Y .Z. conceived and designed the simulation experiment. P .M. did an experi-
ment. P .M. and Y .Z. analyzed the results of the simulation experiment. P .M. and Y .Z. wrote the manuscript text. 
P .M. will summarize and record the experimental results. Both authors reviewed the manuscript.
Funding
This work was supported by Natural Science Foundation of Inner Mongolia Province of China (Grant No. 
2022MS06006).
Competing interests 
The authors declare no competing interests.
Additional information
Correspondence and requests for materials should be addressed to Y .h.Z.
Reprints and permissions information is available at www.nature.com/reprints.
Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and 
institutional affiliations.
Open Access  This article is licensed under a Creative Commons Attribution 4.0 International 
License, which permits use, sharing, adaptation, distribution and reproduction in any medium or 
format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the 
Creative Commons licence, and indicate if changes were made. The images or other third party material in this 
article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the 
material. If material is not included in the article’s Creative Commons licence and your intended use is not 
permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from 
the copyright holder. To view a copy of this licence, visit http:// creat iveco mmons. org/ licen ses/ by/4. 0/.
© The Author(s) 2024
```

---
