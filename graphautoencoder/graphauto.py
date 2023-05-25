import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import dgl
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
# Cora 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
labels = dataset.y
# 그래프 오토인코더 모델 정의
class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, input_dim2,hidden_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        
        self.decoder = nn.Linear(hidden_dim, input_dim2)
    
    def forward(self, x, edge_index):
        encoded = self.encoder(x, edge_index)
        decoded = torch.sigmoid(self.decoder(encoded))
        return decoded

# 모델 인스턴스 생성
input_dim1 = dataset.num_features
input_dim2 = len(dataset.train_mask)
hidden_dim = 64
model = GraphAutoencoder(input_dim1,input_dim2, hidden_dim)

# 옵티마이저 및 손실 함수 정의
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()
data1 = dgl.from_networkx(to_networkx(data))

# 인접 행렬로 변환
adj_matrix = data1.adjacency_matrix().to_dense()# 학습 함수 정의
def train(model, data, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, adj_matrix)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 학습
train(model, data, optimizer, criterion, epochs=100)

# 인접 행렬 복원
model.eval()
with torch.no_grad():
    reconstructed_adj = model(data.x, data.edge_index)
    diff = torch.abs(adj_matrix[1708:] - reconstructed_adj[1708:])
    row_diff_sum = torch.sum(diff, dim=1) # node별로 로스값 합산
    _, top_indices = torch.topk(row_diff_sum, k=10) #가장 높은거 
    for i in range(len(top_indices)):
        idx = top_indices[i]
        row_diff = row_diff_sum[idx].item()
        row = diff[idx] #로스 불러오기
        col_diff_max, col_idx = torch.topk(row, k=5) #그 중 가장 높은거 5개
        #print(col_idx)
        #col = col_idx.item()
test_top_indices = top_indices+1708
# 인접 행렬 출력
wrong_list=[1708, 1733, 1735, 1741, 1764, 1774, 1792, 1793, 1794, 1801, 1802, 1803, 1813, 1832, 1834, 1840, 1844, 1845, 1854, 1892, 1893, 1905, 1907, 1908, 1910, 1911, 1920, 1924, 1937, 1938, 1943, 1977, 1983, 1989, 1990, 1995, 1998, 2003, 2010, 2012, 2013, 2014, 2015, 2016, 2018, 2019, 2023, 2024, 2028, 2031, 2032, 2033, 2035, 2038, 2044, 2047, 2058, 2068, 2071, 2078, 2089, 2101, 2102, 2103, 2104, 2105, 2108, 2111, 2119, 2124, 2125, 2126, 2128, 2129, 2149, 2150, 2153, 2154, 2166, 2168, 2169, 2170, 2175, 2177, 2178, 2187, 2188, 2231, 2246, 2253, 2255, 2256, 2257, 2268, 2274, 2276, 2282, 2291, 2293, 2299, 2301, 2302, 2306, 2313, 2314, 2316, 2319, 2320, 2323, 2330, 2332, 2341, 2345, 2346, 2348, 2349, 2353, 2354, 2355, 2357, 2359, 2360, 2372, 2373, 2385, 2388, 2397, 2402, 2405, 2406, 2409, 2413, 2423, 2425, 2427, 2428, 2433, 2436, 2437, 2438, 2453, 2456, 2462, 2467, 2468, 2469, 2470, 2482, 2484, 2485, 2490, 2496, 2510, 2517, 2533, 2545, 2547, 2553, 2561, 2562, 2569, 2571, 2573, 2578, 2580, 2584, 2586, 2590, 2600, 2606, 2607, 2611, 2615, 2620, 2639, 2640, 2654, 2655, 2656, 2660, 2669, 2690, 2692, 2699, 2705]
print(reconstructed_adj.shape)
cnt = 0
for i in test_top_indices:
    if i in wrong_list:
        cnt += 1
print(cnt)
#이 코드는 주변 노드와 클래스를 비교하기 위함
'''
graph_list = nx.to_dict_of_lists(to_networkx(data))

for i in top_indices:
    #print(graph_list[int(i)])
    print('diffmax',labels[int(i)])
    for j in graph_list[int(i)]:
        print('neigh',labels[j])
    print('end')


### 여기서부터는 시각화를 위함
G = to_networkx(data)

# 노드 색상 설정
num_classes = dataset.num_classes
node_colors = ['lightgray'] * len(G.nodes)
for i in range(len(data.y)):
    if i in top_indices:
        node_colors[i] = plt.cm.tab10(data.y[i].item() / num_classes)

edge_colors = {}

# 해당 노드에만 연결된 엣지 색상 지정
edge_li = []
excluded_edges = []
for edge in G.edges():
    source,target = edge
    if source in top_indices or target in top_indices:
        edge_li.append(tuple((source,target)))


    else:
        excluded_edges.append(tuple((source,target)))
G.remove_edges_from(excluded_edges)

for edge in G.edges():
    source,target = edge
    edge_colors[edge] = 'blue'

#print(edge_li)
# 그래프 시각화
plt.figure(figsize=(20, 20))
# 노드 사이즈 설정
node_size = 100
node_size_list = [node_size] * len(G.nodes)

# 노드 라벨 표시 크기 설정
label_font_size = 5

# 그래프 그리기
pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)
nx.draw_networkx(G, pos=pos, node_color=node_colors, node_size=node_size_list,
                 edge_color=[edge_colors[edge] for edge in G.edges], with_labels=True, font_size=label_font_size)
plt.axis('off')
#plt.show()
plt.savefig('1.png')
'''