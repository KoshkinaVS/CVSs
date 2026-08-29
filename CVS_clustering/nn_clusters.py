import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

class CycloneDataset(Dataset):
    def __init__(self, cyclones, path_list, basenames_list, time_window=3):
        self.cyclones = cyclones
        self.path_list = path_list
        self.basenames_list = basenames_list
        self.time_window = time_window
        self._prepare_pairs()
        
    def _prepare_pairs(self):
        # Группируем треки по файлам (предполагая, что path_list содержит пути к файлам)
        self.track_indices = {}
        for i, path in enumerate(self.path_list):
            file_id = str(path)  # или другая уникальная идентификация трека
            if file_id not in self.track_indices:
                self.track_indices[file_id] = []
            self.track_indices[file_id].append(i)
        
        # Создаем пары индексов
        self.pairs = []
        for file_id, indices in self.track_indices.items():
            # Для каждого трека создаем пары близких по времени точек
            for i in range(len(indices)):
                for j in range(i+1, min(i+self.time_window+1, len(indices))):
                    self.pairs.append((indices[i], indices[j]))  # положительная пара
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        # Нормализуем данные (важно для контрастивного обучения)
        x_i = (self.cyclones[i] - self.cyclones.mean(axis=0)) / self.cyclones.std(axis=0)
        x_j = (self.cyclones[j] - self.cyclones.mean(axis=0)) / self.cyclones.std(axis=0)
        return torch.FloatTensor(x_i), torch.FloatTensor(x_j)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.net(x)

class SimCLR(nn.Module):
    def __init__(self, encoder, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64))
        
    def forward(self, x_i, x_j):
        # Получаем представления
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        # Проекция в пространство для контрастивного обучения
        z_i = self.projection(h_i)
        z_j = self.projection(h_j)
        
        return h_i, h_j, z_i, z_j
    
    def loss(self, z_i, z_j):
        # Вычисляем контрастивную потерю
        batch_size = z_i.size(0)
        
        # Объединяем все представления
        z = torch.cat([z_i, z_j], dim=0)
        
        # Матрица сходств
        sim_matrix = torch.exp(torch.mm(z, z.t()) / self.temperature)
        
        # Маска для положительных пар
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask = mask.repeat(2, 2)
        mask.fill_diagonal_(False)
        
        # Потеря для положительных пар
        pos_sim = torch.cat([torch.diag(sim_matrix, batch_size), 
                            torch.diag(sim_matrix, -batch_size)], dim=0)
        pos_loss = -torch.log(pos_sim).mean()
        
        # Потеря для отрицательных пар
        neg_sim = sim_matrix[mask].view(2*batch_size, -1)
        neg_loss = torch.log(neg_sim.sum(dim=1)).mean()
        
        return (pos_loss + neg_loss) / 2

def train_simclr(dataset, epochs=50, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(input_dim=dataset.cyclones.shape[1])
    model = SimCLR(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for x_i, x_j in dataloader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            
            optimizer.zero_grad()
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            loss = model.loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model.encoder

def get_CVS_clusters_with_simclr():
    # [Ваш существующий код загрузки данных до cyclones = np.array(df_max[param_cols[2:]])]
    
    # Нормализация данных
    cyclones_norm = (cyclones - cyclones.mean(axis=0)) / cyclones.std(axis=0)
    
    # Создаем датасет для контрастивного обучения
    dataset = CycloneDataset(cyclones_norm, path_list, basenames_list)
    
    # Обучаем кодировщик с помощью SimCLR
    encoder = train_simclr(dataset)
    
    # Получаем представления для всех данных
    with torch.no_grad():
        cyclones_tensor = torch.FloatTensor(cyclones_norm)
        embeddings = encoder(cyclones_tensor).numpy()
    
    # Кластеризация в пространстве представлений
    clusters = KMeans(n_clusters=n_clusters).fit_predict(embeddings)
    
    # [Остальной ваш код для сохранения результатов]
    cluster_groups, names_groups = group_cyclones(cyclones, clusters, path_list, basenames_list)
    
    # Создаем список для хранения информации о кластерах и файлах
    cluster_idx_list = []
    names_list = []

    for cluster in cluster_groups.keys():
        for cluster_idx in cluster_groups[cluster]:
            cluster_idx_list.append(cluster)
            names_list.append(names_groups[cluster][cluster_groups[cluster].index(cluster_idx)])

    # Сохраняем результат в CSV
    cluster_names = pd.DataFrame(data={'path': [p[0] for p in names_list], 
                                   'file_name': [p[1] for p in names_list], 
                                   'cluster': cluster_idx_list}) 
    cluster_names.to_csv(f'{path_tracks_dir}/{n_clusters}_clusters_names-{year}_simclr.csv', index=False)