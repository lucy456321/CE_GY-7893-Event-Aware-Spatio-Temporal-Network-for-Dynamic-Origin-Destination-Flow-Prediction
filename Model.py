"""
Event-Aware Spatio-Temporal GNN  
Model Definition (EASTGNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

#Graph attention layer 

class GraphAttentionLayer(nn.Module):
    """Custom Graph Attention Layer (GAT)"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Learnable weight matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args: h: [batch, num_nodes, in_features], adj_matrix: [num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = h.size()
        
        Wh = torch.matmul(h, self.W)  # [batch, num_nodes, out_features]
        
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        Wh2 = Wh.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        Wh_concat = torch.cat([Wh1, Wh2], dim=-1)
        
        e = self.leakyrelu(torch.matmul(Wh_concat, self.a).squeeze(-1))
        
        mask = (adj_matrix == 0).unsqueeze(0).expand(batch_size, -1, -1)
        e = e.masked_fill(mask, float('-inf'))
        
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime


class MultiHeadGraphAttention(nn.Module):
    """Multi-head graph attention"""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, self.head_dim, dropout)
            for _ in range(num_heads)
        ])
        
        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x, adj_matrix) for head in self.heads]
        
        output = torch.cat(head_outputs, dim=-1)
        
        output = self.out_proj(output)
        output = self.dropout(output)
        
        output = self.norm(output)
        
        return output


#Event conditioning module (FiLM)

class EventConditioningLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM)"""
    
    def __init__(self, hidden_dim: int, event_dim: int):
        super().__init__()
        
        self.gamma_net = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, event_features: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(event_features)
        beta = self.beta_net(event_features)
        
        return (1 + gamma) * x + beta


# SPATIAL ENCODING MODULE 

class SpatialEncoder(nn.Module):
    """Spatial graph encoder with multiple GAT layers"""
    
    def __init__(self, hidden_dim: int, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MultiHeadGraphAttention(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = x 
            
            x_new = layer(x, adj_matrix)
            
            x = residual + self.dropout(x_new)
        
        return x


#TEMPORAL ENCODING MODULE

class TemporalEncoder(nn.Module):
    """Temporal encoder with GRU and attention"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, T, N, F = x.shape
        
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, F)
        
        gru_out, hidden = self.gru(x_reshaped)
        
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        attn_out = self.norm(attn_out + gru_out)
        
        output = attn_out[:, -1, :]
        
        output = output.reshape(batch, N, -1)
        
        return output, hidden


#OD FLOW DECODER

class ODFlowDecoder(nn.Module):
    """Decode node embeddings to OD flow matrix"""
    
    def __init__(self, hidden_dim: int, num_zones: int, output_horizons: int):
        super().__init__()
        self.num_zones = num_zones
        self.output_horizons = output_horizons
        
        self.origin_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.dest_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.bilinear = nn.Bilinear(hidden_dim // 2, hidden_dim // 2, 1)
        
        self.horizon_predictor = nn.Sequential(
            nn.Linear(1, output_horizons),
            nn.ReLU()
        )
        
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = node_embeddings.size(0)
        
        origin_features = self.origin_encoder(node_embeddings)
        dest_features = self.dest_encoder(node_embeddings)
        
        origin_exp = origin_features.unsqueeze(2).expand(-1, -1, self.num_zones, -1)
        dest_exp = dest_features.unsqueeze(1).expand(-1, self.num_zones, -1, -1)
        
        od_scores = self.bilinear(
            origin_exp.reshape(-1, origin_features.size(-1)),
            dest_exp.reshape(-1, dest_features.size(-1))
        ).reshape(batch_size, self.num_zones, self.num_zones, 1)
        
        od_predictions = self.horizon_predictor(od_scores)
        
        od_predictions = od_predictions.permute(0, 3, 1, 2)
        
        return F.relu(od_predictions)


#COMPLETE MODEL

class EASTGNNModel(nn.Module):
    """Event-Aware Spatio-Temporal Graph Neural Network"""
    
    def __init__(
        self,
        num_zones: int,
        event_feature_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_temporal_layers: int = 2,
        output_horizons: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_zones = num_zones
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(1, hidden_dim)
        self.event_conditioning = EventConditioningLayer(hidden_dim, event_feature_dim)
        
        self.spatial_encoder = SpatialEncoder(
            hidden_dim, num_layers=num_gnn_layers, num_heads=4, dropout=dropout
        )
        
        self.temporal_encoder = TemporalEncoder(
            hidden_dim, hidden_dim, num_layers=num_temporal_layers, dropout=dropout
        )
        
        self.od_decoder = ODFlowDecoder(hidden_dim, num_zones, output_horizons)
        
    def forward(
        self,
        historical_od: torch.Tensor,
        event_features: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        batch, T, N, _ = historical_od.shape
        
        # 1. Aggregate OD to node-level features
        origin_flows = historical_od.sum(dim=3)
        dest_flows = historical_od.sum(dim=2)
        node_flows = origin_flows + dest_flows
        
        node_features = self.input_proj(node_flows.unsqueeze(-1))  # [B, T, N, H]
        
        # 2. Event-Aware Spatial Encoding 
        flat_node_features = node_features.reshape(batch * T, N, self.hidden_dim)
        flat_event_features = event_features.reshape(batch * T, N, -1)
        
        conditioned_features_flat = self.event_conditioning(
            flat_node_features, flat_event_features
        )
        
        spatial_features_flat = self.spatial_encoder(conditioned_features_flat, adj_matrix)
        
        spatial_features = spatial_features_flat.reshape(batch, T, N, self.hidden_dim)
        
        # 3. Temporal encoding
        temporal_output, _ = self.temporal_encoder(spatial_features)
        
        # 4. Decode to OD predictions
        predictions = self.od_decoder(temporal_output)
        
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        pass


def create_adjacency_matrix(num_zones: int, k: int = 8) -> torch.Tensor:
    """Create adjacency matrix (k-nearest neighbors demo)."""
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        print("Warning: scipy not found. Adjacency matrix will be random and dense.")
        return torch.ones(num_zones, num_zones).float()
    
    coords = np.random.randn(num_zones, 2)
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    adj_matrix = np.zeros((num_zones, num_zones), dtype=np.float32)
    
    for i in range(num_zones):
        nearest = np.argsort(dist_matrix[i])[1:k+1]
        adj_matrix[i, nearest] = 1
        adj_matrix[nearest, i] = 1
    
    adj_matrix[np.arange(num_zones), np.arange(num_zones)] = 1
    
    return torch.from_numpy(adj_matrix).float()
