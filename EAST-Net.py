#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch torchvision torchaudio')


# In[2]:


pip install scipy


# In[2]:


"Data processing"

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import torch
from tqdm import tqdm
import logging
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(
        self,
        hvfhv_path: str,
        events_path: str,
        output_dir: str = './processed_data',
        time_resolution: int = 15,
        num_zones: int = 263
    ):
        self.hvfhv_path = hvfhv_path
        self.events_path = events_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.time_resolution = time_resolution
        self.num_zones = num_zones

        logger.info("=" * 80)
        logger.info("Data processor initialized")
        logger.info("=" * 80)
        logger.info(f"HVFHV data: {hvfhv_path}")
        logger.info(f"Events data: {events_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Time resolution: {time_resolution} minutes")
        logger.info(f"Number of zones: {num_zones}")

    def load_hvfhv_data(self) -> pd.DataFrame:
        """Load and clean HVFHV trip data"""
        logger.info("\n" + "=" * 80)
        logger.info("Loading HVFHV data")
        logger.info("=" * 80)

        try:
            df = pd.read_parquet(self.hvfhv_path)
            logger.info(f"Loaded {len(df):,} trips")

            # Convert datetime columns
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

            # Filter valid location IDs
            before_filter = len(df)
            df = df[
                (df['PULocationID'] >= 1) & (df['PULocationID'] <= self.num_zones) &
                (df['DOLocationID'] >= 1) & (df['DOLocationID'] <= self.num_zones)
            ].copy()
            logger.info(f" After location filter: {len(df):,} trips ({len(df)/before_filter*100:.1f}%)")

            # Remove invalid trips
            df = df[df['trip_time'] > 0]
            df = df[df['trip_miles'] > 0]
            logger.info(f" After validity filter: {len(df):,} trips")

            # Get time range
            start_time = df['pickup_datetime'].min()
            end_time = df['pickup_datetime'].max()
            logger.info(f" Time range: {start_time} to {end_time}")

            return df

        except Exception as e:
            logger.error(f" Error loading HVFHV data: {e}")
            raise

    def load_events_data(self) -> pd.DataFrame:
        """Load and clean events data"""
        logger.info("\n" + "=" * 80)
        logger.info("Loading events data")
        logger.info("=" * 80)

        try:
            df = pd.read_csv(self.events_path)
            logger.info(f" Loaded {len(df):,} events")

            # Convert datetime columns
            df['Start Date/Time'] = pd.to_datetime(df['Start Date/Time'])
            df['End Date/Time'] = pd.to_datetime(df['End Date/Time'])

            # Filter valid location IDs
            before_filter = len(df)
            df = df[df['LocationID'].notna()].copy()
            df = df[(df['LocationID'] >= 1) & (df['LocationID'] <= self.num_zones)]
            logger.info(f" After location filter: {len(df):,} events ({len(df)/before_filter*100:.1f}%)")

            # Event types
            event_types = df['Event Type'].value_counts()
            logger.info(f" Event types: {len(event_types)}")
            for i, (etype, count) in enumerate(event_types.head(10).items()):
                logger.info(f"  {i+1}. {etype}: {count:,}")

            return df

        except Exception as e:
            logger.error(f" Error loading events data: {e}")
            raise

    def create_time_bins(self, start_time: datetime, end_time: datetime) -> pd.DatetimeIndex:
        """Create time bins for aggregation"""
        bins = pd.date_range(
            start=start_time.floor(f'{self.time_resolution}min'),
            end=end_time.ceil(f'{self.time_resolution}min'),
            freq=f'{self.time_resolution}min'
        )
        logger.info(f"\n Created {len(bins)-1} time bins")
        return bins

    def build_od_matrices(self, trips_df: pd.DataFrame, time_bins: pd.DatetimeIndex) -> np.ndarray:
        """Build OD matrices for each time bin"""
        logger.info("\n" + "=" * 80)
        logger.info("Building OD Matrices")
        logger.info("=" * 80)

        num_bins = len(time_bins) - 1
        od_tensor = np.zeros((num_bins, self.num_zones, self.num_zones), dtype=np.float32)

        # Bin trips by pickup time
        trips_df['time_bin'] = pd.cut(
            trips_df['pickup_datetime'],
            bins=time_bins,
            labels=range(num_bins),
            include_lowest=True
        )

        # Count trips per OD pair per time bin
        logger.info("Aggregating trips...")
        for time_idx in tqdm(range(num_bins), desc="Time bins"):
            bin_trips = trips_df[trips_df['time_bin'] == time_idx]

            if len(bin_trips) > 0:
                od_counts = bin_trips.groupby(['PULocationID', 'DOLocationID']).size()

                for (origin, dest), count in od_counts.items():
                    od_tensor[time_idx, int(origin)-1, int(dest)-1] = count

        # Statistics
        logger.info(f"\ OD tensor shape: {od_tensor.shape}")
        logger.info(f" Total flows: {od_tensor.sum():,.0f}")
        logger.info(f" Mean flow per OD pair: {od_tensor.mean():.2f}")
        logger.info(f" Std flow: {od_tensor.std():.2f}")
        logger.info(f" Max flow: {od_tensor.max():.0f}")
        logger.info(f" Non-zero OD pairs: {(od_tensor > 0).sum():,} / {od_tensor.size:,} ({(od_tensor > 0).sum()/od_tensor.size*100:.1f}%)")

        return od_tensor

    def engineer_event_features(
        self,
        events_df: pd.DataFrame,
        time_bins: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, list]:
        """Engineer comprehensive event features"""
        logger.info("\n" + "=" * 80)
        logger.info("Engineering event features")
        logger.info("=" * 80)

        num_bins = len(time_bins) - 1

        # Event type encoding
        event_types = events_df['Event Type'].value_counts().head(10).index.tolist()
        event_type_map = {evt: idx for idx, evt in enumerate(event_types)}
        logger.info(f"Top 10 event types encoded")

        # Feature dimensions
        # 1: event_count
        # 2-11: event_type_onehot (10 types)
        # 12: time_to_event
        # 13: event_duration
        # 14: event_active
        # 15: hour_sin
        # 16: hour_cos
        # 17: day_sin
        # 18: day_cos
        # 19: is_weekend
        # 20: is_rush_hour
        # 21-22: reserved
        num_features = 22
        event_tensor = np.zeros((num_bins, self.num_zones, num_features), dtype=np.float32)

        logger.info("Processing events for each time bin...")
        for time_idx in tqdm(range(num_bins), desc="Time bins"):
            bin_start = time_bins[time_idx]
            bin_end = time_bins[time_idx + 1]
            bin_center = bin_start + (bin_end - bin_start) / 2

            for zone_id in range(1, self.num_zones + 1):
                zone_idx = zone_id - 1

                # Get events in this zone
                zone_events = events_df[events_df['LocationID'] == zone_id]

                if len(zone_events) == 0:
                    # Add temporal features even without events
                    hour = bin_center.hour
                    day = bin_center.dayofweek

                    event_tensor[time_idx, zone_idx, 14] = np.sin(2 * np.pi * hour / 24)
                    event_tensor[time_idx, zone_idx, 15] = np.cos(2 * np.pi * hour / 24)
                    event_tensor[time_idx, zone_idx, 16] = np.sin(2 * np.pi * day / 7)
                    event_tensor[time_idx, zone_idx, 17] = np.cos(2 * np.pi * day / 7)
                    event_tensor[time_idx, zone_idx, 18] = 1.0 if day >= 5 else 0.0
                    event_tensor[time_idx, zone_idx, 19] = 1.0 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0.0
                    continue

                # Feature 1: Event count (active events)
                active_events = zone_events[
                    (zone_events['Start Date/Time'] <= bin_end) &
                    (zone_events['End Date/Time'] >= bin_start)
                ]
                event_tensor[time_idx, zone_idx, 0] = len(active_events)

                # Features 2-11: Event type one-hot
                if len(active_events) > 0:
                    for evt_type in active_events['Event Type'].values:
                        if evt_type in event_type_map:
                            type_idx = event_type_map[evt_type]
                            if type_idx < 10:
                                event_tensor[time_idx, zone_idx, 1 + type_idx] = 1

                # Feature 12: Time to nearest event (hours)
                all_times = pd.concat([
                    zone_events['Start Date/Time'],
                    zone_events['End Date/Time']
                ])
                if len(all_times) > 0:
                    time_diffs = (all_times - bin_center).abs()
                    min_diff = time_diffs.min().total_seconds() / 3600
                    event_tensor[time_idx, zone_idx, 11] = min(min_diff, 24.0)  # Cap at 24 hours

                # Feature 13: Event duration (hours)
                if len(active_events) > 0:
                    durations = (
                        active_events['End Date/Time'] -
                        active_events['Start Date/Time']
                    ).dt.total_seconds() / 3600
                    event_tensor[time_idx, zone_idx, 12] = durations.mean()

                # Feature 14: Event active (binary)
                event_tensor[time_idx, zone_idx, 13] = 1.0 if len(active_events) > 0 else 0.0

                # Features 15-20: Temporal context
                hour = bin_center.hour
                day = bin_center.dayofweek

                event_tensor[time_idx, zone_idx, 14] = np.sin(2 * np.pi * hour / 24)
                event_tensor[time_idx, zone_idx, 15] = np.cos(2 * np.pi * hour / 24)
                event_tensor[time_idx, zone_idx, 16] = np.sin(2 * np.pi * day / 7)
                event_tensor[time_idx, zone_idx, 17] = np.cos(2 * np.pi * day / 7)
                event_tensor[time_idx, zone_idx, 18] = 1.0 if day >= 5 else 0.0
                event_tensor[time_idx, zone_idx, 19] = 1.0 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0.0

        logger.info(f"\ Event tensor shape: {event_tensor.shape}")
        logger.info(f" Features per zone: {num_features}")
        logger.info(f" Bins with events: {(event_tensor[:,:,0] > 0).any(axis=1).sum()} / {num_bins}")

        feature_names = [
            'event_count',
            *[f'event_type_{i}' for i in range(10)],
            'time_to_event', 'event_duration', 'event_active',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_rush_hour', 'reserved1', 'reserved2'
        ]

        return event_tensor, feature_names

    def normalize_data(self, od_tensor: np.ndarray, event_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Normalize data for training"""
        logger.info("\n" + "=" * 80)
        logger.info("Normalizing data")
        logger.info("=" * 80)

        # OD normalization: log1p + standardization
        od_log = np.log1p(od_tensor)
        od_mean = od_log.mean()
        od_std = od_log.std()
        od_normalized = (od_log - od_mean) / (od_std + 1e-8)

        logger.info(f" OD normalization:")
        logger.info(f"  Original range: [{od_tensor.min():.2f}, {od_tensor.max():.2f}]")
        logger.info(f"  Normalized range: [{od_normalized.min():.2f}, {od_normalized.max():.2f}]")

        # Event normalization: per-feature standardization
        event_normalized = np.zeros_like(event_tensor)
        for feat_idx in range(event_tensor.shape[2]):
            feat_data = event_tensor[:, :, feat_idx]
            mean = feat_data.mean()
            std = feat_data.std()
            if std > 1e-8:
                event_normalized[:, :, feat_idx] = (feat_data - mean) / std
            else:
                event_normalized[:, :, feat_idx] = feat_data

        logger.info(f" Event normalization: {event_tensor.shape[2]} features")

        # Statistics for denormalization
        stats = {
            'od_mean': float(od_mean),
            'od_std': float(od_std),
            'od_raw_mean': float(od_tensor.mean()),
            'od_raw_std': float(od_tensor.std()),
            'od_p50': float(np.percentile(od_tensor, 50)),
            'od_p75': float(np.percentile(od_tensor, 75)),
            'od_p90': float(np.percentile(od_tensor, 90)),
            'od_p95': float(np.percentile(od_tensor, 95)),
            'od_p99': float(np.percentile(od_tensor, 99)),
        }

        return od_normalized, event_normalized, stats

    def split_data(
        self,
        od_tensor: np.ndarray,
        event_tensor: np.ndarray,
        time_bins: pd.DatetimeIndex,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict:
        """Split data temporally"""
        logger.info("\n" + "=" * 80)
        logger.info("Splitting data")
        logger.info("=" * 80)

        num_samples = len(od_tensor)
        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * val_ratio)

        splits = {
            'train': {
                'od': od_tensor[:train_size],
                'events': event_tensor[:train_size],
                'times': time_bins[:train_size+1]
            },
            'val': {
                'od': od_tensor[train_size:train_size+val_size],
                'events': event_tensor[train_size:train_size+val_size],
                'times': time_bins[train_size:train_size+val_size+1]
            },
            'test': {
                'od': od_tensor[train_size+val_size:],
                'events': event_tensor[train_size+val_size:],
                'times': time_bins[train_size+val_size:]
            }
        }

        logger.info(f" Train: {len(splits['train']['od'])} samples ({train_ratio*100:.0f}%)")
        logger.info(f" Val:   {len(splits['val']['od'])} samples ({val_ratio*100:.0f}%)")
        logger.info(f" Test:  {len(splits['test']['od'])} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")

        return splits

    def save_processed_data(self, splits: Dict, stats: Dict, metadata: Dict, feature_names: list):
        """Save all processed data"""
        logger.info("\n" + "=" * 80)
        logger.info("SAVING PROCESSED DATA")
        logger.info("=" * 80)

        # Save tensors
        for split_name, split_data in splits.items():
            save_path = self.output_dir / f'{split_name}_data.pt'
            torch.save({
                'od_tensor': torch.from_numpy(split_data['od']),
                'event_tensor': torch.from_numpy(split_data['events']),
                'timestamps': split_data['times']
            }, save_path)
            logger.info(f" Saved {split_name} data: {save_path}")

        # Save statistics
        np.savez(
            self.output_dir / 'statistics.npz',
            **stats
        )
        logger.info(f" Saved statistics: {self.output_dir / 'statistics.npz'}")

        # Save metadata
        metadata['feature_names'] = feature_names
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f" Saved metadata: {self.output_dir / 'metadata.json'}")

        logger.info(f"\n All data saved to: {self.output_dir}")

    def process(self):
        """Main preprocessing pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("Start preprocessing pipeline")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Load data
        trips_df = self.load_hvfhv_data()
        events_df = self.load_events_data()

        # Time range
        data_start = trips_df['pickup_datetime'].min()
        data_end = trips_df['pickup_datetime'].max()

        # Create time bins
        time_bins = self.create_time_bins(data_start, data_end)

        # Build OD matrices
        od_tensor = self.build_od_matrices(trips_df, time_bins)

        # Engineer event features
        event_tensor, feature_names = self.engineer_event_features(events_df, time_bins)

        # Normalize
        od_normalized, event_normalized, stats = self.normalize_data(od_tensor, event_tensor)

        # Split data
        splits = self.split_data(od_normalized, event_normalized, time_bins)

        # Prepare metadata
        metadata = {
            'num_zones': self.num_zones,
            'time_resolution': self.time_resolution,
            'num_time_bins': len(time_bins) - 1,
            'start_time': str(data_start),
            'end_time': str(data_end),
            'num_trips': len(trips_df),
            'num_events': len(events_df),
            'event_feature_dim': event_tensor.shape[2],
            'train_samples': len(splits['train']['od']),
            'val_samples': len(splits['val']['od']),
            'test_samples': len(splits['test']['od'])
        }

        # Save
        self.save_processed_data(splits, stats, metadata, feature_names)

        # Summary
        elapsed = datetime.now() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETED")
        logger.info("=" * 80)
        logger.info(f" Time elapsed: {elapsed}")
        logger.info(f" Output directory: {self.output_dir}")
        logger.info(f" Ready for training!")
        logger.info("=" * 80)


def main():
    """Run preprocessing"""

    # Configuration
    preprocessor = DataPreprocessor(
        hvfhv_path='manhattan_merged.parquet',
        events_path='mapped_events.csv',
        output_dir='./processed_data',
        time_resolution=15,
        num_zones=263
    )

    # Run preprocessing
    preprocessor.process()


if __name__ == '__main__':
    main()


# In[3]:


get_ipython().run_cell_magic('writefile', 'model.py', '"""\nEvent-Aware Spatio-Temporal GNN  \nModel Definition (EASTGNN)\n"""\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom typing import Optional, Tuple\nimport numpy as np\n\n#Graph attention layer \n\nclass GraphAttentionLayer(nn.Module):\n    """Custom Graph Attention Layer (GAT)"""\n    \n    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):\n        super().__init__()\n        self.in_features = in_features\n        self.out_features = out_features\n        self.dropout = dropout\n        self.alpha = alpha\n        \n        # Learnable weight matrix\n        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))\n        nn.init.xavier_uniform_(self.W.data, gain=1.414)\n        \n        # Attention mechanism parameters\n        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))\n        nn.init.xavier_uniform_(self.a.data, gain=1.414)\n        \n        self.leakyrelu = nn.LeakyReLU(self.alpha)\n        \n    def forward(self, h: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:\n        """\n        Forward pass\n        Args: h: [batch, num_nodes, in_features], adj_matrix: [num_nodes, num_nodes]\n        """\n        batch_size, num_nodes, _ = h.size()\n        \n        Wh = torch.matmul(h, self.W)  # [batch, num_nodes, out_features]\n        \n        Wh1 = Wh.unsqueeze(2).expand(-1, -1, num_nodes, -1)\n        Wh2 = Wh.unsqueeze(1).expand(-1, num_nodes, -1, -1)\n        Wh_concat = torch.cat([Wh1, Wh2], dim=-1)\n        \n        e = self.leakyrelu(torch.matmul(Wh_concat, self.a).squeeze(-1))\n        \n        mask = (adj_matrix == 0).unsqueeze(0).expand(batch_size, -1, -1)\n        e = e.masked_fill(mask, float(\'-inf\'))\n        \n        attention = F.softmax(e, dim=-1)\n        attention = F.dropout(attention, self.dropout, training=self.training)\n        \n        h_prime = torch.matmul(attention, Wh)\n        \n        return h_prime\n\n\nclass MultiHeadGraphAttention(nn.Module):\n    """Multi-head graph attention"""\n    \n    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):\n        super().__init__()\n        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"\n        \n        self.num_heads = num_heads\n        self.head_dim = out_features // num_heads\n        \n        self.heads = nn.ModuleList([\n            GraphAttentionLayer(in_features, self.head_dim, dropout)\n            for _ in range(num_heads)\n        ])\n        \n        self.out_proj = nn.Linear(out_features, out_features)\n        self.norm = nn.LayerNorm(out_features)\n        self.dropout = nn.Dropout(dropout)\n        \n    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:\n        head_outputs = [head(x, adj_matrix) for head in self.heads]\n        \n        output = torch.cat(head_outputs, dim=-1)\n        \n        output = self.out_proj(output)\n        output = self.dropout(output)\n        \n        output = self.norm(output)\n        \n        return output\n\n\n#Event conditioning module (FiLM)\n\nclass EventConditioningLayer(nn.Module):\n    """Feature-wise Linear Modulation (FiLM)"""\n    \n    def __init__(self, hidden_dim: int, event_dim: int):\n        super().__init__()\n        \n        self.gamma_net = nn.Sequential(\n            nn.Linear(event_dim, hidden_dim),\n            nn.ReLU(),\n            nn.Linear(hidden_dim, hidden_dim),\n            nn.Tanh()\n        )\n        \n        self.beta_net = nn.Sequential(\n            nn.Linear(event_dim, hidden_dim),\n            nn.ReLU(),\n            nn.Linear(hidden_dim, hidden_dim)\n        )\n        \n    def forward(self, x: torch.Tensor, event_features: torch.Tensor) -> torch.Tensor:\n        gamma = self.gamma_net(event_features)\n        beta = self.beta_net(event_features)\n        \n        return (1 + gamma) * x + beta\n\n\n# SPATIAL ENCODING MODULE \n\nclass SpatialEncoder(nn.Module):\n    """Spatial graph encoder with multiple GAT layers"""\n    \n    def __init__(self, hidden_dim: int, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):\n        super().__init__()\n        \n        self.layers = nn.ModuleList([\n            MultiHeadGraphAttention(hidden_dim, hidden_dim, num_heads, dropout)\n            for _ in range(num_layers)\n        ])\n        \n        self.dropout = nn.Dropout(dropout)\n        \n    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:\n        for layer in self.layers:\n            residual = x \n            \n            x_new = layer(x, adj_matrix)\n            \n            x = residual + self.dropout(x_new)\n        \n        return x\n\n\n#TEMPORAL ENCODING MODULE\n\nclass TemporalEncoder(nn.Module):\n    """Temporal encoder with GRU and attention"""\n    \n    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):\n        super().__init__()\n        \n        self.hidden_dim = hidden_dim\n        \n        self.gru = nn.GRU(\n            input_dim,\n            hidden_dim,\n            num_layers=num_layers,\n            batch_first=True,\n            dropout=dropout if num_layers > 1 else 0\n        )\n        \n        self.attention = nn.MultiheadAttention(\n            hidden_dim,\n            num_heads=4,\n            dropout=dropout,\n            batch_first=True\n        )\n        \n        self.norm = nn.LayerNorm(hidden_dim)\n        self.dropout = nn.Dropout(dropout)\n        \n    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:\n        batch, T, N, F = x.shape\n        \n        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, F)\n        \n        gru_out, hidden = self.gru(x_reshaped)\n        \n        attn_out, _ = self.attention(gru_out, gru_out, gru_out)\n        attn_out = self.norm(attn_out + gru_out)\n        \n        output = attn_out[:, -1, :]\n        \n        output = output.reshape(batch, N, -1)\n        \n        return output, hidden\n\n\n#OD FLOW DECODER\n\nclass ODFlowDecoder(nn.Module):\n    """Decode node embeddings to OD flow matrix"""\n    \n    def __init__(self, hidden_dim: int, num_zones: int, output_horizons: int):\n        super().__init__()\n        self.num_zones = num_zones\n        self.output_horizons = output_horizons\n        \n        self.origin_encoder = nn.Sequential(\n            nn.Linear(hidden_dim, hidden_dim),\n            nn.ReLU(),\n            nn.Dropout(0.1),\n            nn.Linear(hidden_dim, hidden_dim // 2)\n        )\n        \n        self.dest_encoder = nn.Sequential(\n            nn.Linear(hidden_dim, hidden_dim),\n            nn.ReLU(),  \n            nn.Dropout(0.1),\n            nn.Linear(hidden_dim, hidden_dim // 2)\n        )\n        \n        self.bilinear = nn.Bilinear(hidden_dim // 2, hidden_dim // 2, 1)\n        \n        self.horizon_predictor = nn.Sequential(\n            nn.Linear(1, output_horizons),\n            nn.ReLU()\n        )\n        \n    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:\n        batch_size = node_embeddings.size(0)\n        \n        origin_features = self.origin_encoder(node_embeddings)\n        dest_features = self.dest_encoder(node_embeddings)\n        \n        origin_exp = origin_features.unsqueeze(2).expand(-1, -1, self.num_zones, -1)\n        dest_exp = dest_features.unsqueeze(1).expand(-1, self.num_zones, -1, -1)\n        \n        od_scores = self.bilinear(\n            origin_exp.reshape(-1, origin_features.size(-1)),\n            dest_exp.reshape(-1, dest_features.size(-1))\n        ).reshape(batch_size, self.num_zones, self.num_zones, 1)\n        \n        od_predictions = self.horizon_predictor(od_scores)\n        \n        od_predictions = od_predictions.permute(0, 3, 1, 2)\n        \n        return F.relu(od_predictions)\n\n\n#COMPLETE MODEL\n\nclass EASTGNNModel(nn.Module):\n    """Event-Aware Spatio-Temporal Graph Neural Network"""\n    \n    def __init__(\n        self,\n        num_zones: int,\n        event_feature_dim: int,\n        hidden_dim: int = 128,\n        num_gnn_layers: int = 3,\n        num_temporal_layers: int = 2,\n        output_horizons: int = 4,\n        dropout: float = 0.1\n    ):\n        super().__init__()\n        \n        self.num_zones = num_zones\n        self.hidden_dim = hidden_dim\n        \n        self.input_proj = nn.Linear(1, hidden_dim)\n        self.event_conditioning = EventConditioningLayer(hidden_dim, event_feature_dim)\n        \n        self.spatial_encoder = SpatialEncoder(\n            hidden_dim, num_layers=num_gnn_layers, num_heads=4, dropout=dropout\n        )\n        \n        self.temporal_encoder = TemporalEncoder(\n            hidden_dim, hidden_dim, num_layers=num_temporal_layers, dropout=dropout\n        )\n        \n        self.od_decoder = ODFlowDecoder(hidden_dim, num_zones, output_horizons)\n        \n    def forward(\n        self,\n        historical_od: torch.Tensor,\n        event_features: torch.Tensor,\n        adj_matrix: torch.Tensor\n    ) -> torch.Tensor:\n        batch, T, N, _ = historical_od.shape\n        \n        # 1. Aggregate OD to node-level features\n        origin_flows = historical_od.sum(dim=3)\n        dest_flows = historical_od.sum(dim=2)\n        node_flows = origin_flows + dest_flows\n        \n        node_features = self.input_proj(node_flows.unsqueeze(-1))  # [B, T, N, H]\n        \n        # 2. Event-Aware Spatial Encoding \n        flat_node_features = node_features.reshape(batch * T, N, self.hidden_dim)\n        flat_event_features = event_features.reshape(batch * T, N, -1)\n        \n        conditioned_features_flat = self.event_conditioning(\n            flat_node_features, flat_event_features\n        )\n        \n        spatial_features_flat = self.spatial_encoder(conditioned_features_flat, adj_matrix)\n        \n        spatial_features = spatial_features_flat.reshape(batch, T, N, self.hidden_dim)\n        \n        # 3. Temporal encoding\n        temporal_output, _ = self.temporal_encoder(spatial_features)\n        \n        # 4. Decode to OD predictions\n        predictions = self.od_decoder(temporal_output)\n        \n        return predictions\n    \n    def get_attention_weights(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:\n        pass\n\n\ndef create_adjacency_matrix(num_zones: int, k: int = 8) -> torch.Tensor:\n    """Create adjacency matrix (k-nearest neighbors demo)."""\n    try:\n        from scipy.spatial.distance import cdist\n    except ImportError:\n        print("Warning: scipy not found. Adjacency matrix will be random and dense.")\n        return torch.ones(num_zones, num_zones).float()\n    \n    coords = np.random.randn(num_zones, 2)\n    dist_matrix = cdist(coords, coords, metric=\'euclidean\')\n    \n    adj_matrix = np.zeros((num_zones, num_zones), dtype=np.float32)\n    \n    for i in range(num_zones):\n        nearest = np.argsort(dist_matrix[i])[1:k+1]\n        adj_matrix[i, nearest] = 1\n        adj_matrix[nearest, i] = 1\n    \n    adj_matrix[np.arange(num_zones), np.arange(num_zones)] = 1\n    \n    return torch.from_numpy(adj_matrix).float()\n')


# In[4]:


"""
Training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler
from torch.serialization import add_safe_globals
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import warnings
import gc
import os
import scipy.sparse

warnings.filterwarnings('ignore')

from model import EASTGNNModel, create_adjacency_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#1. DATASET 

class ODEventDataset(Dataset):
    """Dataset for OD flows with event features"""
    def __init__(self, data_path: str, time_window: int = 8, prediction_horizon: int = 4):
        super().__init__()
        self.time_window = time_window
        self.prediction_horizon = prediction_horizon

        # Load data 
        data = torch.load(data_path, map_location='cpu', weights_only=False)

        self.od_tensor = data['od_tensor']
        self.event_tensor = data['event_tensor']
        
        # Calculate number of sequential samples
        self.num_samples = len(self.od_tensor) - time_window - prediction_horizon + 1
        
        # Free up the dict immediately to help Garbage Collector
        del data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hist_end = idx + self.time_window
        pred_end = hist_end + self.prediction_horizon

        return {
            'historical_od': self.od_tensor[idx:hist_end],
            'event_features': self.event_tensor[idx:hist_end],
            'target_od': self.od_tensor[hist_end:pred_end],
        }

#2.LOSS FUNCTION

class SurgeAwareLoss(nn.Module):
    def __init__(self, surge_threshold: float = 2.0, surge_weight: float = 3.0, quantile: float = 0.9):
        super().__init__()
        self.surge_threshold = surge_threshold
        self.surge_weight = surge_weight
        self.quantile = quantile

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Base MSE
        mse_loss = F.mse_loss(predictions, targets, reduction='none')

        # 2. Surge Weighted MSE
        batch_mean = targets.mean()
        batch_std = targets.std()
        
        # Create weights tensor (In-place to save memory)
        weights = torch.ones_like(targets)
        weights[targets > (batch_mean + self.surge_threshold * batch_std)] = self.surge_weight
        
        weighted_mse = (weights * mse_loss).mean()

        # 3. MAE
        mae_loss = F.l1_loss(predictions, targets)

        # 4. Quantile Loss 
        errors = targets - predictions
        quantile_loss = torch.mean(torch.max(self.quantile * errors, (self.quantile - 1) * errors))

        # 5. Marginal Losses
        origin_loss = F.mse_loss(predictions.sum(dim=-1), targets.sum(dim=-1))
        dest_loss = F.mse_loss(predictions.sum(dim=-2), targets.sum(dim=-2))

        total_loss = (
            weighted_mse + 
            0.5 * mae_loss + 
            0.2 * quantile_loss + 
            0.1 * origin_loss + 
            0.1 * dest_loss
        )

        return {'total': total_loss, 'mae': mae_loss}

#3.TRAINER

class Trainer:
    def __init__(self, model, train_loader, val_loader, adj_matrix, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.adj_matrix = adj_matrix.to(device)
        self.config = config
        self.device = device
        
        # Detect device type for AMP
        self.device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        self.scaler = torch.amp.GradScaler('cuda') if self.device_type == 'cuda' else None

        self.criterion = SurgeAwareLoss(
            surge_threshold=config['surge_threshold'],
            surge_weight=config['surge_weight'],
            quantile=config['quantile']
        )

        self.optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.metrics_history = []
        
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch in pbar:
            historical_od = batch['historical_od'].to(self.device, non_blocking=True)
            event_features = batch['event_features'].to(self.device, non_blocking=True)
            target_od = batch['target_od'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Context
            with torch.amp.autocast(device_type=self.device_type):
                predictions = self.model(historical_od, event_features, self.adj_matrix)
                loss_dict = self.criterion(predictions, target_od)
                loss = loss_dict['total']

            # Scaled Backward Pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        return {'train_loss': total_loss / num_batches}

    @torch.inference_mode() 
    def validate(self) -> Dict[str, float]:
        """Streaming Validation (Low RAM)"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            historical_od = batch['historical_od'].to(self.device, non_blocking=True)
            event_features = batch['event_features'].to(self.device, non_blocking=True)
            target_od = batch['target_od'].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device_type):
                predictions = self.model(historical_od, event_features, self.adj_matrix)
                loss_dict = self.criterion(predictions, target_od)

            # Accumulate metrics
            batch_size = target_od.size(0)
            total_loss += loss_dict['total'].item() * batch_size
            
            diff = (predictions - target_od).detach().float()
            total_mae += torch.abs(diff).mean().item() * batch_size
            total_mse += torch.pow(diff, 2).mean().item() * batch_size
            
            total_samples += batch_size

        return {
            'val_loss': total_loss / total_samples,
            'MAE': total_mae / total_samples,
            'RMSE': np.sqrt(total_mse / total_samples)
        }

    def train(self, num_epochs: int):
        logger.info(f"STARTING TRAINING - {num_epochs} epochs | Device: {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['train_loss'])

            val_metrics = self.validate()
            self.val_losses.append(val_metrics['val_loss'])
            self.metrics_history.append({**train_metrics, **val_metrics})

            self.scheduler.step(val_metrics['val_loss'])

            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | MAE: {val_metrics['MAE']:.4f}")

            # Save best
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics, is_best=True)

            # Periodic save & Cleanup
            if epoch % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
                gc.collect() 
                if self.device_type == 'cuda':
                    torch.cuda.empty_cache()

        self.save_metrics()
        self.plot_training_curves()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        path = self.checkpoint_dir / ('best_model.pt' if is_best else f'ckpt_ep{epoch}.pt')
        torch.save(checkpoint, path)

    def save_metrics(self):
        with open(self.log_dir / 'training_metrics.json', 'w') as f:
            json.dump({'history': self.metrics_history}, f, indent=2, default=str)

    def plot_training_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(self.log_dir / 'loss_curve.png')
        plt.close()

#4. MAIN UTILS

def register_pandas_global():
    try:
        add_safe_globals({'pandas.core.indexes.datetimes._new_DatetimeIndex': 'pandas.core.indexes.datetimes._new_DatetimeIndex'})
    except:
        pass

def main():
    # 1. Configuration
    optimal_workers = 0 

    config = {
        'data_dir': './processed_data', 'time_window': 8, 'prediction_horizon': 4,
        'num_zones': 263, 'event_feature_dim': 22, 'hidden_dim': 64,
        'num_gnn_layers': 2, 'num_temporal_layers': 2, 'dropout': 0.1,
        'batch_size': 8, # Low batch size for safety
        'num_epochs': 50, 'learning_rate': 1e-3, 'weight_decay': 1e-5, 'grad_clip': 5.0,
        'surge_threshold': 2.0, 'surge_weight': 3.0, 'quantile': 0.9,
        'checkpoint_dir': './checkpoints', 'log_dir': './logs',
        'save_freq': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': optimal_workers
    }
    
    # 2. Setup
    register_pandas_global()
    device = torch.device(config['device'])
    
    # 3. Data Loading
    train_dataset = ODEventDataset(f"{config['data_dir']}/train_data.pt", config['time_window'], config['prediction_horizon'])
    val_dataset = ODEventDataset(f"{config['data_dir']}/val_data.pt", config['time_window'], config['prediction_horizon'])

    # DISABLED
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, 
        num_workers=config['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, 
        num_workers=config['num_workers'], pin_memory=True
    )

    # 4. Model & Train
    adj_matrix = create_adjacency_matrix(config['num_zones'], k=8)
    model = EASTGNNModel(
        num_zones=config['num_zones'], event_feature_dim=config['event_feature_dim'],
        hidden_dim=config['hidden_dim'], num_gnn_layers=config['num_gnn_layers'],
        num_temporal_layers=config['num_temporal_layers'], output_horizons=config['prediction_horizon'],
        dropout=config['dropout']
    )

    try:
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile() for speed.")
    except Exception as e:
        logger.warning(f"Could not compile model (safe to ignore): {e}")

    trainer = Trainer(model, train_loader, val_loader, adj_matrix, config, device)
    trainer.train(config['num_epochs'])

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium') 
    main()


# In[5]:


import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_results(log_dir='./logs'):
    """
    Loads training metrics from JSON and generates detailed plots.
    """
    json_path = Path(log_dir) / 'training_metrics.json'
    

    # 1. Load Data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract lists
    history = data.get('history', [])
    
    if 'train_losses' in data:
        train_losses = data['train_losses']
        val_losses = data['val_losses']
    else:
        # Reconstruct from history list
        train_losses = [entry['train_loss'] for entry in history]
        val_losses = [entry['val_loss'] for entry in history]

    # Extract Metrics (MAE/RMSE)
    mae_history = [entry.get('MAE', 0) for entry in history]
    rmse_history = [entry.get('RMSE', 0) for entry in history]
    epochs = range(1, len(train_losses) + 1)

    # Find Best Epoch (Min Val Loss)
    best_epoch_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_epoch_idx]
    best_epoch = best_epoch_idx + 1

    # PLOTTING 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Loss Curves
    ax1.plot(epochs, train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
    
    # Mark best epoch
    ax1.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5, label=f'Best Epoch ({best_epoch})')
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.3)
    
    ax1.set_title('Loss Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (SurgeAware)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Metrics (MAE & RMSE)
    ax2.plot(epochs, mae_history, label='MAE', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, rmse_history, label='RMSE', color='#d62728', linewidth=2, linestyle='--')
    
    ax2.set_title('Validation Error Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Error Value (Normalized Units)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"   Corresponding MAE: {mae_history[best_epoch_idx]:.4f}")

if __name__ == '__main__':
    # Run the plotter
    plot_training_results()


# In[2]:


get_ipython().system('pip install seaborn')


# In[4]:


get_ipython().system('pip install scikit-learn')


# In[ ]:




