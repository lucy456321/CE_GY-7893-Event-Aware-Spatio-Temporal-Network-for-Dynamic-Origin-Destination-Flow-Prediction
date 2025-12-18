
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

    def load_events_data(self)  pd.DataFrame:
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

    def create_time_bins(self, start_time: datetime, end_time: datetime)  pd.DatetimeIndex:
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

                # Feature 1: Event count 
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
