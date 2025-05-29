from typing import List, Optional, Tuple

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from brs_algo import utils as U
from brs_algo.learning.data.collate import seq_chunk_collate_fn
from brs_algo.learning.data.dataset import ActionSeqChunkDataset


class ActionSeqChunkDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        data_path: str,
        pcd_downsample_points: int,
        pcd_x_range: Tuple[float, float],
        pcd_y_range: Tuple[float, float],
        pcd_z_range: Tuple[float, float],
        mobile_base_vel_action_min: Tuple[float, float, float],
        mobile_base_vel_action_max: Tuple[float, float, float],
        load_visual_obs_in_memory: bool = True,
        multi_view_cameras: Optional[List[str]] = None,
        load_multi_view_camera_rgb: bool = False,
        load_multi_view_camera_depth: bool = False,
        obs_window_size: int,
        action_prediction_horizon: int,
        batch_size: int,
        val_batch_size: Optional[int],
        val_split_ratio: float,
        dataloader_num_workers: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._data_path = data_path
        self._pcd_downsample_points = pcd_downsample_points
        self._pcd_x_range = pcd_x_range
        self._pcd_y_range = pcd_y_range
        self._pcd_z_range = pcd_z_range
        self._mobile_base_vel_action_min = mobile_base_vel_action_min
        self._mobile_base_vel_action_max = mobile_base_vel_action_max
        self._load_visual_obs_in_memory = load_visual_obs_in_memory
        self._multi_view_cameras = multi_view_cameras
        self._load_multi_view_camera_rgb = load_multi_view_camera_rgb
        self._load_multi_view_camera_depth = load_multi_view_camera_depth
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._dataloader_num_workers = dataloader_num_workers
        self._seed = seed
        self._val_split_ratio = val_split_ratio

        self._train_dataset, self._val_dataset = None, None
        self._obs_window_size = obs_window_size
        self._action_prediction_horizon = action_prediction_horizon

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            all_dataset = ActionSeqChunkDataset(
                fpath=self._data_path,
                pcd_downsample_points=self._pcd_downsample_points,
                pcd_x_range=self._pcd_x_range,
                pcd_y_range=self._pcd_y_range,
                pcd_z_range=self._pcd_z_range,
                mobile_base_vel_action_min=self._mobile_base_vel_action_min,
                mobile_base_vel_action_max=self._mobile_base_vel_action_max,
                load_visual_obs_in_memory=self._load_visual_obs_in_memory,
                multi_view_cameras=self._multi_view_cameras,
                load_multi_view_camera_rgb=self._load_multi_view_camera_rgb,
                load_multi_view_camera_depth=self._load_multi_view_camera_depth,
                seed=self._seed,
                action_prediction_horizon=self._action_prediction_horizon,
                obs_window_size=self._obs_window_size,
            )
            self._train_dataset, self._val_dataset = U.sequential_split_dataset(
                all_dataset,
                split_portions=[1 - self._val_split_ratio, self._val_split_ratio],
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=min(self._batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=seq_chunk_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            num_workers=min(self._val_batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=seq_chunk_collate_fn,
        )
