# Copyright (C) 2021-2022 Intel Corporation
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import os
import os.path as osp
import zipfile
from pathlib import Path
from typing import BinaryIO

from datumaro.components.dataset import Dataset
from datumaro.components.transformer import ItemTransform

from cvat.apps.dataset_manager.bindings import (
    GetCVATDataExtractor,
    detect_dataset,
    import_dm_annotations,
)
from cvat.apps.dataset_manager.util import make_zip_archive
from cvat.apps.engine.models import DimensionType

from .registry import dm_env, exporter, importer


class RemoveTrackingInformation(ItemTransform):
    def transform_item(self, item):
        annotations = list(item.annotations)
        for anno in annotations:
            if hasattr(anno, "attributes") and "track_id" in anno.attributes:
                del anno.attributes["track_id"]
        return item.wrap(annotations=annotations)


class KittiSensorMetadata:
    """Helper class to handle KITTI sensor metadata and calibration data"""

    def __init__(self, calibration_path=None):
        self.calib_cam_to_cam = {}
        self.calib_imu_to_velo = {}
        self.calib_velo_to_cam = {}
        self.timestamps = {}
        if calibration_path:
            self.load_calibration_data(calibration_path)

    def load_calibration_data(self, calibration_path):
        """Load KITTI calibration data from files"""
        try:
            # Load camera-to-camera calibration
            cam_to_cam_file = osp.join(calibration_path, 'calib_cam_to_cam.txt')
            if osp.exists(cam_to_cam_file):
                self.calib_cam_to_cam = self._parse_calib_file(cam_to_cam_file)

            # Load IMU-to-Velodyne calibration
            imu_to_velo_file = osp.join(calibration_path, 'calib_imu_to_velo.txt')
            if osp.exists(imu_to_velo_file):
                self.calib_imu_to_velo = self._parse_calib_file(imu_to_velo_file)

            # Load Velodyne-to-camera calibration
            velo_to_cam_file = osp.join(calibration_path, 'calib_velo_to_cam.txt')
            if osp.exists(velo_to_cam_file):
                self.calib_velo_to_cam = self._parse_calib_file(velo_to_cam_file)

            # Load timestamps if available
            timestamps_file = osp.join(calibration_path, 'timestamps.txt')
            if osp.exists(timestamps_file):
                with open(timestamps_file, 'r') as f:
                    self.timestamps = [line.strip() for line in f.readlines()]

        except Exception as e:
            print(f"Warning: Could not load KITTI calibration data: {e}")

    def _parse_calib_file(self, filepath):
        """Parse KITTI calibration file format"""
        calib_data = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, values = line.split(':', 1)
                        key = key.strip()
                        # Parse numerical values
                        try:
                            values = [float(x) for x in values.strip().split()]
                            calib_data[key] = values
                        except ValueError:
                            # Handle non-numerical values
                            calib_data[key] = values.strip()
        except Exception as e:
            print(f"Warning: Could not parse calibration file {filepath}: {e}")
        return calib_data

    def get_sensor_metadata(self, frame_id=None, sensor_type='velodyne'):
        """Get sensor metadata for a given frame and sensor type"""
        metadata = {
            'sensor_type': sensor_type,
            'frame_id': frame_id,
        }

        # Add timestamp if available
        if self.timestamps and frame_id is not None and frame_id < len(self.timestamps):
            metadata['timestamp'] = self.timestamps[frame_id]

        # Add calibration data based on sensor type
        if sensor_type == 'velodyne' or sensor_type == 'lidar':
            metadata['calibration'] = {
                'velo_to_cam': self.calib_velo_to_cam,
                'imu_to_velo': self.calib_imu_to_velo
            }
        elif sensor_type.startswith('camera') or sensor_type.startswith('image'):
            metadata['calibration'] = {
                'cam_to_cam': self.calib_cam_to_cam,
                'velo_to_cam': self.calib_velo_to_cam
            }

        return metadata

    def export_calibration_data(self, output_dir):
        """Export calibration data to KITTI format files"""
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Export camera-to-camera calibration
            if self.calib_cam_to_cam:
                with open(osp.join(output_dir, 'calib_cam_to_cam.txt'), 'w') as f:
                    for key, values in self.calib_cam_to_cam.items():
                        if isinstance(values, list):
                            f.write(f"{key}: {' '.join(map(str, values))}\n")
                        else:
                            f.write(f"{key}: {values}\n")

            # Export IMU-to-Velodyne calibration
            if self.calib_imu_to_velo:
                with open(osp.join(output_dir, 'calib_imu_to_velo.txt'), 'w') as f:
                    for key, values in self.calib_imu_to_velo.items():
                        if isinstance(values, list):
                            f.write(f"{key}: {' '.join(map(str, values))}\n")
                        else:
                            f.write(f"{key}: {values}\n")

            # Export Velodyne-to-camera calibration
            if self.calib_velo_to_cam:
                with open(osp.join(output_dir, 'calib_velo_to_cam.txt'), 'w') as f:
                    for key, values in self.calib_velo_to_cam.items():
                        if isinstance(values, list):
                            f.write(f"{key}: {' '.join(map(str, values))}\n")
                        else:
                            f.write(f"{key}: {values}\n")

            # Export timestamps
            if self.timestamps:
                with open(osp.join(output_dir, 'timestamps.txt'), 'w') as f:
                    for timestamp in self.timestamps:
                        f.write(f"{timestamp}\n")

        except Exception as e:
            print(f"Warning: Could not export calibration data: {e}")


@exporter(name="Kitti Raw Format", ext="ZIP", version="1.0", dimension=DimensionType.DIM_3D)
def _export_images(dst_file, temp_dir, task_data, save_images=False):
    with GetCVATDataExtractor(
        task_data,
        include_images=save_images,
        format_type="kitti_raw",
        dimension=DimensionType.DIM_3D,
    ) as extractor:
        dataset = Dataset.from_extractors(extractor, env=dm_env)
        dataset.transform(RemoveTrackingInformation)
        dataset.export(temp_dir, "kitti_raw", save_media=save_images, reindex=True)

        # Export sensor metadata and calibration data if available
        try:
            calibration_dir = osp.join(temp_dir, 'calibration')
            sensor_metadata = KittiSensorMetadata()

            # TODO: Extract calibration data from CVAT task metadata
            # For now, create placeholder calibration files
            sensor_metadata.export_calibration_data(calibration_dir)

        except Exception as e:
            print(f"Warning: Could not export sensor metadata: {e}")

    make_zip_archive(temp_dir, dst_file)


@importer(name="Kitti Raw Format", ext="ZIP, XML", version="1.0", dimension=DimensionType.DIM_3D)
def _import(src_file: BinaryIO, temp_dir, instance_data, load_data_callback=None, **kwargs):
    if zipfile.is_zipfile(src_file):
        zipfile.ZipFile(src_file).extractall(temp_dir)

        # Look for calibration data
        calibration_path = None
        for root, dirs, files in os.walk(temp_dir):
            if any(f.startswith('calib_') for f in files):
                calibration_path = root
                break

        # Load sensor metadata if available
        sensor_metadata = KittiSensorMetadata(calibration_path) if calibration_path else None

        detect_dataset(
            temp_dir, format_name="kitti_raw", importer=dm_env.importers.get("kitti_raw")
        )
        dataset = Dataset.import_from(temp_dir, "kitti_raw", env=dm_env)

        # Add sensor metadata to the dataset items if available
        if sensor_metadata:
            try:
                for item in dataset:
                    # Extract frame ID from filename (assuming format like 000000.bin, 000000.png)
                    frame_id = None
                    try:
                        frame_id = int(osp.splitext(osp.basename(item.id))[0])
                    except ValueError:
                        pass

                    # Determine sensor type from file extension
                    ext = osp.splitext(item.id)[1].lower()
                    if ext in ['.bin', '.pcd']:
                        sensor_type = 'velodyne'
                    elif ext in ['.png', '.jpg', '.jpeg']:
                        sensor_type = 'camera'
                    else:
                        sensor_type = 'unknown'

                    metadata = sensor_metadata.get_sensor_metadata(frame_id, sensor_type)
                    if metadata:
                        # Store metadata as item attributes
                        item.attributes.update({
                            'sensor_metadata': json.dumps(metadata)
                        })

            except Exception as e:
                print(f"Warning: Could not attach sensor metadata: {e}")
    else:
        tmp_src_file_link = Path(temp_dir) / "tracklet_labels.xml"
        tmp_src_file_link.symlink_to(src_file.name)
        dataset = Dataset.import_from(str(tmp_src_file_link.absolute()), "kitti_raw", env=dm_env)

    if load_data_callback is not None:
        load_data_callback(dataset, instance_data)
    import_dm_annotations(dataset, instance_data)
