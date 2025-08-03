# Copyright (C) 2024 Intel Corporation
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
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


class NuScenesSensorMetadata:
    """Helper class to handle nuScenes sensor metadata"""

    def __init__(self, metadata_path=None):
        self.sensors = {}
        self.calibrated_sensors = {}
        self.ego_poses = {}
        self.samples = {}
        self.sample_data = {}
        if metadata_path:
            self.load_metadata(metadata_path)

    def load_metadata(self, metadata_path):
        """Load nuScenes metadata from JSON files"""
        try:
            # Load sensor information
            if osp.exists(osp.join(metadata_path, 'sensor.json')):
                with open(osp.join(metadata_path, 'sensor.json'), 'r') as f:
                    for sensor in json.load(f):
                        self.sensors[sensor['token']] = sensor

            # Load calibrated sensor information
            if osp.exists(osp.join(metadata_path, 'calibrated_sensor.json')):
                with open(osp.join(metadata_path, 'calibrated_sensor.json'), 'r') as f:
                    for cal_sensor in json.load(f):
                        self.calibrated_sensors[cal_sensor['token']] = cal_sensor

            # Load ego poses
            if osp.exists(osp.join(metadata_path, 'ego_pose.json')):
                with open(osp.join(metadata_path, 'ego_pose.json'), 'r') as f:
                    for ego_pose in json.load(f):
                        self.ego_poses[ego_pose['token']] = ego_pose

            # Load samples
            if osp.exists(osp.join(metadata_path, 'sample.json')):
                with open(osp.join(metadata_path, 'sample.json'), 'r') as f:
                    for sample in json.load(f):
                        self.samples[sample['token']] = sample

            # Load sample data
            if osp.exists(osp.join(metadata_path, 'sample_data.json')):
                with open(osp.join(metadata_path, 'sample_data.json'), 'r') as f:
                    for sample_data in json.load(f):
                        self.sample_data[sample_data['token']] = sample_data

        except Exception as e:
            print(f"Warning: Could not load nuScenes metadata: {e}")

    def get_sensor_metadata(self, sample_data_token):
        """Get sensor metadata for a given sample_data token"""
        if sample_data_token not in self.sample_data:
            return None

        sample_data = self.sample_data[sample_data_token]
        cal_sensor_token = sample_data.get('calibrated_sensor_token')
        ego_pose_token = sample_data.get('ego_pose_token')

        metadata = {
            'timestamp': sample_data.get('timestamp'),
            'filename': sample_data.get('filename'),
            'sensor_modality': sample_data.get('sensor_modality'),
            'channel': sample_data.get('channel'),
        }

        # Add calibrated sensor info
        if cal_sensor_token and cal_sensor_token in self.calibrated_sensors:
            cal_sensor = self.calibrated_sensors[cal_sensor_token]
            metadata['calibration'] = {
                'translation': cal_sensor.get('translation'),
                'rotation': cal_sensor.get('rotation'),
                'camera_intrinsic': cal_sensor.get('camera_intrinsic', [])
            }

            # Add sensor type info
            sensor_token = cal_sensor.get('sensor_token')
            if sensor_token and sensor_token in self.sensors:
                sensor = self.sensors[sensor_token]
                metadata['sensor'] = {
                    'channel': sensor.get('channel'),
                    'modality': sensor.get('modality')
                }

        # Add ego pose info
        if ego_pose_token and ego_pose_token in self.ego_poses:
            ego_pose = self.ego_poses[ego_pose_token]
            metadata['ego_pose'] = {
                'translation': ego_pose.get('translation'),
                'rotation': ego_pose.get('rotation'),
                'timestamp': ego_pose.get('timestamp')
            }

        return metadata


@exporter(name="nuScenes Format", ext="ZIP", version="1.0", dimension=DimensionType.DIM_3D)
def _export_nuscenes(dst_file, temp_dir, task_data, save_images=False):
    """Export annotations in nuScenes format"""
    with GetCVATDataExtractor(
        task_data,
        include_images=save_images,
        format_type="nuscenes",
        dimension=DimensionType.DIM_3D,
    ) as extractor:
        dataset = Dataset.from_extractors(extractor, env=dm_env)
        dataset.transform(RemoveTrackingInformation)

        # Create nuScenes-style directory structure
        dataset.export(temp_dir, "nuscenes", save_media=save_images, reindex=True)

        # Add metadata if available
        try:
            metadata_dir = osp.join(temp_dir, 'metadata')
            os.makedirs(metadata_dir, exist_ok=True)

            # Export sensor metadata in JSON format compatible with nuScenes
            sensor_metadata = NuScenesSensorMetadata()
            # TODO: Extract sensor metadata from CVAT task data

        except Exception as e:
            print(f"Warning: Could not export sensor metadata: {e}")

    make_zip_archive(temp_dir, dst_file)


@importer(name="nuScenes Format", ext="ZIP", version="1.0", dimension=DimensionType.DIM_3D)
def _import_nuscenes(src_file: BinaryIO, temp_dir, instance_data, load_data_callback=None, **kwargs):
    """Import nuScenes format dataset with sensor metadata"""
    if zipfile.is_zipfile(src_file):
        zipfile.ZipFile(src_file).extractall(temp_dir)

        # Look for nuScenes metadata
        metadata_path = None
        for root, dirs, files in os.walk(temp_dir):
            if 'sensor.json' in files and 'sample.json' in files:
                metadata_path = root
                break

        # Load sensor metadata if available
        sensor_metadata = NuScenesSensorMetadata(metadata_path) if metadata_path else None

        # Detect and import the dataset
        detect_dataset(
            temp_dir, format_name="nuscenes", importer=dm_env.importers.get("nuscenes")
        )
        dataset = Dataset.import_from(temp_dir, "nuscenes", env=dm_env)

        # Add sensor metadata to the dataset items if available
        if sensor_metadata:
            try:
                for item in dataset:
                    # Try to match filename to sample_data
                    for token, sample_data in sensor_metadata.sample_data.items():
                        if sample_data.get('filename', '').endswith(item.id):
                            metadata = sensor_metadata.get_sensor_metadata(token)
                            if metadata:
                                # Store metadata as item attributes
                                item.attributes.update({
                                    'sensor_metadata': json.dumps(metadata)
                                })
                            break
            except Exception as e:
                print(f"Warning: Could not attach sensor metadata: {e}")
    else:
        # Handle direct JSON metadata files
        dataset = Dataset.import_from(temp_dir, "nuscenes", env=dm_env)

    if load_data_callback is not None:
        load_data_callback(dataset, instance_data)
    import_dm_annotations(dataset, instance_data)