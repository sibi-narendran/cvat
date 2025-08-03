import json
import os
import tempfile
import zipfile
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from cvat.apps.dataset_manager.formats.nuscenes import NuScenesSensorMetadata
from cvat.apps.dataset_manager.formats.velodynepoint import KittiSensorMetadata
from cvat.apps.engine.media_extractors import SensorMetadataExtractor
from cvat.apps.engine.models import SensorMetadata, EgoPose

from .utils import ApiTestBase, FormatTestBase


class TestSensorMetadata(ApiTestBase):
    def setUp(self):
        super().setUp()

    def _create_mock_nuscenes_data(self):
        """Create mock nuScenes metadata structure"""
        metadata = {
            'sensor.json': [
                {
                    'token': 'sensor_1',
                    'channel': 'CAM_FRONT',
                    'modality': 'camera'
                },
                {
                    'token': 'sensor_2',
                    'channel': 'LIDAR_TOP',
                    'modality': 'lidar'
                }
            ],
            'calibrated_sensor.json': [
                {
                    'token': 'cal_sensor_1',
                    'sensor_token': 'sensor_1',
                    'translation': [1.5, 0.0, 1.8],
                    'rotation': [1.0, 0.0, 0.0, 0.0],
                    'camera_intrinsic': [
                        [1266.4, 0.0, 816.3],
                        [0.0, 1266.4, 491.5],
                        [0.0, 0.0, 1.0]
                    ]
                },
                {
                    'token': 'cal_sensor_2',
                    'sensor_token': 'sensor_2',
                    'translation': [0.0, 0.0, 1.8],
                    'rotation': [1.0, 0.0, 0.0, 0.0],
                    'camera_intrinsic': []
                }
            ],
            'ego_pose.json': [
                {
                    'token': 'ego_1',
                    'translation': [1000.0, 2000.0, 0.0],
                    'rotation': [1.0, 0.0, 0.0, 0.0],
                    'timestamp': 1532402927647951
                }
            ],
            'sample_data.json': [
                {
                    'token': 'sample_data_1',
                    'filename': 'samples/CAM_FRONT/image_01.jpg',
                    'calibrated_sensor_token': 'cal_sensor_1',
                    'ego_pose_token': 'ego_1',
                    'timestamp': 1532402927647951,
                    'fileformat': 'jpg',
                    'width': 1600,
                    'height': 900,
                    'sensor_modality': 'camera',
                    'channel': 'CAM_FRONT'
                },
                {
                    'token': 'sample_data_2',
                    'filename': 'samples/LIDAR_TOP/lidar_01.pcd',
                    'calibrated_sensor_token': 'cal_sensor_2',
                    'ego_pose_token': 'ego_1',
                    'timestamp': 1532402927647951,
                    'fileformat': 'pcd.bin',
                    'sensor_modality': 'lidar',
                    'channel': 'LIDAR_TOP'
                }
            ]
        }
        return metadata

    def _create_mock_kitti_data(self):
        """Create mock KITTI calibration data"""
        calib_data = {
            'calib_cam_to_cam.txt': [
                'calib_time: 09-Jan-2012 13:57:47',
                'corner_dist: 9.950000e-02',
                'S_00: 1.392000e+03 5.120000e+02',
                'K_00: 9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00',
                'D_00: -3.691481e-01 1.968681e-01 1.353473e-03 5.677587e-04 -6.770705e-02',
                'R_00: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00',
                'T_00: 0.000000e+00 0.000000e+00 0.000000e+00',
                'P_00: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00',
            ],
            'calib_velo_to_cam.txt': [
                'calib_time: 15-Mar-2012 17:57:40',
                'R: 7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01 9.998621e-01 7.523790e-03 1.480755e-02',
                'T: -4.069766e-03 -7.631618e-02 -2.717806e-01',
                'delta_f: 0.000000e+00 0.000000e+00',
                'delta_c: 0.000000e+00 0.000000e+00'
            ],
            'timestamps.txt': [
                '2011-09-26 13:02:44.464178848',
                '2011-09-26 13:02:44.564047104',
                '2011-09-26 13:02:44.663975360'
            ]
        }
        return calib_data

    def test_nuscenes_metadata_extraction(self):
        """Test nuScenes metadata extraction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock nuScenes data
            metadata = self._create_mock_nuscenes_data()

            # Write metadata files
            for filename, data in metadata.items():
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    json.dump(data, f)

            # Test metadata extraction
            sensor_metadata = NuScenesSensorMetadata(temp_dir)

            # Verify sensors were loaded
            self.assertIn('sensor_1', sensor_metadata.sensors)
            self.assertIn('sensor_2', sensor_metadata.sensors)

            # Test metadata retrieval
            cam_metadata = sensor_metadata.get_sensor_metadata('sample_data_1')
            self.assertIsNotNone(cam_metadata)
            self.assertIn('calibration', cam_metadata)
            self.assertIn('ego_pose', cam_metadata)

    def test_kitti_metadata_extraction(self):
        """Test KITTI metadata extraction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock KITTI calibration files
            calib_data = self._create_mock_kitti_data()

            for filename, lines in calib_data.items():
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write('\n'.join(lines))

            # Test metadata extraction
            sensor_metadata = KittiSensorMetadata(temp_dir)

            # Verify calibration data was loaded
            self.assertIn('calib_cam_to_cam', sensor_metadata.__dict__)
            self.assertIn('calib_velo_to_cam', sensor_metadata.__dict__)

            # Test metadata retrieval
            cam_metadata = sensor_metadata.get_sensor_metadata(0, 'camera')
            self.assertIsNotNone(cam_metadata)
            self.assertEqual(cam_metadata['sensor_type'], 'camera')

    def test_sensor_metadata_extractor(self):
        """Test the SensorMetadataExtractor class"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            os.makedirs(os.path.join(temp_dir, 'images'))
            os.makedirs(os.path.join(temp_dir, 'lidar'))

            # Create dummy image file
            image = Image.new('RGB', (100, 100), color='red')
            image.save(os.path.join(temp_dir, 'images', 'test.jpg'))

            # Create dummy lidar file
            with open(os.path.join(temp_dir, 'lidar', 'test.bin'), 'wb') as f:
                f.write(b'dummy lidar data')

            # Test generic metadata extraction
            extractor = SensorMetadataExtractor(temp_dir)
            result = extractor.extract_metadata()

            self.assertTrue(result)
            self.assertGreater(len(extractor.sensor_metadata), 0)

    def test_api_sensor_metadata_endpoint(self):
        """Test the API endpoint for sensor metadata"""
        # Create a task with some data
        task_spec = {
            "name": "test sensor metadata task",
            "labels": [
                {"name": "car", "color": "#ff0000", "attributes": []}
            ]
        }

        response = self._create_task(task_spec)
        task_id = response.json()['id']

        # Test getting sensor metadata (should be empty initially)
        response = self.client.get(f'/api/tasks/{task_id}/sensor-metadata')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 0)

    def test_api_ego_poses_endpoint(self):
        """Test the API endpoint for ego poses"""
        # Create a task with some data
        task_spec = {
            "name": "test ego poses task",
            "labels": [
                {"name": "car", "color": "#ff0000", "attributes": []}
            ]
        }

        response = self._create_task(task_spec)
        task_id = response.json()['id']

        # Test getting ego poses (should be empty initially)
        response = self.client.get(f'/api/tasks/{task_id}/ego-poses')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 0)

        # Test with frame_id parameter
        response = self.client.get(f'/api/tasks/{task_id}/ego-poses?frame_id=0')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 0)

        # Test with invalid frame_id
        response = self.client.get(f'/api/tasks/{task_id}/ego-poses?frame_id=invalid')
        self.assertEqual(response.status_code, 400)

    def _create_task(self, task_spec):
        """Helper method to create a task"""
        return self.client.post('/api/tasks', data=task_spec, format='json')