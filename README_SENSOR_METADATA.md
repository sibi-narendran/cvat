# CVAT Sensor Metadata Support

This document describes the enhanced sensor metadata support in CVAT for 3D LiDAR and point cloud annotation workflows.

## Overview

CVAT now supports loading and managing sensor metadata for autonomous driving datasets, including:

- **nuScenes dataset format** with full sensor calibration and ego pose data
- **Enhanced KITTI support** with calibration metadata
- **Sensor metadata API** for accessing calibration and pose information
- **3D annotation workflows** with sensor context

## Supported Dataset Formats

### nuScenes Format

CVAT can now import nuScenes datasets with full sensor metadata including:

- **Sensor definitions** (cameras, LiDAR, radar)
- **Calibrated sensor data** (intrinsic/extrinsic parameters)
- **Ego vehicle poses** (global positioning and orientation)
- **Temporal synchronization** (timestamps and frame alignment)

#### nuScenes Dataset Structure
```
nuscenes_dataset.zip
├── metadata/
│   ├── sensor.json
│   ├── calibrated_sensor.json
│   ├── ego_pose.json
│   ├── sample.json
│   └── sample_data.json
├── samples/
│   ├── CAM_FRONT/
│   ├── CAM_BACK/
│   ├── LIDAR_TOP/
│   └── RADAR_FRONT/
└── sweeps/
    ├── CAM_FRONT/
    ├── LIDAR_TOP/
    └── RADAR_FRONT/
```

### Enhanced KITTI Format

The KITTI loader now supports sensor calibration data:

- **Camera calibration** (intrinsic/extrinsic parameters)
- **LiDAR-to-camera transforms**
- **IMU-to-Velodyne transforms**
- **Temporal information** (timestamps)

#### KITTI Dataset Structure
```
kitti_dataset.zip
├── calibration/
│   ├── calib_cam_to_cam.txt
│   ├── calib_velo_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── timestamps.txt
├── velodyne_points/
│   └── data/
├── image_02/
│   └── data/
└── oxts/
    └── data/
```

## Database Schema

### SensorMetadata Model

Stores sensor calibration and configuration data:

```python
class SensorMetadata(models.Model):
    data = models.ForeignKey(Data, on_delete=models.CASCADE)
    sensor_name = models.CharField(max_length=100)
    sensor_type = models.CharField(max_length=50)  # camera, lidar, radar, imu, gps
    modality = models.CharField(max_length=50)     # rgb, depth, lidar, etc.

    # Calibration matrices
    intrinsic_matrix = models.JSONField(null=True, blank=True)
    extrinsic_matrix = models.JSONField(null=True, blank=True)
    distortion_coefficients = models.JSONField(null=True, blank=True)

    # Sensor properties
    resolution = models.JSONField(null=True, blank=True)
    field_of_view = models.JSONField(null=True, blank=True)
    frequency = models.FloatField(null=True, blank=True)
    timestamp_offset = models.FloatField(default=0.0)

    additional_metadata = models.JSONField(null=True, blank=True)
```

### EgoPose Model

Stores ego vehicle poses for 3D data:

```python
class EgoPose(models.Model):
    data = models.ForeignKey(Data, on_delete=models.CASCADE)
    frame_id = models.PositiveIntegerField()

    # Pose in global coordinates
    translation = models.JSONField()  # [x, y, z]
    rotation = models.JSONField()     # [w, x, y, z] quaternion
    timestamp = models.BigIntegerField(null=True, blank=True)

    coordinate_system = models.CharField(max_length=50, default='global')
    pose_source = models.CharField(max_length=50)  # gps_imu, slam, etc.
    pose_confidence = models.FloatField(null=True, blank=True)
```

## API Endpoints

### Sensor Metadata API

Get sensor metadata for a task:
```
GET /api/tasks/{task_id}/sensor-metadata
```

Response:
```json
[
  {
    "id": 1,
    "sensor_name": "CAM_FRONT",
    "sensor_type": "camera",
    "modality": "rgb",
    "intrinsic_matrix": [
      [1266.4, 0.0, 816.3],
      [0.0, 1266.4, 491.5],
      [0.0, 0.0, 1.0]
    ],
    "extrinsic_matrix": {
      "translation": [1.5, 0.0, 1.8],
      "rotation": [1.0, 0.0, 0.0, 0.0]
    },
    "resolution": [1600, 900],
    "frequency": 12.0,
    "additional_metadata": {
      "channel": "CAM_FRONT",
      "fileformat": "jpg"
    }
  }
]
```

### Ego Poses API

Get ego poses for a task:
```
GET /api/tasks/{task_id}/ego-poses
GET /api/tasks/{task_id}/ego-poses?frame_id=0
```

Response:
```json
[
  {
    "id": 1,
    "frame_id": 0,
    "translation": [1000.0, 2000.0, 0.0],
    "rotation": [1.0, 0.0, 0.0, 0.0],
    "timestamp": 1532402927647951,
    "coordinate_system": "global",
    "pose_source": "gps_imu",
    "pose_confidence": 0.95
  }
]
```

## Usage Examples

### Importing nuScenes Dataset

1. **Prepare dataset**: Ensure your nuScenes dataset includes metadata JSON files
2. **Create task**: Upload the dataset ZIP file to CVAT
3. **Select format**: Choose "nuScenes Format" from the import dialog
4. **Access metadata**: Use the API endpoints to retrieve sensor calibration and ego poses

### Importing KITTI with Calibration

1. **Prepare dataset**: Include calibration files in a `calibration/` directory
2. **Create task**: Upload the dataset ZIP file to CVAT
3. **Select format**: Choose "Kitti Raw Format" from the import dialog
4. **Access metadata**: Calibration data will be automatically parsed and stored

### Using Sensor Metadata in Annotations

The sensor metadata can be used for:

- **Coordinate transformations** between sensor frames
- **3D projection** of annotations onto different camera views
- **Temporal alignment** of multi-sensor data
- **Quality assessment** using pose confidence scores

## Code Examples

### Accessing Sensor Metadata Programmatically

```python
from cvat.apps.engine.models import SensorMetadata, EgoPose

# Get all sensors for a task
task_id = 123
sensors = SensorMetadata.objects.filter(data__task_id=task_id)

for sensor in sensors:
    print(f"Sensor: {sensor.sensor_name} ({sensor.sensor_type})")
    if sensor.intrinsic_matrix:
        print(f"Intrinsics: {sensor.intrinsic_matrix}")

# Get ego poses for specific frames
ego_poses = EgoPose.objects.filter(
    data__task_id=task_id,
    frame_id__in=[0, 1, 2]
).order_by('frame_id')

for pose in ego_poses:
    print(f"Frame {pose.frame_id}: {pose.translation}")
```

### Using Metadata Extractors

```python
from cvat.apps.engine.media_extractors import SensorMetadataExtractor

# Extract metadata from dataset
extractor = SensorMetadataExtractor('/path/to/dataset', 'nuscenes')
if extractor.extract_metadata():
    print(f"Found {len(extractor.sensor_metadata)} sensors")
    print(f"Found {len(extractor.ego_poses)} ego poses")
```

## Migration

To enable sensor metadata support in an existing CVAT installation:

1. **Run migration**:
   ```bash
   python manage.py migrate
   ```

2. **Update data loaders**: The new format loaders are automatically registered

3. **Import datasets**: Use the new import formats for datasets with metadata

## Limitations and Future Work

### Current Limitations

- Metadata is read-only through the API
- Limited to supported dataset formats (nuScenes, KITTI)
- No visualization of sensor poses in the 3D interface

### Planned Enhancements

- **Interactive metadata editing** through the UI
- **Visual sensor pose display** in 3D workspace
- **Additional format support** (Waymo, Argoverse)
- **Automatic coordinate frame transformations** in annotations
- **Temporal interpolation** of ego poses

## Contributing

To extend sensor metadata support:

1. **Add new format loaders** in `cvat/apps/dataset_manager/formats/`
2. **Extend metadata extractors** in `cvat/apps/engine/media_extractors.py`
3. **Update API serializers** in `cvat/apps/engine/serializers.py`
4. **Add tests** in `tests/python/rest_api/test_sensor_metadata.py`

## References

- [nuScenes Dataset](https://www.nuscenes.org/)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [CVAT 3D Annotation Documentation](https://docs.cvat.ai/docs/manual/basics/3d-object-annotation/)