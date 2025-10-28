# Finetuning OlmoEarth on a classification task - Example

| # | Section |
| - | - |
| 0 | [Goal](#0-goal) |
| 1 | [Environment Setup](#1-environment-setup) |
| 2 | [Prepare the Dataset](#2-prepare-the-dataset) |
| 3 | [Define the Training Configuration](#3-define-the-training-configuration) |
| 4 | [Launch Fine-Tuning](#4-launch-fine-tuning) |
| 5 | [Run Inference With Your Fine-Tuned Model](#5-run-inference-with-your-fine-tuned-model) |


## 0. Goal
Let's build a burned area detection model using OlmoEarth. We will finetune the base model on a burned area mapping task and use it to detect fire perimeters on a unseen past fires.


## 1. Setup
We recommend installing using uv. See Installing uv for instructions to install uv. Once uv is installed:
```shell
git clone https://github.com/allenai/olmoearth_projects.git
cd olmoearth_projects
uv sync
source .venv/bin/activate
```

## 2. Gathering and prepping the dataset
Letʼs start off by using fire perimeter data from CalFire. These come as polygons associated with a contained date, which will use to determine when our satellite snapshots should be captured (within 4 weeks of the contained date).

### 2a. Data Download, filtering (year >=2020) and label creation
The following script downloads the data from CalFire and prepares it for training. It specifically:
- downloads the data from CalFire (viewer: https://experience.arcgis.com/experience/b72eede32897423683a94c61bf9d3027)
- filters out fires which happened before 2020 and create a "label" column (label value = 'burnt' for fire polygons)
- creates negative polygons (label value = 'unburnt') by drawing ring polygons around each fire perimeter, with a 150m gap to account for uncertainty
- projects all polygons to EPSG:4326

<p align="center">
  <img src="./Example%20fire%20polygons.png" alt="Description" width="200">
  <br>
  <em>Figure 1: Example of 'burnt' and 'unburnt' ring polygons</em>
</p>

First, set `DATA_DIR` to the directory where you want to store the downloaded CalFire dataset.
```shell
export DATA_DIR=/path/to/your/data  # Replace with your desired data directory
python3 ./rslp/burn-scar/scripts/calfire-data-prep.py --data-dir $DATA_DIR --gap_width 150
```

### 2b. window (aka task) geometry creation
In order to prepare the dataset that will be used for finetuning, we need to create spatiotemporal windows. A *window* roughly corresponds to a training or val/test example, and is essentially a geographic area, coupled with a time range, over which we want a model to make some prediction.
In our case, our windows should be big enough to envelop each polygon, with a minimal size that ensures the model can see a consistent pre-set input size.
The following script creates a window (or task) geometry of minimal 128x128 pixel size around each polygon:
```shell
python3 ./rslp/burn-scar/scripts/OER_taskgeom_creation.py $DATA_DIR/Calfp_2020-2025.gdb --min_box_size_pix 128
```

<p align="center">
  <img src="./Example%20windows.png" alt="Description" width="300">
  <br>
  <em>Figure 1: Example of 'burnt' and 'unburnt' windows</em>
</p>

### 2c. Creating the standardized annotation files
Olmoearth_run can fully automate the dataset ingestion and preparation pipeline, provided with a couple of standardized geojson annotation files.
The following script takes our polygon and window geometries (*Calfp_2020-2025_bbox.gdb*) and convert them into the desired olmoearth_run friendly geojson files.
We are specifying the window/task geometry column created in step 2b.
First, set `PROJECT_PATH` to the directory where you want to store the project configuration files.
```shell
export PROJECT_PATH=olmoearth_projects/projects/burn-scar-tutorial # Replace with desired project path
python ./rslp/burn-scar/scripts/OER_annotation_creation.py $DATA_DIR/Calfp_2020-2025_bbox.gdb --outdir $PROJECT_PATH --id-col polygon_id --taskgeom-col task_geom
```

### 2d. Window creation
Now that our window and polygon geometries are ready, we need to specify how olmoearth_run should interpret them and build the associated dataset windows.
This is the role of the `olmoearth_run.yaml` config file (located at `olmoearth_project/olmoearth_run_data/burn-scar-tutorial/olmoearth_run.yaml`).

For example, here's our window preparation configuration. The *polygon_to_raster_window_preparer* class rasterizes our 'burnt' and 'unburnt' polygons on the window/task footprint defined in 2b. As we intend to leverage fully the resolution of Sentinel2 data, we specify the *window_resolution* to 10m.
Additionally, we want the train/val/test sets to be spatially splitted, using a grid size of 1000 pixels (10km).

```yaml
window_prep:
  labeled_window_preparer:
    class_path: olmoearth_run.runner.tools.labeled_window_preparers.polygon_to_raster_window_preparer.PolygonToRasterWindowPreparer
    init_args:
      window_resolution: 10.0

  data_splitter:
    class_path: olmoearth_run.runner.tools.data_splitters.spatial_data_splitter.SpatialDataSplitter
    init_args:
      train_prop: 0.7
      val_prop: 0.15
      test_prop: 0.15
      grid_size: 1000
  label_layer: "label"
  label_property: "category"
  group_name: "spatial_split_10km"
  split_property: "split"
```

See the full configuration file [here](olmoearth_project/olmoearth_run_data/burn-scar-tutorial/olmoearth_run.yaml).

Let's use olmoearth_run to build these windows:

```shell
export OER_DATASET_PATH=path/to/your/oerun_dataset/folder # Replace with desired dataset folder path
python -m rslp.main olmoearth_run prepare_labeled_windows --project_path olmoearth_project/olmoearth_run_data/burn-scar-tutorial/ --scratch_path $OER_DATASET_PATH$
```

### 2e. Remote Sensing data
At this point we need to create a dataset.json file that defines our dataset schema: which layers exist, their type (raster/vector), formats, and optionally how to auto-populate them via a data_source.
In our case, we have so far created our windows and our label layer in a vector format, so we need to reflect this in our config file.
Additionally, we specify the remote sensing data we want to add to our dataset: Sentinel2.

```json
{
    "layers": {
        "label": {
            "type": "vector",
            "format": {
                "name": "geojson",
                "coordinate_mode": "pixel"
            }
        },
        "sentinel2_l2a": {
            "type": "raster",
            "band_sets": [
                {
                "bands": [
                    "B02",
                    "B03",
                    "B04",
                    "B08"
                ],
                "dtype": "uint16"
                },
                {
                "bands": [
                    "B05",
                    "B06",
                    "B07",
                    "B8A",
                    "B11",
                    "B12"
                ],
                "dtype": "uint16",
                "zoom_offset": -1
                },
                {
                "bands": [
                    "B01",
                    "B09"
                ],
                "dtype": "uint16",
                "zoom_offset": -2
                }
            ],
            "data_source": {
                "cache_dir": "cache/planetary_computer",
                "duration": "45d",
                "harmonize": true,
                "ingest": false,
                "query": { "eo:cloud_cover": { "lt": 50 }},
                "name": "rslearn.data_sources.planetary_computer.Sentinel2",
                "sort_by": "eo:cloud_cover"
            }

        }
    }
}
```

Let's launch the Sentinel2 data fetching and stitching to match our windows:

```shell
python -m rslp.main olmoearth_run build_dataset_from_windows --project_path olmoearth_project/olmoearth_run_data/burn-scar-tutorial/ --scratch_path $OER_DATASET_PATH$
```


## 3. Defining and training the model (DRAFT)

First let's download the OlmoEarth pretrained model weights. Depending on the complexity of the task, your finetuning budget, and your GPU memory, you can select:
  - OlmoEarth nano: {X} parameters - {HF link}:
  - OlmoEarth tiny: {X} parameters, - {HF link}:
  - OlmoEarth base: {X} parameters, - {HF link}:
  - OlmoEarth large: {X} parameters, - {HF link}:


Now we need to design our model architecture, training loop and evaluation metrics, and define how the data should be pre-processed and sent to the model.
Behind the scenes, we use Lightning to coordinate and run the fine-tuning job. This allows us to configure every aspect of the job in a single configuration file.

Let's create it:
`burn_scar_model.yaml`:

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    # This part defines the model architecture.
    # Essentially we apply the OlmoEarth backbone with a classification head
    model:
      class_path: rslearn.models.multitask.MultiTaskModel
      init_args:
        encoder:
          - class_path: rslp.helios.model.Helios
            init_args:
              checkpoint_path: "{CHECKPOINT_PATH}"
              selector: ["encoder"]
              forward_kwargs:
                patch_size: {PATCH_SIZE}
        decoders:
          burn-scar_classification:
            - class_path: rslearn.models.pooling_decoder.PoolingDecoder
              init_args:
                in_channels: {ENCODER_EMBEDDING_SIZE}
                out_channels: 2
            - class_path: rslearn.train.tasks.classification.ClassificationHead
    lr: 0.0001
    scheduler:
      class_path: rslearn.train.scheduler.PlateauScheduler
      init_args:
        factor: 0.2
        patience: 2
        min_lr: 0
        cooldown: 10
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    # Replace this with the dataset path.
    path: /weka/dfive-default/rslearn-eai/datasets/burn-scar/California/2020-25
    # This defines the layers that should be read for each window.
    # The key ("image" / "targets") is what the data will be called in the model,
    # while the layers option specifies which layers will be read.
    inputs:
      sentinel2_l2a:
        data_type: "raster"
        layers: ["sentinel2_l2a"]
        bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        passthrough: true #if false only what the Task returns via process_inputs is sent to the model. If true the raw input (here is the raster) is added to the Task output, which is then accessible for transforms or visualization
        dtype: FLOAT32
      label:
        data_type: "vector"
        layers: ["label"]
        is_target: true
    task:
      class_path: rslearn.train.tasks.multi_task.MultiTask
      init_args:
        tasks:
          burn-scar_classification:
            class_path: rslearn.train.tasks.classification.ClassificationTask
            init_args:
              property_name: "category"
              classes: ["burnt", "unburnt"]
              enable_f1_metric: true
              metric_kwargs:
                average: "micro"
        input_mapping:
          burn-scar_classification:
            label: "targets"
    batch_size: 32
    num_workers: 32
    # These define different options for different phases/splits, like training,
    # validation, and testing.
    # Here we use the same transform across splits except training where we add a
    # flipping augmentation.
    # For now we are using the same windows for training and validation.
    default_config:
      transforms:
        - class_path: rslearn.train.transforms.concatenate.Concatenate
          init_args:
            selections:
              sentinel2_l2a: []
            output_selector: sentinel2_l2a
        - class_path: rslp.helios.norm.HeliosNormalize
          init_args:
            config_fname: "/opt/helios/data/norm_configs/computed.json"
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        - class_path: rslearn.train.transforms.pad.Pad
          init_args:
            size: 8
            mode: "center"
            image_selectors: ["sentinel2_l2a"]
    train_config:
      groups: ["groundtruth_polygon_split_window_32"]
      tags:
        split: "train"
    val_config:
      groups: ["groundtruth_polygon_split_window_32"]
      tags:
        split: "val"
    test_config:
      groups: ["groundtruth_polygon_split_window_32"]
      tags:
        split: "val"
trainer:
  max_epochs: 100
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: "epoch"
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 1
        save_last: true
        monitor: val_loss
        mode: min
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 2
rslp_project: placeholder
rslp_experiment: placeholder
```

Now we can train the model:

### 3.  Finetune model
```shell
export GROUP_PATH=/weka/dfive-default/rslearn-eai/datasets/burn-scar/oerun/dataset/windows/post_random_split
find $GROUP_PATH -maxdepth 2 -name "metadata.json" -exec cat {} \; | grep -oE "train|val|test" | sort | uniq -c | awk 'BEGIN{printf "{"} {printf "%s\"%s\": %d", (NR>1?", ":""), $2, $1} END{print "}"}'
python -m rslp.main olmoearth_run finetune --project_path /weka/dfive-default/yawenz/rslearn_projects/olmoearth_run_data/nandi/finetune --scratch_path /weka/dfive-default/yawenz/datasets/scratch_ft_v3
```

## 4. Inference: using your fine-tuned model


> ---
> **TODO**
> - Inference pipeline using olmoearth_run
> ---
