# Label Quality

`olmoearth_projects` demonstrates how OlmoEarth can be applied to downstream applications.
Specifically, given a set of labels, `olmoearth_projects` demonstrates how to finetune, evaluate and apply OlmoEarth over a spatial area.

The quality of the model's predictions depend on the quality of the labels.
Assessing the quality of the labels is best done by domain experts.
However, the functions in this folder also provide some indication of how well suited a set of labels are for mapping.

#### Spatial Clustering

This function assesses how spatially clustered classes are.
In general, we'd like different classes to be well spatially distributed:

```
xoxoxox
oxoxoxo
xoxoxox
```
is more desirable than
```
xxx
xxx
   ooo
   ooo
```
We measure this by running a spatial KNN on the dataset - for each instance in the dataset, we define its class
to be the mode of the K nearest (spatial) points. High accuracies indicate high spatial clustering.

```python
import geopandas as gpd
import pandas as pd

from olmoearth_projects.utils.label_quality.spatial_clustering import spatial_clustering

df = pd.DataFrame(
    {
        "City": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas"],
        "Country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela"],
        # highly clustered labels
        "label": [0, 0, 0, 1, 1],
        "Latitude": [-34.58, -15.78, -33.45, 4.60, 10.48],
        "Longitude": [-58.66, -47.91, -70.66, -74.08, -66.86],
    }
)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
)
assert spatial_clustering(gdf[["label", "geometry"]], k=1) == 1
```

### Spatial extent

This function assesses how much of the total labelled area each class occupies.
In general, we would like each class to occupy a large fraction of the total labelled area:

```
x xox x
ox x xo
x xox x
```
is more desirable then
```
x x x x
 x xoxo
x xoxox
```

### Examples

An example of how to run this is on an rslearn dataset is in [the `mozambique_lulc` project](../../projects/mozambique_lulc/check_label_quality.py):

```console
$ python olmoearth_projects/projects/mozambique_lulc/check_label_quality.py --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202 --split train

Checking label quality for 3821 instances.
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Check name ┃ Metric     ┃                 Value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│    label_imbalance │ beans      │     0.260664747448312 │
│    label_imbalance │ corn       │   0.24522376341271918 │
│    label_imbalance │ cassava    │   0.18503009683328972 │
│    label_imbalance │ sesame     │   0.10049725202826486 │
│    label_imbalance │ rice       │   0.18555352002093692 │
│    label_imbalance │ sorghum    │  0.013609002878827532 │
│    label_imbalance │ millet     │   0.00942161737764983 │
│ spatial_clustering │ beans_f1   │    0.9885401096163426 │
│ spatial_clustering │ corn_f1    │    0.9946581196581196 │
│ spatial_clustering │ cassava_f1 │    0.9728571428571429 │
│ spatial_clustering │ sesame_f1  │                   1.0 │
│ spatial_clustering │ rice_f1    │    0.9943661971830987 │
│ spatial_clustering │ sorghum_f1 │    0.9902912621359223 │
│ spatial_clustering │ millet_f1  │                   1.0 │
│     spatial_extent │ beans      │    0.7829124125842517 │
│     spatial_extent │ corn       │    0.8357589381957512 │
│     spatial_extent │ cassava    │    0.9383923435655623 │
│     spatial_extent │ sesame     │ 0.0003614488654102921 │
│     spatial_extent │ rice       │    0.7653946614854196 │
│     spatial_extent │ sorghum    │ 3.266172530759744e-07 │
│     spatial_extent │ millet     │ 0.0001509740414169792 │
└────────────────────┴────────────┴───────────────────────┘
```
