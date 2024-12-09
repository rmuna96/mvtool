# Mitral Valve (MV) tool

## Extract morphological metrics from multi-class segmentation of mitral valve from 3D ultrasound.

This repository is designed to validate the performance of deep learning models 
for automatic mitral valve segmentation from 3D ultrasound data. It provides 
tools to automatically measure clinically relevant metrics that are commonly used 
to characterize mitral valve morphology during the end-systole (ES) phase. 
These metrics are crucial in clinical settings for assessing the anatomy and 
function of the mitral valve.

## Requirements

Refer to `requirements.txt` for the list of required Python packages.

## Usage

### Compute morphological metrics from multi-class (anterior and posterior leaflet) MV segmentation

Run the following command:\
`$ python metricsfromnifti.py [OPTIONS] IDIR ODIR`  
For more details, use:\
`$ python metricsfromnifti.py -h`.

Alternatively,    
`$ python metricsfromh5.py [OPTIONS] IDIR ODIR`\
for NTNU collaboation.

This tool computes morphological metrics based on the provided segmentation mask (target or prediction). 
The anatomical landmarks and metric definitions are based on  [[1]](#1). The metrics include:
* Annulus
  - antero-posterior (AP) distance [mm]
  - anterolateral-posteriormedial (AL-PM) distance [mm]
  - inter-trigonal (IT) distance [mm]
  - commisural width (CW) [mm]
  - annular height (AH) [mm]
  - annular perimeter [mm]
  - sphericity index (SI)
  - non-planar angle (NPA) [°]
  - annular area 2D [mm<sup>2</sup>]
  - annular area 3D [mm<sup>2</sup>]
* Leaflet
  - leaflet length [mm]
  - leaflet billowing height [mm]
  - leaflet surface area [mm<sup>2</sup>]
  - leaflet angle [°]
  - tenting height [mm]
  - tenting area [mm<sup>2</sup>]
  - tenting volume [mm<sup>3</sup>]

### Input Data Requirements
#### Multi-Class Segmentation Masks
* Format: NIfTI
* Labels:
  * 0: Annulus
  * 1: Anterior Leaflet
  * 2: Posterior Leaflet\

#### Alternative Input (for NTNU Collaboration):
* Format: NIfTI
* Labels:
  * 0: Anterior Leaflet
  * 1: Posterior Leaflet
* Folder Structure:
  
```
|-- Input/
    |-- vol01
    |-- vol02
    |-- ...
|-- Prediction/
    |-- anterior-01
    |-- anterior-02
    |-- ...
    |-- posterior-01
    |-- ...
|-- Target/
    |-- anterior-01
    |-- anterior-02
    |-- ...
    |-- posterior-01
    |-- ...
|-- VolumeGeometry/
    |-- directions
    |-- origin
    |-- resolution
```

### Output
The output is a JSON file with all computed metrics. Example:

```JSON
{
    "annulus": {
        "antero-posterior (AP) distance [mm]": 38.02983474731445,
        "anterolateral-posteriormedial (AL-PM) distance [mm]": 35.96678161621094,
        "intertrigonal (IT) distance [mm]": 35.53448298298349,
        "commissuralwidth (CW) [mm]": 35.9667854309082,
        "annular height (AH) [mm]": 9.45460577827402,
        "annular perimeter [mm]": 122.0244140625,
        "sphericity index (SI)": 1.0573598146438599,
        "non-planar angle (NAP) [deg]": 151.58218334441787,
        "annular area 2D [mm^2]": 1086.128053834045,
        "annular area 3D [mm^2]": 1122.9941331189966
    },
    "lealet": {
        "anterior": {
            "leaflet length [mm]": 25.655336380004883,
            "leaflet billowing height [mm]": 7.646640034050424,
            "leaflet surface area [mm^2]": 798.52627877487,
            "leaflet angle [deg]": 18.022349229951754
        },
        "posterior": {
            "leaflet length [mm]": 14.195778846740723,
            "leaflet billowing height [mm]": 6.652636562193375,
            "leaflet surface area [mm^2]": 811.948169342759
        },
        "tenting": {
            "height [mm]": 6.343684071073352,
            "area [mm^2]": 286.63934833296855,
            "volume [mm^3]": 2302.7401530195807
        }
    }
}
```
## References 
<a id="1">[1]</a>
Oliveira, Diana, et al. "Geometric description for the anatomy of the mitral valve: a review." Journal of Anatomy 237.2 
(2020): 209-224.
