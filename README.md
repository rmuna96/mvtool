# Mitral Valve (MV) tool

## Extract morphological metrics from multi-class segmentation of mitral valve from 3D ultrasound.

This repository is intended to be used to validate the performance of deep learning models for automatic mitral valve 
segmentation from 3D ultrasound by automatically measuring metrics commonly used in the clinic to characterize mitral 
valve morphology.

## Requirements

See `requirements.txt` for all needed Python packages.

## Usage

### Compute morphological metrics from multi-class (anterior and posterior leaflet) MV segmentation

`$ python metricsfromh5.py [OPTIONS] IDIR ODIR`. For more information see `$ python metricsfromh5.py -h`.

This will compute morphological metrics according to the proivded segmentation mask (target or prediction). The 
definitions of the anatomical landmarks and the metrics were taken from [[1]](#1). All the metrics computed by this 
tool are reported below:
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

All the input data must be HDF files with the following structure:
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
The output is a json file with all the computed metrics as shown below:
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