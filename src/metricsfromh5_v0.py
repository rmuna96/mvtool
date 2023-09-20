"""
Tool box to extract morphological metrics from multi-class (anterior and posterior leaflet)
segmentation mask of mitral valve from 3DTEE. Version of the tool box for NTNU-Polimi
collaboration.
"""

import argparse
import json
import os
from os.path import join
import numpy as np
import pyvista as pv
from utils import (
    h52vtk,
    getmodelfromlabelmask,
    splinefit,
    soapfilmannulusinterpolation,
    planefit,
    clmodeling,
    distancebtwpolydata)


def landmarkdetection(annulus_pd, annulus_skeleton, bestfitplane, referenceplane, n, ctr, aleaflet_pd, pleaflet_pd,
                      plot=False):
    """
        Detect anatomical landmarks for mitral valve anatomy
        quantification.
    """
    coaptationline, saddlehorn = clmodeling(bestfitplane, annulus_skeleton, aleaflet_pd, n, ctr, plot=False)

    _, projected_point = bestfitplane.find_closest_cell(saddlehorn, return_closest_point=True)

    # ---> Define clipping planes
    direction = ctr - projected_point
    medial_lateralplane = referenceplane.copy()
    medial_lateralplane.rotate_vector(vector=direction, angle=90, point=ctr, inplace=True)
    anterior_posteriorplane = medial_lateralplane.copy()
    anterior_posteriorplane.rotate_vector(vector=n, angle=90, point=ctr, inplace=True)

    # ---> Idenitfy anterior, posterior, medial and lateral annulus portion
    annulusanterior_pd = annulus_skeleton.clip_surface(anterior_posteriorplane)
    annulusposterior_pd = annulus_skeleton.clip_surface(anterior_posteriorplane, invert=False)

    lateral_pt, medial_pt = annulusanterior_pd.points[0], annulusanterior_pd.points[-1]

    # ---> Find posterior horn
    posteriorhorn = annulusposterior_pd.slice(normal=medial_lateralplane.face_normals[0]).points

    # ---> Find lateral horn
    _, medialcommisure_pt = annulus_skeleton.find_closest_cell(coaptationline.points[0], return_closest_point=True)

    # ---> Find medial horn
    _, lateralcommisure_pt = annulus_skeleton.find_closest_cell(coaptationline.points[-1], return_closest_point=True)

    # ---> Find anterior point, coaptation point and anteriorLeaflet length
    direction = saddlehorn - posteriorhorn
    ml_plane = pv.Disc(center=ctr, inner=0, outer=100, normal=direction, r_res=50, c_res=50)
    ap_plane = ml_plane.copy()
    ap_plane.rotate_vector(vector=n, angle=90, point=ctr, inplace=True)

    aleaflet_slice = aleaflet_pd.slice(normal=medial_lateralplane.face_normals[0])
    cpoint_idx = np.argmax(np.linalg.norm(aleaflet_slice.points - saddlehorn, axis=1))
    aleaflet_cpoint = aleaflet_slice.points[cpoint_idx]

    # ---> Find posterior point, coaptation point
    pleaflet_slice = pleaflet_pd.slice(normal=medial_lateralplane.face_normals[0])
    cpoint_idx = np.argmax(np.linalg.norm(pleaflet_slice.points - posteriorhorn, axis=1))
    pleaflet_cpoint = pleaflet_slice.points[cpoint_idx]

    landmarks = {
        "annulus": {
            "saddlehorn": saddlehorn,
            "posteriorhorn": posteriorhorn,
            "lateralpt": lateral_pt,
            "medialpt": medial_pt,
            "medialcommisurept": medialcommisure_pt,
            "lateralcommisurept": lateralcommisure_pt,
        },
        "leaflet": {
            "aleafletcpoint": aleaflet_cpoint,
            "pleafletcpoint": pleaflet_cpoint,
        }
    }

    if plot:
        max_xyz = np.max(annulus_pd.points, axis=0).tolist()
        min_xyz = np.min(annulus_pd.points, axis=0).tolist()

        bb_extremities = [None] * (len(max_xyz) + len(min_xyz))
        bb_extremities[::2] = max_xyz
        bb_extremities[1::2] = min_xyz

        annulus_bb = pv.Box(bb_extremities)

        pl = pv.Plotter()
        pl.add_mesh(annulus_skeleton)
        pl.add_points(medialcommisure_pt)
        pl.add_points(lateralcommisure_pt)
        pl.add_points(lateral_pt)
        pl.add_points(medial_pt)
        pl.add_points(saddlehorn)
        pl.add_points(posteriorhorn)
        pl.add_points(aleaflet_cpoint)
        pl.add_points(pleaflet_cpoint)
        pl.add_mesh(annulus_bb.extract_feature_edges(90))
        pl.show()

    return landmarks


def anatomyquantification(annulus_skeleton, soapfilmannulus, bestfitplane, referenceplane,
                          aleaflet_pd, pleaflet_pd, landmarks, odir, folder, name):
    """
        Quantify mitral valve morphology.
    """
    # ---> Compute leaflet height respect to 3D annulus plane
    aleaflet_pd, aleafletheight = distancebtwpolydata(aleaflet_pd, soapfilmannulus)
    pleaflet_pd, pleafletheight = distancebtwpolydata(pleaflet_pd, soapfilmannulus)


    # ---> Compute antero-posterior diameter and commissural width
    anteriorposteriod = np.linalg.norm(landmarks["annulus"]["saddlehorn"] - landmarks["annulus"]["posteriorhorn"])
    mediallaterald = np.linalg.norm(
        landmarks["annulus"]["medialpt"] - landmarks["annulus"]["lateralpt"])
    commissuralwidth = np.linalg.norm(
        landmarks["annulus"]["medialcommisurept"] - landmarks["annulus"]["lateralcommisurept"])

    # ---> Compute circumference and annular area
    annulararea = soapfilmannulus.extract_surface().area
    annularcirc = annulus_skeleton.compute_arc_length()['arc_length'][-1]
    annulararea2d = bestfitplane.extract_surface().area

    # ---> Compute annular height
    annulus_skeleton.compute_implicit_distance(referenceplane, inplace=True)
    dist = annulus_skeleton["implicit_distance"]
    highest_negatived = np.min(dist)
    highest_positived = np.max(dist)

    annularheight = highest_positived - highest_negatived

    # ---> Compute anterior leaflet length
    aleafletlength = np.linalg.norm(landmarks["annulus"]["saddlehorn"] - landmarks["leaflet"]["aleafletcpoint"])

    # ---> Compute posterior leaflet length
    pleafletlength = np.linalg.norm(landmarks["annulus"]["posteriorhorn"] - landmarks["leaflet"]["pleafletcpoint"])

    # ---> Compute leaflet surface area
    aleafletarea = aleaflet_pd.area / 2
    pleafletarea = pleaflet_pd.area / 2

    metrics = {
        "annulus": {
            "anteriorposteriodiameter [mm]": anteriorposteriod.astype('float64'),
            "mediallateraldiameter [mm]": mediallaterald.astype('float64'),
            "commissuralwidth [mm]": commissuralwidth.astype('float64'),
            "annularheight [mm]": annularheight.astype('float64'),
            "annularcircumference [mm]": annularcirc.astype('float64'),
            "annulararea2d [mm^2]": annulararea2d.astype('float64'),
            "annulararea [mm^2]": annulararea.astype('float64'),
        },
        "lealet": {
            "anterior": {
                "leafletlength [mm]": aleafletlength.astype('float64'),
                "leafletheight [mm]": aleafletheight.astype('float64'),
                "leafletarea [mm^2]": aleafletarea.astype('float64'),
            },
            "posterior": {
                "leafletlength [mm]": pleafletlength.astype('float64'),
                "leafletheight [mm]": pleafletheight.astype('float64'),
                "leafletarea [mm^2]": pleafletarea.astype('float64'),
            },
        }
    }

    os.makedirs(join(odir, folder), exist_ok=True)
    with open(join(odir, folder, name + '.json'), 'w') as outflie:
        json.dump(metrics, outflie, indent=4)

    return metrics


def main(args):

    filenames = os.listdir(join(args.idir))

    for fn in filenames:
        iname = os.path.splitext(fn)[0]
        image, prediction, target = h52vtk(join(args.idir, fn))

        if args.mask == 'target':
            aleaflet_pd = getmodelfromlabelmask(target, 1, [1, 1])
            pleaflet_pd = getmodelfromlabelmask(target, 1, [2, 2])
        else:
            aleaflet_pd = getmodelfromlabelmask(prediction, 1, [1, 1])
            pleaflet_pd = getmodelfromlabelmask(prediction, 1, [2, 2])

        annulus_pd, annulus_skeleton = splinefit(aleaflet_pd, pleaflet_pd)

        soapfilmannulus = soapfilmannulusinterpolation(annulus_pd, annulus_skeleton)

        bestfitplane, referenceplane, n, ctr = planefit(annulus_pd)

        try:
            landmarks = landmarkdetection(annulus_pd, annulus_skeleton, bestfitplane, referenceplane, n, ctr,
                                          aleaflet_pd, pleaflet_pd, plot=False)
        except Exception:
            print('The segmentation mask is not sufficiently accurate to be processed')
            break

        _ = anatomyquantification(annulus_skeleton, soapfilmannulus, bestfitplane, referenceplane,
                                  aleaflet_pd, pleaflet_pd, landmarks, args.odir, args.mask, iname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', type=str, default='./images', help='input directory')
    parser.add_argument('--odir', type=str, default='./metrics', help='output directory')
    parser.add_argument('--mask', type=str, default='target', help='mask to be processed')
    args = parser.parse_args()
    main(args)
