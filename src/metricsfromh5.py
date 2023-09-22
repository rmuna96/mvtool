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

from functions import (
    h52vtk,
    getmodelfromlabelmask,
    splinefit,
    soapfilmannulusinterpolation,
    planefit,
    clmodeling,
    anter_postsplit,
    distancebtwpolydata)

from utils import Logger
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def landmarkdetection(annulus_pd, annulus_skeleton, bestfitplane, referenceplane, n, ctr, aleaflet_pd, pleaflet_pd,
                      plot=False):
    """
        Detect anatomical landmarks for mitral valve anatomy
        quantification.
    """
    coaptationline, saddlehorn = clmodeling(bestfitplane, annulus_skeleton, aleaflet_pd, n, ctr, plot=False)

    _, saddlehorn_onplane = bestfitplane.find_closest_cell(saddlehorn, return_closest_point=True)

    # ---> Define clipping planes
    direction = ctr - saddlehorn_onplane
    medial_lateralplane = referenceplane.copy()
    medial_lateralplane.rotate_vector(vector=direction, angle=90, point=ctr, inplace=True)
    anterior_posteriorplane = medial_lateralplane.copy()
    anterior_posteriorplane.rotate_vector(vector=n, angle=90, point=ctr, inplace=True)

    # ---> Idenitfy anterior, posterior, medial and lateral annulus portion
    annulusanterior_pd = annulus_skeleton.clip_surface(anterior_posteriorplane)
    annulusposterior_pd = annulus_skeleton.clip_surface(anterior_posteriorplane, invert=False)

    lateral_pt, medial_pt = annulusanterior_pd.points[0], annulusanterior_pd.points[-1]

    # ---> Find posterior horn
    posteriorhorn = annulusposterior_pd.slice(normal=medial_lateralplane.face_normals[0]).points[0]

    _, posteriorhorn_onplane = bestfitplane.find_closest_cell(posteriorhorn, return_closest_point=True)

    # ---> Define anterior and posterior portion (2/5 - 3/5 rule)
    medialcommisure_pt, lateralcommisure_pt = anter_postsplit(annulus_skeleton, ctr, n, direction,
                                                              anterior_posteriorplane, plot=False)

    cw_midpt = 0.5 * (medialcommisure_pt + lateralcommisure_pt)

    # ---> Find anterior point, coaptation point and anteriorLeaflet length
    direction = saddlehorn - posteriorhorn
    ml_plane = pv.Disc(center=ctr, inner=0, outer=100, normal=direction, r_res=50, c_res=50)
    ap_plane = ml_plane.copy()
    ap_plane.rotate_vector(vector=n, angle=90, point=ctr, inplace=True)

    aleaflet_slice = aleaflet_pd.slice(normal=medial_lateralplane.face_normals[0])
    cpoint_idx = np.argmax(np.linalg.norm(aleaflet_slice.points - saddlehorn, axis=1))
    aleaflet_cpoint = aleaflet_slice.points[cpoint_idx]

    # ---> Find posterior point, leaflet coaptation point
    pleaflet_slice = pleaflet_pd.slice(normal=medial_lateralplane.face_normals[0])
    cpoint_idx = np.argmax(np.linalg.norm(pleaflet_slice.points - posteriorhorn, axis=1))
    pleaflet_cpoint = pleaflet_slice.points[cpoint_idx]

    # ---> Find coaptation point
    _, clpts_onplane = bestfitplane.find_closest_cell(coaptationline.points, return_closest_point=True)
    coapt_pt = coaptationline.points[np.argmax(np.linalg.norm(coaptationline.points - clpts_onplane, axis=1))]

    landmarks = {
        "annulus": {
            "saddlehorn": saddlehorn,
            "saddlehornonplane": saddlehorn_onplane,
            "posteriorhorn": posteriorhorn,
            "posteriorhornonplane": posteriorhorn_onplane,
            "lateralpt": lateral_pt,
            "medialpt": medial_pt,
            "medialcommisurept": medialcommisure_pt,
            "lateralcommisurept": lateralcommisure_pt,
            "cwmidpt": cw_midpt,
        },
        "leaflet": {
            "aleafletcpoint": aleaflet_cpoint,
            "pleafletcpoint": pleaflet_cpoint,
            "clpoints": coaptationline.points,
            "clptsonplane": clpts_onplane,
            "coaptpt": coapt_pt,
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
        pl.add_points(medialcommisure_pt, label="medialcommisure")
        pl.add_points(lateralcommisure_pt, label="lateralcommisure")
        pl.add_points(cw_midpt, label="cwmidpoint")
        pl.add_points(lateral_pt, label="lateralpt")
        pl.add_points(medial_pt, label="medialpt")
        pl.add_points(saddlehorn, label="saddlehorn")
        pl.add_points(posteriorhorn, label="posteriorhorn")
        pl.add_points(saddlehorn_onplane, label="shonplane")
        pl.add_points(posteriorhorn_onplane, label="phonplane")
        pl.add_points(aleaflet_cpoint, label="aleafletcpt")
        pl.add_points(pleaflet_cpoint, label="pleafletcpt")
        pl.add_points(coapt_pt, label="coaptpt")
        pl.add_mesh(annulus_bb.extract_feature_edges(90))
        pl.add_legend()
        pl.show()

    return landmarks


def anatomyquantification(annulus_skeleton, soapfilmannulus, bestfitplane, n, ctr, r, referenceplane,
                          aleaflet_pd, pleaflet_pd, landmarks, odir, folder, name):
    """
        Quantify mitral valve morphology.
    """

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

    # ---> Compute inter-trigonal distance
    itdist = 19.280 * np.exp(0.017 * mediallaterald)    #compute according [1]

    # ---> Comnpute sphericity index
    ci = anteriorposteriod/commissuralwidth

    # ---> Compute non-planar angle
    ba = landmarks["annulus"]["cwmidpt"] - landmarks["annulus"]["saddlehorn"]
    bc = landmarks["annulus"]["cwmidpt"] - landmarks["annulus"]["posteriorhorn"]

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    NPA = np.arccos(cosine_angle) * 180/np.pi

    # ---> Compute anterior leaflet length
    aleafletlength = np.linalg.norm(landmarks["annulus"]["saddlehorn"] - landmarks["leaflet"]["aleafletcpoint"])

    # ---> Compute posterior leaflet length
    pleafletlength = np.linalg.norm(landmarks["annulus"]["posteriorhorn"] - landmarks["leaflet"]["pleafletcpoint"])

    # ---> Compute leaflet height respect to 3D annulus plane
    aleaflet_pd, aleafletheight = distancebtwpolydata(aleaflet_pd, soapfilmannulus)
    pleaflet_pd, pleafletheight = distancebtwpolydata(pleaflet_pd, soapfilmannulus)

    # ---> Compute leaflet surface area
    aleafletarea = aleaflet_pd.area / 2
    pleafletarea = pleaflet_pd.area / 2

    # ---> Compute tenting height, tenting area and tentng volume

    tentingheight = np.max(np.linalg.norm(landmarks["leaflet"]["clpoints"] -
                                                                         landmarks["leaflet"]["clptsonplane"], axis=1))
    tentingarea = pv.PolyData([
        landmarks["annulus"]["saddlehornonplane"],
        landmarks["annulus"]["posteriorhornonplane"],
        landmarks["leaflet"]["coaptpt"],
    ], strips=[3, 1, 2, 3]).area

    tentingvolume = 1/3 * np.pi * r ** 2 * np.dot(-n, (landmarks["leaflet"]["coaptpt"]-ctr)/np.linalg.norm(
        landmarks["leaflet"]["coaptpt"]-ctr)) * np.linalg.norm(landmarks["leaflet"]["coaptpt"]-ctr)

    ba = landmarks["annulus"]["saddlehornonplane"] - landmarks["leaflet"]["coaptpt"]
    bc = landmarks["annulus"]["saddlehornonplane"] - landmarks["annulus"]["posteriorhornonplane"]

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    aleafletangle = np.arccos(cosine_angle) * 180 / np.pi

    ba = landmarks["annulus"]["posteriorhornonplane"] - landmarks["leaflet"]["coaptpt"]
    bc = landmarks["annulus"]["posteriorhornonplane"] - landmarks["annulus"]["saddlehornonplane"]

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    pleafletangle = np.arccos(cosine_angle) * 180 / np.pi

    metrics = {
        "annulus": {
            "antero-posterior (AP) distance [mm]": anteriorposteriod.astype('float64'),
            "anterolateral-posteriormedial (AL-PM) distance [mm]": mediallaterald.astype('float64'),
            "intertrigonal (IT) distance [mm]": itdist.astype('float64'),
            "commissuralwidth (CW) [mm]": commissuralwidth.astype('float64'),
            "annular height (AH) [mm]": annularheight.astype('float64'),
            "annular perimeter [mm]": annularcirc.astype('float64'),
            "sphericity index (SI)": ci.astype('float64'),
            "non-planar angle (NAP) [deg]": NPA.astype('float64'),
            "annular area 2D [mm^2]": annulararea2d.astype('float64'),
            "annular area 3D [mm^2]": annulararea.astype('float64'),
        },
        "lealet": {
            "anterior": {
                "leaflet length [mm]": aleafletlength.astype('float64'),
                "leaflet billowing height [mm]": aleafletheight.astype('float64'),
                "leaflet surface area [mm^2]": aleafletarea.astype('float64'),
                "leaflet angle [deg]": aleafletangle.astype('float64'),
            },
            "posterior": {
                "leaflet length [mm]": pleafletlength.astype('float64'),
                "leaflet billowing height [mm]": pleafletheight.astype('float64'),
                "leaflet surface area [mm^2]": pleafletarea.astype('float64'),
            },
            "tenting": {
                "height [mm]": tentingheight.astype('float64'),
                "area [mm^2]": tentingarea.astype('float64'),
                "volume [mm^3]": tentingvolume.astype('float64'),

            },
        }
    }

    os.makedirs(join(odir, folder), exist_ok=True)
    with open(join(odir, folder, name + '.json'), 'w') as outflie:
        json.dump(metrics, outflie, indent=4)

    return metrics


def main(args):

    logger = Logger(join(args.odir, args.mask), 'logs').get_logger()

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

        bestfitplane, referenceplane, n, ctr, r = planefit(annulus_pd)

        try:
            landmarks = landmarkdetection(annulus_pd, annulus_skeleton, bestfitplane, referenceplane, n, ctr,
                                          aleaflet_pd, pleaflet_pd, plot=False)
        except Exception:
            logger.info(f'{iname} is not sufficiently accurate to be processed')
            continue

        _ = anatomyquantification(annulus_skeleton, soapfilmannulus, bestfitplane, n, ctr, r, referenceplane,
                                  aleaflet_pd, pleaflet_pd, landmarks, args.odir, args.mask, iname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics from segmentation mask')
    parser.add_argument('idir', type=str, default='./images', help='input directory')
    parser.add_argument('odir', type=str, default='./metrics', help='output directory')
    parser.add_argument('--mask', type=str, default='target', help='mask to be processed (target o prediction)')
    args = parser.parse_args()
    main(args)
