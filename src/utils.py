import h5py
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import SimpleITK as sitk
import numpy as np
import pyvista as pv
import scipy
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation
from math import cos, sin, sqrt


def h52vtk(fname):
    """
        Create a VTK Image from h5 file. This function returns 3 VTK Images
        one for the input image, one for the prediction and one for the
        groundtruth.
    """

    m2mm = 1000

    hdf = h5py.File(fname, 'r')
    size = list(hdf['Input']['vol01'][()].shape)
    origin = list(hdf['VolumeGeometry']['origin'][()] * m2mm)
    bbox = hdf['VolumeGeometry']['directions'][()]
    dir1 = bbox[0, :]
    dir2 = bbox[1, :]
    dir3 = bbox[2, :]
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)
    dir3 = dir3 / np.linalg.norm(dir3)
    direction = list((dir1[0], dir2[0], dir3[0], dir1[1], dir2[1], dir3[1], dir1[2], dir2[2], dir3[2]))
    spacing = list(hdf['VolumeGeometry']['resolution'][()] * m2mm)

    ncomp = 1  # for scalar images

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    inp = vtk.vtkImageData()

    img = hdf['Input']["vol01"][()].astype(np.float32)

    inp.SetDimensions(size)
    inp.SetSpacing(spacing)
    inp.SetOrigin(origin)
    inp.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        inp.SetDirectionMatrix(direction)

    depth_array = numpy_to_vtk(img.ravel())
    depth_array.SetNumberOfComponents(ncomp)
    inp.GetPointData().SetScalars(depth_array)

    inp.Modified()

    prediction = vtk.vtkImageData()

    ant = hdf['Prediction']["anterior-01"][()].astype(np.uint8)
    post = hdf['Prediction']["posterior-01"][()].astype(np.uint8)
    background = np.zeros_like(ant)
    background[(ant == 0) & (post == 0)] = 1
    pred_1hot = np.stack([background, ant, post])
    pred = np.argmax(pred_1hot, axis=0, keepdims=True).squeeze(0)

    prediction.SetDimensions(size)
    prediction.SetSpacing(spacing)
    prediction.SetOrigin(origin)
    prediction.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        prediction.SetDirectionMatrix(direction)

    depth_array = numpy_to_vtk(pred.ravel())
    depth_array.SetNumberOfComponents(ncomp)
    prediction.GetPointData().SetScalars(depth_array)

    prediction.Modified()

    target = vtk.vtkImageData()

    ant = hdf['Target']["anterior-01"][()].astype(np.uint8)
    post = hdf['Target']["posterior-01"][()].astype(np.uint8)
    background = np.zeros_like(ant)
    background[(ant == 0) & (post == 0)] = 1
    gt_1hot = np.stack([background, ant, post])
    gt = np.argmax(gt_1hot, axis=0, keepdims=True).squeeze(0)

    target.SetDimensions(size)
    target.SetSpacing(spacing)
    target.SetOrigin(origin)
    target.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        target.SetDirectionMatrix(direction)

    depth_array = numpy_to_vtk(gt.ravel())
    depth_array.SetNumberOfComponents(ncomp)
    target.GetPointData().SetScalars(depth_array)

    target.Modified()
    return inp, prediction, target


def getmodelfromlabelmask(image, numlabels, rangevalues):
    """
        Apply discrete marching cubes algorithm to a vtk image.
        Return a pyvista object.
    """
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(image)
    dmc.ComputeNormalsOn()
    dmc.ComputeGradientsOn()
    dmc.GenerateValues(numlabels, rangevalues)  # rangevalues: range of labels
    dmc.Update()
    pd = pv.wrap(dmc.GetOutput())  # pyvista wrapping
    return pd


def distancebtwpolydata(pd1, pd2):
    """
        pd1 and pd2 must be pyvista object
    """
    pd1 = pd1.extract_surface()
    pd2 = pd2.extract_surface()
    _ = pd1.compute_implicit_distance(pd2, inplace=True)
    dist = pd1['implicit_distance']

    return pd1, np.max(dist)


def rotationmatrix(axis, theta):
    """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def splinefit(aleaflet_pd, pleaflet_pd, plot=False):
    """
        Delinate the perimeter of mitral valve leaflets and reconstruct the annulus
        model.
    """
    valve_pd = aleaflet_pd + pleaflet_pd
    points = valve_pd.points.T

    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    m = np.dot(x, x.T)
    n = np.linalg.svd(m)[0][:, -1]  # normal direction
    r = np.linalg.svd(m)[0][:, 1]  # radial direction
    rot_90 = rotationmatrix(n, np.deg2rad(90))  # rotation matrix
    alpha = 0
    p = 0
    angular_offset = 10  # degrees
    l_rawspline = int(370 / angular_offset)  # spline length
    points_rawspline = np.zeros((l_rawspline, 3))
    while alpha < 360:
        alpha_rad = np.deg2rad(alpha)
        rot_alpha = rotationmatrix(n, alpha_rad)  # rotation matrix
        r_rot = np.dot(rot_alpha, r)  # rotating radial direction
        r_vec = ctr - 20 * r_rot
        rot_disc = pv.Disc(center=r_vec + 10 * (r_vec - ctr) / np.linalg.norm(r_vec - ctr),
                           outer=2 * np.linalg.norm(r_vec - ctr),
                           normal=np.dot(rot_90, r_rot), c_res=50)
        box = rot_disc.extrude(np.dot(rot_90, r_rot), capping=True)
        selected = valve_pd.select_enclosed_points(box)
        ptscloud = valve_pd.extract_points(selected['SelectedPoints'].view(bool),
                                           adjacent_cells=False)
        idx = np.argmax(np.linalg.norm(ptscloud.points - ctr, axis=1))
        points_rawspline[p] = ptscloud.points[idx]

        alpha += 10
        p += 1

    points_rawspline = points_rawspline[~np.isnan(points_rawspline)]
    points_rawspline = points_rawspline.reshape(int(points_rawspline.shape[0] / 3), 3)

    tck, u = splprep(points_rawspline.T, u=None, k=5, s=20, per=1)
    fittedpoints = splev(np.linspace(u.min(), u.max(), 1000), tck)

    points_fittedspline = np.array(fittedpoints).T

    fittedspline_reference = pv.Spline(points_fittedspline)
    fittedspline = fittedspline_reference.tube(radius=1)

    if plot:
        pl = pv.Plotter()
        pl.add_mesh(aleaflet_pd)
        pl.add_mesh(pleaflet_pd)
        pl.add_mesh(fittedspline, opacity=0.4)
        pl.add_mesh(fittedspline_reference, color='red')
        pl.show()

    return fittedspline, fittedspline_reference


def soapfilmannulusinterpolation(annulus_pd, annulus_skeleton, plot=False):
    """
        Compute 3D surface interpolating the annulus. Inputs are rotated to have principal
        directions coherent with the x, y, z axes.
    """

    points = annulus_pd.points.T
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    m = np.dot(x, x.T)
    pdirs = np.linalg.svd(m)[0]

    r, _, = Rotation.align_vectors(pdirs, np.eye(3))

    rot_matrix = np.vstack([np.hstack([r.as_matrix(), np.expand_dims([0, 0, 0], 0).T]), np.array([0, 0, 0, 1])])

    annulus_rot = annulus_pd.transform(rot_matrix, transform_all_input_vectors=True, inplace=False)
    annulus_skeleton_rot = annulus_skeleton.transform(rot_matrix, transform_all_input_vectors=True, inplace=False)

    # ---> Define the mesh grid on the annulus
    x_probe, y_probe, z_probe = annulus_rot.points.T[0], annulus_rot.points.T[1], annulus_rot.points.T[2]

    xi, yi = np.linspace(x_probe.min(), x_probe.max(), 400), np.linspace(y_probe.min(), y_probe.max(), 400)
    xi, yi = np.meshgrid(xi, yi)

    # ---> Sample point for the interpolation
    choice = np.arange(0, annulus_rot.points.T[0].shape[0], annulus_rot.points.T[0].shape[0] / 4000, dtype=int)
    spline = scipy.interpolate.Rbf(x_probe[choice], y_probe[choice], z_probe[choice], function='thin_plate', smooth=250)
    zi = spline(xi, yi)

    # ---> Create a pyvista grid from the interpolated point
    grid = pv.StructuredGrid(xi, yi, zi)

    extruded_annulus = annulus_skeleton_rot.extrude([0, 0, 10], capping=False)
    extruded_annulus.extrude([0, 0, -10], inplace=True, capping=False)
    if np.dot(extruded_annulus.points[0] - ctr, extruded_annulus.cell_normals[0]) < 0:
        film = grid.clip_surface(extruded_annulus, invert=False)
    else:
        film = grid.clip_surface(extruded_annulus, invert=True)
    film = film.connectivity(largest=True)
    film.transform(rot_matrix.T, transform_all_input_vectors=True, inplace=True)

    if plot:
        pl = pv.Plotter()
        pl.add_mesh(annulus_pd)
        pl.add_mesh(film)
        pl.add_axes()
        pl.show()

    return film


def leafletmedialsurface(vtkimage, label=None):
    """
        Extracting the medial surface of a leaflet from a
        vtk image. Conversion to sitk is necessary for
        the use of sitk-based filters. Return medial surface
        of anterior and posterior leaflet.
    """
    sitkimage = vtk2sitk(vtkimage)
    spacing = sitkimage.GetSpacing()
    min_spacing = min(spacing)
    sitkleaflet = sitk.BinaryThreshold(sitkimage, lowerThreshold=label, upperThreshold=label, insideValue=1,
                                       outsideValue=0)
    sitkthinnedleaflet = sitk.BinaryThinning(sitkleaflet)

    npthinnedleaflet = sitk.GetArrayFromImage(sitkthinnedleaflet)

    fgidx = np.flip(np.argwhere(npthinnedleaflet == 1), 1).tolist()

    voxel_pts = []
    for idx in fgidx:
        sitkpt = sitkthinnedleaflet.TransformIndexToPhysicalPoint(idx)
        voxel_pts.append(sitkpt)

    voxel_pts = np.array(voxel_pts)
    medialsurf = pv.PolyData(voxel_pts)
    medialsurf.delaunay_2d(inplace=True, alpha=min_spacing*2)
    medialsurf.smooth(1000, inplace=True)

    return medialsurf


def spherefit(spx, spy, spz):
    """
        Fit a sphere to a 3D point cloud.
    """
    #   Assemble the A matrix
    spx = np.array(spx)
    spy = np.array(spy)
    spz = np.array(spz)
    a = np.zeros((len(spx), 4))
    a[:, 0] = spx*2
    a[:, 1] = spy*2
    a[:, 2] = spz*2
    a[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spx), 1))
    f[:, 0] = (spx*spx) + (spy*spy) + (spz*spz)
    c, residules, rank, singval = np.linalg.lstsq(a, f)

    #   solve for the radius
    t = (c[0]*c[0])+(c[1]*c[1])+(c[2]*c[2])+c[3]
    radius = sqrt(t)
    center = [c[0].item(), c[1].item(), c[2].item()]

    return radius, center


def planefit(annulus_pd, plot=False):
    """
        Find the best fit plane to the annulus. Return the best fit plane
        bounded by the annulus, the best fit plane, the normal and the center
        of the annulus.
    """
    points = annulus_pd.points.T

    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    m = np.dot(x, x.T)
    n = np.linalg.svd(m)[0][:, -1]

    r, c = spherefit(points[0], points[1], points[2])

    bestfitdisc = pv.Disc(center=ctr, outer=r, inner=0, normal=n, r_res=50, c_res=50)
    bestfitplane = pv.Disc(center=ctr, outer=r * 1.5, inner=0, normal=n, r_res=50, c_res=50)

    if plot:
        pl = pv.Plotter()
        pl.add_mesh(annulus_pd)
        pl.add_mesh(bestfitdisc)
        pl.show()

    return bestfitdisc, bestfitplane, n, ctr


def clmodeling(fitplane, annulus_skeleton, aleaflet_pd, normal, center, plot=False):
    """
        Model and localize the coaptation line by finding the free
        margin of the anterior leaflet. Return the coaptation line
        model as polydata and the saddle horn coordinates.
    """
    # ---> Find Saddle Horn
    annulus_skeleton.compute_implicit_distance(fitplane, inplace=True)
    dist = annulus_skeleton["implicit_distance"]
    if np.dot(normal, fitplane.cell_normals[0]) < 0:
        saddle_horn_distance = np.min(dist)
    else:
        saddle_horn_distance = np.max(dist)
    saddle_horn_idx = np.where(dist == saddle_horn_distance)
    saddle_horn = annulus_skeleton.points[saddle_horn_idx[0][0]]
    _, projected_point = fitplane.find_closest_cell(saddle_horn, return_closest_point=True)

    ap_plane = fitplane.rotate_vector(vector=projected_point - center, angle=90, point=center)

    coaptation_pts = []
    for i, angle in enumerate(range(-40, 45)):
        slice_plane = ap_plane.rotate_vector(vector=normal, angle=angle, point=saddle_horn)
        idx = np.argmax(np.linalg.norm(aleaflet_pd.slice(normal=slice_plane.cell_normals[0], origin=saddle_horn
                                                         ).points - saddle_horn, axis=1), axis=0)
        coaptation_pts.append(aleaflet_pd.slice(normal=slice_plane.cell_normals[0], origin=saddle_horn).points[idx])

    points = np.array(coaptation_pts)
    tck, u = splprep(points.T, u=None, k=5, s=30)
    fittedpoints = splev(np.linspace(u.min(), u.max(), 1000), tck)
    fittedpoints = np.array(fittedpoints).T
    coaptation_line = pv.Spline(fittedpoints)

    if plot:
        pl = pv.Plotter()
        pl.add_mesh(annulus_skeleton)
        pl.add_mesh(coaptation_line)
        pl.add_mesh(saddle_horn)
        pl.show()

    return coaptation_line, saddle_horn


def vtk2sitk(vtkimg, debug=False):
    """
        Takes a VTK image, returns a SimpleITK image.
    """
    sd = vtkimg.GetPointData().GetScalars()
    npdata = vtk_to_numpy(sd)

    dims = list(vtkimg.GetDimensions())
    origin = vtkimg.GetOrigin()
    spacing = vtkimg.GetSpacing()

    if debug:
        print("dims:", dims)
        print("origin:", origin)
        print("spacing:", spacing)

        print("numpy type:", npdata.dtype)
        print("numpy shape:", npdata.shape)

    dims.reverse()
    npdata.shape = tuple(dims)
    if debug:
        print("new shape:", npdata.shape)
    sitkimg = sitk.GetImageFromArray(npdata)
    sitkimg.SetSpacing(spacing)
    sitkimg.SetOrigin(origin)

    if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
        direction = vtkimg.GetDirectionMatrix()
        d = []
        for y in range(3):
            for x in range(3):
                d.append(direction.GetElement(y, x))
        sitkimg.SetDirection(d)
    return sitkimg


def sitk2vtk(img, debugon=False):
    """Convert a SimpleITK image to a VTK image, via numpy."""

    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    # there doesn't seem to be a way to specify the image orientation in VTK

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    if debugon:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))

    vtk_image = vtk.vtkImageData()

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    # depth_array = numpy_support.numpy_to_vtk(i2.ravel(), deep=True,
    #                                          array_type = vtktype)
    depth_array = numpy_to_vtk(i2.ravel())
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()
    #
    if debugon:
        print("Volume object inside sitk2vtk")
        print(vtk_image)
        #        print("type = ", vtktype)
        print("num components = ", ncomp)
        print(size)
        print(origin)
        print(spacing)
        print(vtk_image.GetScalarComponentAsFloat(0, 0, 0, 0))

    return vtk_image
