import sep
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import numpy
import fitsio
import os
from matplotlib.patches import Ellipse
import pandas
from skimage.transform import SimilarityTransform, AffineTransform
from coordio.zernike import ZernFit, unitDiskify, unDiskify
from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY


# get default tables, any image will do
ff = fits.open(glob.glob("duPontSparseImg/59499/proc*.fits")[10])
positionerTable = pandas.DataFrame(ff[2].data)
wokCoords = pandas.DataFrame(ff[3].data)
fiducialCoords = pandas.DataFrame(ff[4].data)

# print("fiducial coords\n\n")
# print(fiducialCoords)
# print("\n\n")

def rms(actual, predicted):
    sqerr = (actual - predicted)**2
    mse = numpy.mean(sqerr)
    return numpy.sqrt(mse)


def organize():
    allImgFiles = glob.glob("duPontSparseImg/59499/proc*.fits")
    metImgFiles = []
    apMetImgFiles = []
    badFiles = []

    imgStack = []

    for ii, f in enumerate(allImgFiles):
        print("%s: %i of %i"%(f, ii, len(allImgFiles)))
        g = fits.open(f)

        maxCounts = numpy.max(g[1].data)
        if maxCounts > 62000:
            os.remove(f)
            continue

        imgStack.append(g[1].data)

        if g[1].header["LED2"] == 0:
            metImgFiles.append(f)
        else:
            apMetImgFiles.append(f)

    print(len(metImgFiles), len(apMetImgFiles))

    medianImg = numpy.median(imgStack, axis=0)
    fitsio.write("medianImg.fits", medianImg)

    with open("metImgs.txt", "w") as f:
        for mf in metImgFiles:
            f.write(mf+"\n")

    with open("apMetImgs.txt", "w") as f:
        for mf in apMetImgFiles:
            f.write(mf+"\n")


def addElipses(ax, objects):
    # plot an ellipse for each object
    # for i in range(len(objects)):
    for idx, row in objects.iterrows():
        # print("i", i)
        xy = (row['x'], row['y'])
        width = 6*row["a"]
        height = 6*row["b"]
        angle = row["theta"] * 180. / numpy.pi
        e = Ellipse(xy=xy,
                    width=width,
                    height=height,
                    angle=angle)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)


def extract(imgFile):
    imgData = fitsio.read(imgFile)
    imgData = numpy.array(imgData, dtype="float")
    bkg = sep.Background(imgData)
    bkg_image = bkg.back()
    data_sub = imgData - bkg_image
    objects = sep.extract(data_sub, 3.5, err=bkg.globalrms)
    objects = pandas.DataFrame(objects)

    # ecentricity
    objects["ecentricity"] = 1 - objects["b"] / objects["a"]

    # slope of ellipse (optical distortion direction)
    objects["slope"] = numpy.tan(objects["theta"] + numpy.pi/2) # rotate by 90
    # intercept of optical distortion direction
    objects["intercept"] = objects["y"] - objects["slope"] * objects["x"]

    objects = objects[objects["npix"] > 100]

    # filter on most eliptic, this is an assumption!!!!
    objects["outerFIF"] = objects.ecentricity > 0.15

    return imgData, objects


def findOpticalCenter(imgFile, plot=False):
    # psf angles point toward the center of the
    # optical system...
    imgData, objects = extract(imgFile)

    outerFIFs = objects[objects.outerFIF == True]

    # find the best point for the center of the distortion
    A = numpy.ones((len(outerFIFs), 2))
    A[:,1] = -1 * outerFIFs.slope
    b = outerFIFs.intercept
    out = numpy.linalg.lstsq(A, b)

    yOpt = out[0][0]
    xOpt = out[0][1]

    if plot:
        plt.figure(figsize=(10,10))
        plt.imshow(equalize_hist(imgData), origin="lower")
        ax = plt.gca()
        addElipses(ax, objects)
        xs = numpy.arange(imgData.shape[1])
        miny = 0
        maxy = imgData.shape[0]
        for idx, ofif in outerFIFs.iterrows():
            ys = ofif.slope * xs + ofif.intercept
            keep = (ys > 0) & (ys < maxy)

            ax.plot(xs[keep],ys[keep],':', color="cyan", alpha=1)
            plt.plot(xOpt, yOpt, "or")  #center of coma?
        plt.show()

    return xOpt, yOpt


SimTrans = SimilarityTransform(
    translation = numpy.array([-467.96317617, -356.55049009]),
    rotation = 0.003167701580600088,
    scale = 0.11149215438621102
)


def associateFiducials(imgFile, plot=False):
    xOptCCD, yOptCCD = findOpticalCenter(imgFile)

    imgData, objects = extract(imgFile)

    # scale pixels to mm roughly
    xCCD = objects.x.to_numpy()
    yCCD = objects.y.to_numpy()
    xWok = fiducialCoords.xWok.to_numpy()
    yWok = fiducialCoords.yWok.to_numpy()
    meanCCDX = numpy.mean(xCCD)
    meanCCDY = numpy.mean(yCCD)
    stdCCDX = numpy.std(xCCD)
    stdCCDY = numpy.std(yCCD)

    stdWokX = numpy.std(xWok)
    stdWokY = numpy.std(yWok)

    # scale to rough wok coords enough to make association
    roughWokX = (xCCD - meanCCDX) / stdCCDX * stdWokX
    roughWokY = (yCCD - meanCCDY) / stdCCDY * stdWokY


    xCCDFound = []
    yCCDFound = []
    roughWokXFound = []
    roughWokYFound = []
    xWokFound = []
    yWokFound = []

    for cmmx, cmmy in zip(xWok, yWok):
        dist = numpy.sqrt((roughWokX - cmmx)**2 + (roughWokY - cmmy)**2)
        amin = numpy.argmin(dist)

        # throw out outer fiducials
        if objects.outerFIF.to_numpy()[amin]:
            continue

        xCCDFound.append(xCCD[amin])
        yCCDFound.append(yCCD[amin])
        roughWokXFound.append(roughWokX[amin])
        roughWokYFound.append(roughWokY[amin])
        xWokFound.append(cmmx)
        yWokFound.append(cmmy)


    tform = SimilarityTransform()
    # tform = AffineTransform()
    src = numpy.array([xCCDFound, yCCDFound]).T
    dest = numpy.array([xWokFound, yWokFound]).T
    tform.estimate(src, dest)

    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # where is the optical center in wok coords
    xyOptWok = tform([[xOptCCD, yOptCCD]])
    print("xyOptWok", xyOptWok)

    xyFit = tform(src)
    xyError = (dest - xyFit)*1000 # error in microns

    print("rms similarity microns", rms(xyFit*1000, dest*1000))


    # xyFit -= xyOptWok
    # dest -= xyOptWok

    # theta = numpy.pi/2
    # rotMat = numpy.array([
    #     [numpy.cos(theta), -numpy.sin(theta)],
    #     [numpy.sin(theta), numpy.cos(theta)]
    # ])

    # xyFit = (rotMat @ xyFit.T).T
    # dest = (rotMat @ dest.T).T
    # rotate by 90
    # for orders in range(3,20):
    #     zf = ZernFit(xyFit[:,0], xyFit[:,1], dest[:,0], dest[:,1], method="grad", orders=orders)
    #     zxyFit = zf.apply(xyFit[:,0], xyFit[:,1])
    #     zxyFit = numpy.array(zxyFit).T

    #     zxyError = (dest - zxyFit)*1000 # error in microns

    #     print("order: %i, points %i rms zern microns"%(len(zf.coeff), len(zf.zxyStack)), rms(zxyFit*1000, dest*1000))


    # cross validate

    # scaleR = 1.1*numpy.max(numpy.sqrt(xyFit[:,0]**2+xyFit[:,1]**2))
    # for orders in range(3,20):
    #     errs = []

    #     for ii in range(len(xyFit)):
    #         _xyFit = xyFit.copy()
    #         _dest = dest.copy()
    #         xyTest = xyFit[ii]
    #         destTest = dest[ii]
    #         _xyFit = numpy.delete(_xyFit, ii, axis=0)
    #         _dest = numpy.delete(_dest, ii, axis=0)
    #         zf = ZernFit(_xyFit[:,0], _xyFit[:,1], _dest[:,0], _dest[:,1], method="grad", orders=orders, scaleR=scaleR)
    #         zxyFit = zf.apply(xyTest[0], xyTest[1])
    #         err = (destTest-numpy.array(zxyFit))*1000
    #         errs.append(err)

    #     print("order", orders, "unbiased error rms", numpy.sqrt(numpy.mean(numpy.array(errs)**2)))


    # zf = ZernFit(xyFit[:,0], xyFit[:,1], dest[:,0], dest[:,1], method="grad", orders=5)
    # zxyFit = zf.apply(xyFit[:,0], xyFit[:,1])
    # zxyFit = numpy.array(zxyFit).T

    # zxyError = (dest - zxyFit)*1000 # error in microns

    # print("order: rms zern microns",rms(zxyFit*1000, dest*1000))

    # try desi's zb fitter
    # cross validate it
    # idlist = [0,1,2,3,4]
    polids= numpy.array([0,1,2,3,4,5,6,9,20,27,28,29,30],dtype=int)
    nextCoeff = 5

    _x, _y, scaleR = unitDiskify(xyFit[:,0], xyFit[:,1])
    nxyFit = numpy.array([_x, _y]).T
    _x, _y, scaleR = unitDiskify(dest[:,0], dest[:,1], scaleR)
    nxyDest = numpy.array([_x,_y]).T

    # plt.figure()
    # plt.plot(nxyFit[:,0], nxyFit[:,1], '.k')
    # plt.plot(nxyDest[:,0], nxyDest[:,1], 'xk')
    # plt.show()

    errs = []
    for ii in range(len(xyFit)):
        _xyFit = nxyFit.copy()
        _dest = nxyDest.copy()
        fitCheck = numpy.array(nxyFit[ii,:]).reshape((1,2))
        destCheck = numpy.array(nxyDest[ii,:]).reshape((1,2))
        _xyFit = numpy.delete(_xyFit, ii, axis=0)
        _dest = numpy.delete(_dest, ii, axis=0)
        polids, coeffs = fitZhaoBurge(_xyFit[:,0], _xyFit[:,1], _dest[:,0], _dest[:,1], polids=polids)
        dx, dy = getZhaoBurgeXY(polids, coeffs, fitCheck[:,0], fitCheck[:,1])
        zxfit = fitCheck[:,0] + dx
        zyfit = fitCheck[:,1] + dy
        zxfit, zyfit = unDiskify(zxfit,zyfit,scaleR)
        zxyfit = numpy.array([zxfit, zyfit]).T
        errs.append(dest[ii]-zxyfit.squeeze())

        # zxfit = xyFit[:,0] + dx
        # zyfit = xyFit[:,1] + dy

        # zxyFit = numpy.array([zxfit, zyfit]).T

        # zxyError = (dest - zxyFit)*1000

    errs = numpy.array(errs) * 1000
    plt.figure()
    plt.hist(numpy.linalg.norm(errs, axis=1))
    plt.show()
    print("polids", polids)
    print("unbiased rms zernike fit", numpy.sqrt(numpy.mean(errs**2)))


    polids, coeffs = fitZhaoBurge(xyFit[:,0], xyFit[:,1], dest[:,0], dest[:,1], polids=polids)
    dx, dy = getZhaoBurgeXY(polids, coeffs, xyFit[:,0], xyFit[:,1])
    zxfit = xyFit[:,0] + dx
    zyfit = xyFit[:,1] + dy
    zxyFit = numpy.array([zxfit, zyfit]).T
    zxyError = (dest-zxyFit)*1000

    print("rms zernike fit", numpy.sqrt(numpy.mean(zxyError**2)))
    # plt.figure()
    # plt.plot(polids, coeffs, '.-k')
    # plt.show()

    if plot:
        plt.figure(figsize=(10, 10))
        plt.title("Rough Transform + Assoc")
        plt.plot(xWok, yWok, 'or')
        # plot nearest neighbor detections
        for ii in range(len(xWokFound)):
            plt.plot(roughWokXFound[ii], roughWokYFound[ii], 'xb')
            plt.plot([xWokFound[ii], roughWokXFound[ii]], [yWokFound[ii], roughWokYFound[ii]], 'k')

        # plot all detections
        plt.plot(roughWokX, roughWokY, '*g')

        plt.figure(figsize=(10,10))
        plt.title("SimilarityTransform")
        plt.quiver(dest[:,0], dest[:,1], xyError[:,0], xyError[:,1], angles="xy")


        plt.figure(figsize=(10,10))
        plt.title("Zern Transform")
        plt.quiver(dest[:,0], dest[:,1], zxyError[:,0], zxyError[:,1], angles="xy")


        plt.show()


# with open("metImgs.txt", "r") as f:
#     metImgs = f.readlines()

# for metImg in metImgs:

associateFiducials("medianImg.fits", plot=True)

