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
# from coordio.zernike import ZernFit, unitDiskify, unDiskify
from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY
import coordio
from kaiju import RobotGridCalib
from kaiju.utils import plotOne
import coordio
from coordio import defaults
import time
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
from functools import partial


# get default tables, any image will do, they're all the same
ff = fits.open(glob.glob("duPontSparseImg/59499/proc*.fits")[10])
# positionerTable = ff[2].data  # was an error in mapping for 1033, instead use coordio
positionerTable = coordio.defaults.positionerTableCalib
wokCoords = ff[3].data
fiducialCoords = pandas.DataFrame(ff[4].data)
# id like to do the following, but astropy fits tables
# but i get an endianness problem (https://github.com/astropy/astropy/issues/1156)
#xyCMM = fiducialCoords[["xWok", "yWok"]].to_numpy()
xCMM = fiducialCoords.xWok.to_numpy()
yCMM = fiducialCoords.yWok.to_numpy()
xyCMM = numpy.array([xCMM, yCMM]).T

# print("fiducial coords\n\n")
# print(fiducialCoords)
# print("\n\n")
# desipolids= numpy.array([0,1,2,3,4,5,6,9,20,27,28,29,30],dtype=int)
# lcopolids= numpy.array([0,1,2,3,4,5,6,9,20,28,29],dtype=int)

# best guess at first transform from image to wok coords
SimTrans = SimilarityTransform(
    translation = numpy.array([-467.96317617, -356.55049009]),
    rotation = 0.003167701580600088,
    scale = 0.11149215438621102
)


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
    # objects["outerFIF"] = objects.ecentricity > 0.15

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


class RoughTransform(object):
    def __init__(self, objects, fiducialCoords):
        # scale pixels to mm roughly
        xCCD = objects.x.to_numpy()
        yCCD = objects.y.to_numpy()
        xWok = fiducialCoords.xWok.to_numpy()
        yWok = fiducialCoords.yWok.to_numpy()
        self.meanCCDX = numpy.mean(xCCD)
        self.meanCCDY = numpy.mean(yCCD)
        self.stdCCDX = numpy.std(xCCD)
        self.stdCCDY = numpy.std(yCCD)

        self.stdWokX = numpy.std(xWok)
        self.stdWokY = numpy.std(yWok)

        # scale to rough wok coords enough to make association
        self.roughWokX = (xCCD - self.meanCCDX) / self.stdCCDX * self.stdWokX
        self.roughWokY = (yCCD - self.meanCCDY) / self.stdCCDY * self.stdWokY

    def apply(self, xCCD, yCCD):
        wokX = (xCCD - self.meanCCDX) / self.stdCCDX * self.stdWokX
        wokY = (yCCD - self.meanCCDY) / self.stdCCDY * self.stdWokY
        return wokX, wokY


def argNearestNeighbor(xyA, xyB):
    """loop over xy list A, find nearest neighbor in list B
    return the indices in list b that match A
    """
    xyA = numpy.array(xyA)
    xyB = numpy.array(xyB)
    out = []
    distance = []
    for x, y in xyA:
        dist = numpy.sqrt((x - xyB[:, 0])**2 + (y - xyB[:, 1])**2)
        amin = numpy.argmin(dist)
        distance.append(dist[amin])
        out.append(amin)

    return numpy.array(out), numpy.array(distance)


class FullTransfrom(object):
    # use fiducials to fit this
    polids = numpy.array([0,1,2,3,4,5,6,9,20,28,29],dtype=int)

    def __init__(self, xyCCD, xyWok):
        # first fit a transrotscale model
        self.simTrans = SimilarityTransform()
        self.simTrans.estimate(xyCCD, xyWok)

        # apply the model to the data
        xySimTransFit = self.simTrans(xyCCD)

        # use zb polys to get the rest of the way
        # use leave-one out xverification to
        # estimate "unbiased" errors in fit
        self.unbiasedErrs = []
        for ii in range(len(xyCCD)):
            _xyWok = xyWok.copy()
            _xySimTransFit = xySimTransFit.copy()
            _xyWok = numpy.delete(_xyWok, ii, axis=0)
            _xySimTransFit = numpy.delete(_xySimTransFit, ii, axis=0)
            fitCheck = numpy.array(xySimTransFit[ii,:]).reshape((1,2))
            destCheck = numpy.array(xyWok[ii,:]).reshape((1,2))

            polids, coeffs = fitZhaoBurge(
                _xySimTransFit[:,0], _xySimTransFit[:,1],
                _xyWok[:,0], _xyWok[:,1], polids=self.polids
            )

            dx, dy = getZhaoBurgeXY(polids, coeffs, fitCheck[:,0], fitCheck[:,1])
            zxfit = fitCheck[:,0] + dx
            zyfit = fitCheck[:,1] + dy
            zxyfit = numpy.array([zxfit, zyfit]).T
            self.unbiasedErrs.append(destCheck.squeeze()-zxyfit.squeeze())

        self.unbiasedErrs = numpy.array(self.unbiasedErrs)
        self.unbiasedRMS = numpy.sqrt(numpy.mean(self.unbiasedErrs**2))

        # now do the "official fit", using all points

        polids, self.coeffs = fitZhaoBurge(
            xySimTransFit[:,0], xySimTransFit[:,1],
            xyWok[:,0], xyWok[:,1], polids=self.polids
        )

        dx, dy = getZhaoBurgeXY(
            polids, self.coeffs, xySimTransFit[:,0], xySimTransFit[:,1]
        )

        xWokFit = xySimTransFit[:,0] + dx
        yWokFit = xySimTransFit[:,1] + dy
        xyWokFit = numpy.array([xWokFit, yWokFit]).T
        self.errs = xyWok - xyWokFit
        self.rms = numpy.sqrt(numpy.mean(self.errs**2))

    def apply(self, xyCCD):
        # return wok xy
        xySimTransFit = self.simTrans(xyCCD)
        dx, dy = getZhaoBurgeXY(
            self.polids, self.coeffs, xySimTransFit[:,0], xySimTransFit[:,1]
        )
        xWokFit = xySimTransFit[:,0] + dx
        yWokFit = xySimTransFit[:,1] + dy
        xyWokFit = numpy.array([xWokFit, yWokFit]).T
        return xyWokFit


def getPositionerCoordinates(imgFile):
    """ commanded locations for robots """
    ff = fits.open(imgFile)
    tt = ff[-1].data
    return tt


def positionerToWok(
        positionerID, alphaDeg, betaDeg,
        xBeta=None, yBeta=None, la=None,
        alphaOffDeg=None, betaOffDeg=None,
        dx=None, dy=None #, dz=None
    ):
    posRow = positionerTable[positionerTable.positionerID == positionerID]
    assert len(posRow) == 1


    if xBeta is None:
        xBeta = posRow.metX
    if yBeta is None:
        yBeta = posRow.metY
    if la is None:
        la = posRow.alphaArmLen
    if alphaOffDeg is None:
        alphaOffDeg = posRow.alphaOffset
    if betaOffDeg is None:
        betaOffDeg = posRow.betaOffset
    if dx is None:
        dx = posRow.dx
    if dy is None:
        dy = posRow.dy
    # if dz is None:
    #     dz = posRow.dz

    xt, yt = coordio.conv.positionerToTangent(
        alphaDeg, betaDeg, xBeta, yBeta,
        la, alphaOffDeg, betaOffDeg
    )

    if hasattr(xt, "__len__"):
        zt = numpy.zeros(len(xt))
    else:
        zt = 0


    # import pdb; pdb.set_trace()
    wokRow = wokCoords[wokCoords.holeID == posRow.holeID.values[0]]

    b = numpy.array([wokRow.xWok, wokRow.yWok, wokRow.zWok])

    iHat = numpy.array([wokRow.ix, wokRow.iy, wokRow.iz])
    jHat = numpy.array([wokRow.jx, wokRow.jy, wokRow.jz])
    kHat = numpy.array([wokRow.kx, wokRow.ky, wokRow.kz])

    xw, yw, zw = coordio.conv.tangentToWok(
        xt, yt, zt, b, iHat, jHat, kHat,
        elementHeight=coordio.defaults.POSITIONER_HEIGHT, scaleFac=1,
        dx=dx, dy=dy, dz=0

    )

    return xw, yw, zw, xt, yt, b


def solveImage(imgFile, plot=False):
    # associate fiducials and fit
    # transfrom
    imgData, objects = extract(imgFile.strip())
    # first transform to rough wok xy
    # by default transfrom
    xyCCD = objects[["x", "y"]].to_numpy()
    # apply an initial guess at
    # trans/rot/scale
    xyWokRough = SimTrans(xyCCD)

    # first associate fiducials and build
    # a good transform
    argFound, distance = argNearestNeighbor(xyCMM, xyWokRough)
    # print("max fiducial distance", numpy.max(distance))
    xyFiducialCCD = xyCCD[argFound]

    ft = FullTransfrom(xyFiducialCCD, xyCMM)

    # print("rms's in microns", ft.unbiasedRMS*1000, ft.rms*1000)

    # transform all CCD detections to wok space
    xyWokMeas = ft.apply(xyCCD)

    pc = getPositionerCoordinates(imgFile)

    # import pdb; pdb.set_trace()
    # should add this to to the data table?
    # computed desiredWok positions?
    # again, to_numpy is failing due to endianness
    cmdAlpha = pc["cmdAlpha"]
    cmdBeta = pc["cmdBeta"]
    # cmdAlpha = pc["alphaReport"]
    # cmdBeta = pc["betaReport"]
    positionerID = pc["positionerID"]

    xExpectPos = []
    yExpectPos = []
    for pid, ca, cb in zip(positionerID, cmdAlpha, cmdBeta):
        xw, yw, zw, xt, yt, b = positionerToWok(
            pid, ca, cb
        )
        xExpectPos.append(xw)
        yExpectPos.append(yw)

    xyExpectPos = numpy.array([xExpectPos, yExpectPos]).T

    argFound, distance = argNearestNeighbor(xyExpectPos, xyWokMeas)
    # print("mean/max positioner distance", numpy.mean(distance), numpy.max(distance))
    xyWokRobotMeas = xyWokMeas[argFound]

    if plot:
        rg = RobotGridCalib()
        for pid, ca, cb in zip(positionerID, cmdAlpha, cmdBeta):
            xw, yw, zw, xt, yt, b = positionerToWok(
                pid, ca, cb
            )
            # import pdb; pdb.set_trace()
            r = rg.robotDict[pid]
            r.setAlphaBeta(ca, cb)
        ax = plotOne(1, robotGrid=rg, isSequence=False, returnax=True)
        for rid, r in rg.robotDict.items():
            ax.text(r.basePos[0], r.basePos[1], str(rid) + ":" + str(r.holeID))
        ax.plot(xyWokMeas[:,0], xyWokMeas[:,1], 'ok')
        plt.savefig("debug.png", dpi=350)
        plt.close()

        plt.figure()
        plt.hist(distance)
        plt.savefig("debug2.png")

    return pandas.DataFrame(
        {
            "robotID": positionerID,
            "cmdAlpha": cmdAlpha,
            "cmdBeta": cmdBeta,
            "xWokMeas": xyWokRobotMeas[:, 0],
            "yWokMeas": xyWokRobotMeas[:, 1],
            "xWokExpect": xExpectPos,
            "yWokExpect": yExpectPos
        }
    )


def compileMetrology():
    with open("metImgs.txt", "r") as f:
        metImgs = f.readlines()

    metImgs = [x.strip() for x in metImgs]

    p = Pool(cpu_count()-1)
    dfList = p.map(solveImage, metImgs)
    p.close()

    # dfList = []
    # for metImg in metImgs:
    #     print("processing img",metImg)

    #     dfList.append(solveImage(metImg, plot=True))

    dfList = pandas.concat(dfList)

    dfList.to_csv("duPontSparseMeas.csv", index=False)


def forwardModel(x, robotID, alpha, beta):
    xBeta, la, alphaOff, betaOff, dx, dy = x
    xw, yw, zw, xt, yt, b = positionerToWok(
        robotID, alpha, beta,
        xBeta=xBeta, la=la,
        alphaOffDeg=alphaOff, betaOffDeg=betaOff,
        dx=dx, dy=dy
    )
    return xw, yw


def minimizeMe(x, robotID, alpha, beta, xWok, yWok):
    xw, yw = forwardModel(x, robotID, alpha, beta)
    return numpy.sum((xw-xWok)**2 + (yw-yWok)**2)


def fitOneRobot(rID):
    df = pandas.read_csv("duPontSparseMeas.csv")
    dfR = df[df.robotID==rID]
    xExpect = dfR.xWokExpect.to_numpy()
    yExpect = dfR.yWokExpect.to_numpy()
    xMeas = dfR.xWokMeas.to_numpy()
    yMeas = dfR.yWokMeas.to_numpy()

    #minimize
    x0 = numpy.array([
        defaults.MET_BETA_XY[0], defaults.ALPHA_LEN,
        0, 0, 0, 0
    ])
    args = (rID, dfR.cmdAlpha, dfR.cmdBeta, xMeas, yMeas)
    tstart = time.time()
    out = minimize(minimizeMe, x0, args, method="Powell")
    # print(out.x - x0)
    # print(rID, "alpha arm len", out.x[1])
    tend = time.time()
    # print("took %.2f"%(tend-tstart))

    xFit, yFit = forwardModel(out.x, rID, dfR.cmdAlpha, dfR.cmdBeta)


    dx = (xMeas-xFit)
    dy = (yMeas-yFit)
    umRMS = numpy.sqrt(numpy.mean(dx**2+dy**2))*1000
    return out.x, umRMS

def fitMetrology():
    # loop version
    df = pandas.read_csv("duPontSparseMeas.csv")
    robotIDs = set(df.robotID)
    calibs = []
    for rID in robotIDs:
        calibs.append(fitOneRobot(rID))

def fitMetrology2():
    # loop version
    df = pandas.read_csv("duPontSparseMeas.csv")
    robotIDs = set(df.robotID)
    p = Pool(cpu_count()-1)
    out = p.map(fitOneRobot, robotIDs)
    for robotID, (x, rms) in zip(robotIDs, out):
        print(robotID, rms)
    p.close()


compileMetrology()
fitMetrology2()

"""
process:

"""
