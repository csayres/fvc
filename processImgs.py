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
import coordio
from coordio import defaults
import time
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
from functools import partial
import seaborn as sns


# get default tables, any image will do, they're all the same
# ff = fits.open(glob.glob("duPontSparseImg/59499/proc*.fits")[10])
# positionerTable = ff[2].data  # was an error in mapping for 1033, instead use coordio
positionerTable = coordio.defaults.calibration.positionerTable
wokCoords = coordio.defaults.calibration.wokCoords
fiducialCoords = coordio.defaults.calibration.fiducialCoords

posWokCoords = pandas.merge(positionerTable, wokCoords, on="holeID").reset_index()

# wokCoords = ff[3].data
# fiducialCoords = pandas.DataFrame(ff[4].data)
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

utahBasePath = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fps/engineering/osu/forConor/LVC"
laptopBasePath = "duPontSparseImg"

def organize(basePath):
    allImgFiles = glob.glob(basePath + "/59499/proc*.fits")
    metImgFiles = []
    apMetImgFiles = []
    badFiles = []

    meanImg = None

    for ii, f in enumerate(allImgFiles):
        imgNumber = int(f.split(".")[-2].split("-")[-1])
        if imgNumber < 142:
            continue

        print("%s: %i of %i"%(f, ii, len(allImgFiles)))
        g = fits.open(f)

        maxCounts = numpy.max(g[1].data)
        if maxCounts > 62000:
            os.remove(f)
            continue

        if meanImg is None:
            meanImg = g[1].data
        else:
            meanImg += g[1].data

        if g[1].header["LED2"] == 0:
            metImgFiles.append(f)
        else:
            apMetImgFiles.append(f)

    print("nimgs", len(metImgFiles), len(apMetImgFiles))

    medianImg = meanImg / (len(metImgFiles)+len(apMetImgFiles))
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


# def extract(imgFile):
#     imgData = fitsio.read(imgFile)
#     imgData = numpy.array(imgData, dtype="float")
#     bkg = sep.Background(imgData)
#     bkg_image = bkg.back()
#     data_sub = imgData - bkg_image
#     objects = sep.extract(data_sub, 3.5, err=bkg.globalrms)
#     objects = pandas.DataFrame(objects)

#     # ecentricity
#     objects["ecentricity"] = 1 - objects["b"] / objects["a"]

#     # slope of ellipse (optical distortion direction)
#     objects["slope"] = numpy.tan(objects["theta"] + numpy.pi/2) # rotate by 90
#     # intercept of optical distortion direction
#     objects["intercept"] = objects["y"] - objects["slope"] * objects["x"]

#     objects = objects[objects["npix"] > 100]

#     # filter on most eliptic, this is an assumption!!!!
#     # objects["outerFIF"] = objects.ecentricity > 0.15

#     return imgData, objects


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
    # def __init__(self, centroids, expectedTargCoords):
    def __init__(self, xyCCD, xyWok):
        # scale pixels to mm roughly
        xCCD = xyCCD[:,0]
        yCCD = xyCCD[:,1]
        xWok = xyWok[:,0]
        yWok = xyWok[:,1]


        self.meanCCDX = numpy.mean(xCCD)
        self.meanCCDY = numpy.mean(yCCD)
        self.stdCCDX = numpy.std(xCCD)
        self.stdCCDY = numpy.std(yCCD)

        self.stdWokX = numpy.std(xWok)
        self.stdWokY = numpy.std(yWok)

        # scale to rough wok coords enough to make association


    def apply(self, xyCCD):
        xCCD = xyCCD[:,0]
        yCCD = xyCCD[:,1]
        roughWokX = (xCCD - self.meanCCDX) / self.stdCCDX * self.stdWokX
        roughWokY = (yCCD - self.meanCCDY) / self.stdCCDY * self.stdWokY
        return numpy.array([roughWokX, roughWokY]).T
        # return self.simTrans(xyCCD)
        # xCCD = xyCCD[:,0]
        # yCCD = xyCCD[:,1]
        # wokX = (xCCD - self.meanCCDX) / self.stdCCDX * self.stdWokX
        # wokY = (yCCD - self.meanCCDY) / self.stdCCDY * self.stdWokY
        # return numpy.array([wokX, wokY]).T


class FullTransform(object):
    # use fiducials to fit this
    # polids = numpy.arange(12, dtype=int)
    polids = numpy.array([0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],dtype=int) # lco
    # polids = numpy.array([0,1,2,3,4,5,6,9,20,27,28,29,30],dtype=int) # desi terms
    def __init__(self, xyCCD, xyWok):
        # first fit a transrotscale model
        # self.simTrans = SimilarityTransform()
        self.simTrans = AffineTransform()
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

    def apply(self, xyCCD, zb=True):
        # return wok xy
        xySimTransFit = self.simTrans(xyCCD)

        if zb:
            dx, dy = getZhaoBurgeXY(
                self.polids, self.coeffs, xySimTransFit[:,0], xySimTransFit[:,1]
            )
            xWokFit = xySimTransFit[:,0] + dx
            yWokFit = xySimTransFit[:,1] + dy
            xyWokFit = numpy.array([xWokFit, yWokFit]).T
        else:
            xyWokFit = xySimTransFit
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

    posRow = posWokCoords[posWokCoords.positionerID == positionerID]
    # print(posRow)
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
    # wokRow = wokCoords[wokCoords.holeID == posRow.holeID.values[0]]

    b = numpy.array([posRow.xWok, posRow.yWok, posRow.zWok])

    iHat = numpy.array([posRow.ix, posRow.iy, posRow.iz])
    jHat = numpy.array([posRow.jx, posRow.jy, posRow.jz])
    kHat = numpy.array([posRow.kx, posRow.ky, posRow.kz])

    xw, yw, zw = coordio.conv.tangentToWok(
        xt, yt, zt, b, iHat, jHat, kHat,
        elementHeight=coordio.defaults.POSITIONER_HEIGHT, scaleFac=1,
        dx=dx, dy=dy, dz=0

    )

    return xw, yw, zw, xt, yt, b


def solveImage(imgFile, guessTransform, zb=True, plot=False):
    global xyCMM
    print("using %i points for fiducial transform"%len(xyCMM))
    print("imgFile", imgFile)
    # associate fiducials and fit
    # transfrom
    imgData = fitsio.read(imgFile)
    objects = extract(imgData)
    # print("found", len(objects), "in", imgFile)
    # first transform to rough wok xy
    # by default transfrom
    xyCCD = objects[["x", "y"]].to_numpy()
    # apply an initial guess at
    # trans/rot/scale
    xyWokRough = guessTransform.apply(xyCCD)

    # first associate fiducials and build
    # a good transform
    argFound, distance = argNearestNeighbor(xyCMM, xyWokRough)
    # print("max fiducial distance", numpy.max(distance))
    xyFiducialCCD = xyCCD[argFound]

    ft = FullTransform(xyFiducialCCD, xyCMM)

    print("rms's in microns unbias, bias", ft.unbiasedRMS*1000, ft.rms*1000)

    # transform all CCD detections to wok space
    xyWokMeas = ft.apply(xyCCD, zb=zb)
    xyFiducialMeas = ft.apply(xyFiducialCCD, zb=zb)

    pc = getPositionerCoordinates(imgFile)

    # import pdb; pdb.set_trace()
    # should add this to to the data table?
    # computed desiredWok positions?
    # again, to_numpy is failing due to endianness
    cmdAlpha = pc["cmdAlpha"]
    cmdBeta = pc["cmdBeta"]
    # cmdAlpha = pc["alphaReport"]
    # cmdBeta = pc["betaReport"]
    # print("\n\n", cmdAlpha, "\n\n")
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

    plt.figure(figsize=(8,8))
    plt.title("full transform")
    plt.plot(xyWokMeas[:,0], xyWokMeas[:,1], 'o', ms=4, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1, label="centroid")
    plt.plot(xExpectPos, yExpectPos, 'xk', ms=3, label="expected met")
    # overplot fiducials
    plt.plot(xCMM, yCMM, "D", ms=6, markerfacecolor="None", markeredgecolor="cornflowerblue", markeredgewidth=1, label="expected fid")
    plt.axis("equal")
    plt.legend()
    plt.xlim([-350, 350])
    plt.ylim([-350,350])
    plt.savefig("%sfull.png"%imgFile, dpi=350)
    plt.close()

    if plot:
        # plot the fiducial fit quiver and
        # histogram
        plt.figure()
        plt.title("Fiducial Fit residuals")
        dx = xyCMM[:,0] - xyFiducialMeas[:,0]
        dy = xyCMM[:,1] - xyFiducialMeas[:,1]
        plt.quiver(xyFiducialMeas[:,0], xyFiducialMeas[:,1], dx, dy)
        plt.savefig("%s.fiducialquiver.png"%imgFile.strip(), dpi=250)
        plt.xlabel("wok x (mm)")
        plt.ylabel("wok y (mm)")
        plt.close()

        plt.figure()
        plt.hist(numpy.sqrt(dx**2+dy**2)*1000)
        plt.xlabel("fiducial fit error (um)")
        plt.savefig("%s.fiducialerror.png"%imgFile.strip(), dpi=250)
        plt.close()


        # print("im trying to plot you")
        from kaiju import RobotGridCalib
        from kaiju.utils import plotOne
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
        plt.savefig("%s.viz.png"%imgFile.strip(), dpi=350)
        plt.close()

        plt.figure()
        plt.hist(distance)
        plt.savefig("debug2.png")
        plt.close()

    return pandas.DataFrame(
        {
            "robotID": positionerID,
            "cmdAlpha": cmdAlpha,
            "cmdBeta": cmdBeta,
            "xWokMeas": xyWokRobotMeas[:, 0],
            "yWokMeas": xyWokRobotMeas[:, 1],
            "xWokExpect": xExpectPos,
            "yWokExpect": yExpectPos,
            "imgFile": [imgFile]*len(positionerID),
        }
    )


def compileMetrology(multiprocess=True, plot=False):
    with open("metImgs.txt", "r") as f:
        metImgs = f.readlines()

    metImgs = [x.strip() for x in metImgs]

    si = partial(solveImage, plot=plot)
    if multiprocess:
        p = Pool(cpu_count()-1)
        dfList = p.map(si, metImgs)
        p.close()

    else:
        dfList = []
        for metImg in metImgs:
            dfList.append(solveImage(metImg, plot=plot))

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
    # clip outliers?
    dist = numpy.sqrt((xw-xWok)**2 + (yw-yWok)**2)

    # meandist = numpy.mean(dist)
    # stddist = numpy.std(dist)
    # keepdist = dist[dist < meandist*5*stddist]
    # return numpy.sum(keepdist)

    return numpy.sum(dist)


def fitOneRobot(rID, csvFile):
    # turn this into an object w/ attrs
    # !!!!!!!!! warning hack!!!!
    # df = pandas.read_csv("apoSafe20111126.csv")
    df = pandas.read_csv(csvFile)
    dfR = df[df.robotID==rID]
    xExpect = dfR.xWokExpect.to_numpy()
    yExpect = dfR.yWokExpect.to_numpy()
    xMeas = dfR.xWokMeas.to_numpy()
    yMeas = dfR.yWokMeas.to_numpy()

    # print("len alphas", len(dfR.cmdAlpha))
    # plt.hist(dfR.cmdAlpha)
    # plt.show()
    # import pdb; pdb.set_trace()

    #minimize
    x0 = numpy.array([
        defaults.MET_BETA_XY[0], defaults.ALPHA_LEN,
        0, 0, 0, 0
    ])
    args = (rID, dfR.cmdAlpha, dfR.cmdBeta, xMeas, yMeas)
    tstart = time.time()
    out = minimize(minimizeMe, x0, args, method="Powell")
    # print(out.x - x0)
    print("minimize result", rID, out.success)
    tend = time.time()
    # print("took %.2f"%(tend-tstart))

    xFit, yFit = forwardModel(out.x, rID, dfR.cmdAlpha, dfR.cmdBeta)

    # where we would be with zero calib
    xNC, yNC = forwardModel(x0, rID, dfR.cmdAlpha, dfR.cmdBeta)

    return out.x, xMeas, yMeas, xFit, yFit, dfR.imgFile, xNC, yNC


# def fitMetrology():
#     # loop version
#     df = pandas.read_csv("duPontSparseMeas.csv")
#     robotIDs = set(df.robotID)
#     calibs = []
#     for rID in robotIDs:
#         calibs.append(fitOneRobot(rID))


def calibrateRobots(inFile, outfileName, calibResultsFileName):
    # multiprocess version
    df = pandas.read_csv(inFile)

    robotIDs = set(df.robotID)

    _fitOneRobot = partial(fitOneRobot, csvFile=inFile)

    # out = []
    # for robotID in robotIDs:
    #     out.append(fitOneRobot(robotID))

    p = Pool(cpu_count()-1)
    out = p.map(_fitOneRobot, robotIDs)
    p.close()

    xm = []
    ym = []
    _dx = []
    _dy = []
    _umErr = []
    _robots = []
    _dxNC = []
    _dyNC = []

    npt = positionerTable.copy().reset_index()

    # convert datatypes
    npt["metY"] = npt["metY"].astype(float)
    npt["alphaOffset"] = npt["alphaOffset"].astype(float)
    npt["betaOffset"] = npt["betaOffset"].astype(float)
    npt["dx"] = npt["dx"].astype(float)
    npt["dy"] = npt["dy"].astype(float)

    for robotID, (x, xMeas, yMeas, xFit, yFit, imgFile, xNC, yNC) in zip(robotIDs, out):
        # update calibration table
        xBeta, alphaArmLen, alphaOff, betaOff, xOff, yOff = x
        print("xOff", xOff)
        idx = list(npt.positionerID).index(robotID)
        npt.at[idx, "alphaArmLen"] = alphaArmLen
        npt.at[idx, "metX"] = xBeta
        npt.at[idx, "alphaOffset"] = alphaOff
        npt.at[idx, "betaOffset"] = betaOff
        npt.at[idx, "dx"] = xOff
        npt.at[idx, "dy"] = yOff

        print("n points", len(xMeas))
        imgFile = list(imgFile)
        dx = (xFit - xMeas)
        dy = (yFit - yMeas)
        dxNC = (xNC - xMeas)
        dyNC = (yNC - yMeas)
        _dxNC.append(dxNC)
        _dyNC.append(dyNC)
        errs = numpy.sqrt(dx**2 + dy**2)
        xm.append(xMeas)
        ym.append(yMeas)
        _dx.append(dx)
        _dy.append(dy)
        _umErr.append(errs*1000)
        _robots.append([robotID]*len(dx))

        # meanErr = numpy.mean(errs)*1000
        # umRMS = numpy.sqrt(numpy.mean(dx**2+dy**2))*1000
        # print("robotID umRMS", robotID, umRMS)
        # plt.figure()
        # plt.title(str(robotID))
        # plt.hist(errs*1000)
        # plt.show()

    # import pdb; pdb.set_trace()

    # drop unnamed columns
    # npt = npt.loc[:, ~npt.columns.str.contains('^Unnamed')]
    npt.to_csv(outfileName)

    xm = numpy.array(xm).flatten()
    ym = numpy.array(ym).flatten()
    _dx = numpy.array(_dx).flatten()
    _dy = numpy.array(_dy).flatten()
    _umErr = numpy.array(_umErr).flatten()
    _robots = numpy.array(_robots).flatten()
    _dxNC = numpy.array(_dxNC).flatten()
    _dyNC = numpy.array(_dyNC).flatten()

    df = pandas.DataFrame({
        "xMeas": xm,
        "yMeas": ym,
        "dx": _dx,
        "dy": _dy,
        "dxNC": _dxNC,
        "dyNC": _dyNC,
        "umErr": _umErr,
        "robotID": _robots
    })

    df.to_csv(calibResultsFileName)


def plotCalibResults(calibResultsFile, removeOutliers=False):
    df = pandas.read_csv(calibResultsFile)

    # plot non calibrated quiver

    # detect outliers > 1mm?
    if removeOutliers:
        df = df[df.umErr < 200]

    plt.figure(figsize=(9, 9))
    plt.title("uncalibrated robot")
    Q = plt.quiver(df.xMeas, df.yMeas, df.dxNC, df.dyNC, scale=20, angles="xy")
    plt.quiverkey(Q, X=0.3, Y=1.1, U=2,
                 label='Quiver key, length = 2 mm', labelpos='E')
    plt.savefig("quiverFitNC.png", dpi=350)
    plt.xlabel("wok x (mm)")
    plt.ylabel("wok y (mm)")


    plt.figure(figsize=(9, 9))
    plt.title("calibrated robot")
    Q = plt.quiver(df.xMeas, df.yMeas, df.dx, df.dy, scale=2, angles="xy")

    plt.quiverkey(Q, X=0.3, Y=1.1, U=0.010,
                 label='Quiver key, length = 10 microns', labelpos='E')
    plt.xlabel("wok x (mm)")
    plt.ylabel("wok y (mm)")
    plt.savefig("quiverFit.png", dpi=350)


    plt.figure(figsize=(10, 5))
    plt.title("Calibrated Blind Move Error")
    sns.boxplot(x="robotID", y="umErr", data=df)
    plt.xticks(rotation=45)
    # plt.ylim([0,300])
    # plt.savefig("group%i.png"%ii, dpi=350)
    # plt.close()

    # for ii, _df in enumerate(numpy.split(df,20)):
    #     plt.figure(figsize=(10, 5))
    #     plt.title("Calibrated Blind Move Error")
    #     sns.boxplot(x="robotID", y="umErr", data=_df)
    #     plt.xticks(rotation=45)
    #     plt.ylim([0,300])
    #     plt.savefig("group%i.png"%ii, dpi=350)
    #     plt.close()
    # plt.savefig("fitStatsZoom.png", dpi=250)



def medianCombine(imgFiles, doMean=False):
    # imgFiles = glob.glob("apoFullImg/59518/proc*.fits")
    # f = fits.open(imgFiles[0])
    d = fitsio.read(imgFiles[0])
    s = d.shape
    print("image shape", s)
    nPix = len(d.flatten())
    nImg = len(imgFiles)
    medArr = numpy.zeros((nImg,nPix))
    for ii, imgFile in enumerate(imgFiles):
        d = fitsio.read(imgFile)
        medArr[ii,:] = d.flatten()

    if doMean:
        medArr = numpy.mean(medArr, axis=0)
    else:
        medArr = numpy.median(medArr, axis=0)

    medArr = medArr.reshape(s)

    # plt.figure()
    # plt.imshow(equalize_hist(d), origin="lower")
    # plt.xlim([10,20])
    # plt.ylim([10,20])

    # plt.figure()
    # plt.imshow(equalize_hist(d.flatten().reshape(s)))
    # plt.xlim([10,20])
    # plt.ylim([10,20])
    # plt.show()

    hdu = fits.PrimaryHDU(medArr)
    hdu.writeto("medianImg.fits", overwrite=True)
    # fitsio.write("medianImg.fits", medArr)


def extract(imgData):
    # run source extractor,
    # do some filtering
    # return the extracted centroids
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

    # ignore everything less than 100 pixels
    # something changed when rick bumped things?
    objects = objects[objects["npix"] > 100]


    print("got", len(objects), "centroids")
    print("expected", len(positionerTable)+len(fiducialCoords), "centroids")
    # filter on most eliptic, this is an assumption!!!!
    # objects["outerFIF"] = objects.ecentricity > 0.15

    return objects

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

def initialFiducialTransform(keepThresh=None, fiducialTableFilename=None):
    # saved from earlier
    global xyCMM
    imgData = fitsio.read("medianImg.fits")
    # plt.figure()
    # plt.imshow(equalize_hist(imgData), origin="lower")
    # plt.show()
    centroids = extract(imgData)
    centroids = centroids[centroids.npix > 2000]
    print("n centroids", len(centroids))
    print("n cmm", len(xCMM))
    print("total fiducials: ", 60)

    xyCCD = centroids[["x", "y"]].to_numpy()

    rt = RoughTransform(xyCCD, xyCMM)
    xyWokRough = rt.apply(xyCCD)

    # # remove fiducials at large radius
    # keep = numpy.linalg.norm(xyCMM, axis=1) < 310
    # xyCMM = xyCMM[keep,:]

    argFound, dist = argNearestNeighbor(xyCMM, xyWokRough)
    xyFiducialCCD = xyCCD[argFound]
    xyFiducialWokRough = xyWokRough[argFound]


    plt.figure(figsize=(8,8))
    plt.title("rough fiducial association")
    plt.plot(xyWokRough[:,0], xyWokRough[:,1], 'o', ms=4, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1, label="centroid")
    # plt.plot(expectedTargCoords.xWokMetExpect.to_numpy(), expectedTargCoords.yWokMetExpect.to_numpy(), 'xk', ms=3, label="expected met")
    # overplot fiducials
    plt.plot(xCMM, yCMM, "D", ms=6, markerfacecolor="None", markeredgecolor="cornflowerblue", markeredgewidth=1, label="expected fid")
    for cmm, rough in zip(xyCMM, xyFiducialWokRough):
        plt.plot([cmm[0], rough[0]], [cmm[1], rough[1]], "-k")
    plt.axis("equal")
    plt.legend()
    plt.savefig("roughassoc.png", dpi=350)

    firstGuessTF = FullTransform(xyFiducialCCD, xyCMM)
    print("full trans 1 bias, unbias", firstGuessTF.rms*1000, firstGuessTF.unbiasedRMS*1000)
    xyWokMeas = firstGuessTF.apply(xyFiducialCCD, zb=True)


    plt.figure(figsize=(8,8))
    plt.title("full transform 1")
    plt.plot(xyWokMeas[:,0], xyWokMeas[:,1], 'o', ms=4, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1, label="centroid")
    #plt.plot(expectedTargCoords.xWokMetExpect.to_numpy(), expectedTargCoords.yWokMetExpect.to_numpy(), 'xk', ms=3, label="expected met")
    # overplot fiducials
    plt.plot(xCMM, yCMM, "D", ms=6, markerfacecolor="None", markeredgecolor="cornflowerblue", markeredgewidth=1, label="expected fid")
    plt.axis("equal")
    plt.title("outlier include")
    plt.legend()
    plt.xlim([-350, 350])
    plt.ylim([-350,350])
    plt.savefig("full.png", dpi=350)

    dxy = xyCMM - xyWokMeas

    plt.figure()
    plt.hist(numpy.linalg.norm(dxy,axis=1)*1000)
    plt.savefig("fiducialhist.png", dpi=350)

    plt.figure()
    plt.quiver(xyCMM[:,0], xyCMM[:,1], dxy[:,0], dxy[:,1], angles="xy")
    plt.axis("equal")
    plt.title("outlier include")
    plt.savefig("fiducialquiver.png", dpi=350)

    # plt.show()

    # ignor errors > 100 micron and refit

    if keepThresh is not None:
        keep = numpy.linalg.norm(dxy, axis=1)*1000 < keepThresh
        ft = FullTransform(xyFiducialCCD[keep,:], xyCMM[keep,:])
        print("full trans 2 bias, unbias", ft.rms*1000, ft.unbiasedRMS*1000)
        xyCMM = xyCMM[keep,:]
        xyWokMeas = ft.apply(xyFiducialCCD[keep,:], zb=True)

    # apply this transform to all found fiducials, create a new table
    plt.figure(figsize=(8,8))
    plt.title("full transform 1")
    plt.plot(xyWokMeas[:,0], xyWokMeas[:,1], 'o', ms=4, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1, label="centroid")
    #plt.plot(expectedTargCoords.xWokMetExpect.to_numpy(), expectedTargCoords.yWokMetExpect.to_numpy(), 'xk', ms=3, label="expected met")
    # overplot fiducials
    plt.plot(xCMM, yCMM, "D", ms=6, markerfacecolor="None", markeredgecolor="cornflowerblue", markeredgewidth=1, label="expected fid")
    plt.title("outlier remove")
    plt.axis("equal")
    plt.legend()
    plt.xlim([-350, 350])
    plt.ylim([-350,350])

    dxy = xyCMM - xyWokMeas

    plt.figure()
    plt.hist(numpy.linalg.norm(dxy,axis=1)*1000)

    plt.figure()
    plt.quiver(xyCMM[:,0], xyCMM[:,1], dxy[:,0], dxy[:,1], angles="xy")
    plt.title("outlier remove")
    plt.axis("equal")

    # import pdb; pdb.set_trace()
    # finally write a new fiducial table if indicated
    if fiducialTableFilename is not None:
        # convert all CCD detections and assume they're fiducials
        assert len(xyCCD) == 60 # there should be 60 fiducials (including GFAs)
        xyWokMeas = ft.apply(xyCCD, zb=True)
        nfw = fiducialCoords.copy().reset_index()

        xyWokOld = nfw[["xWok", "yWok"]].to_numpy()

        thetaMeas = numpy.degrees(numpy.arctan2(xyWokMeas[:,1], xyWokMeas[:,0]))
        thetaMeas = thetaMeas % 360


        argFound, distance = argNearestNeighbor(xyWokOld, xyWokMeas)
        xyNew = xyWokMeas[argFound, :]

        nfw["xWok"] = xyNew[:,0]
        nfw["yWok"] = xyNew[:,1]

        # next find fiducials not in table (these will be GFAs)
        indNotFound = list(set(list(range(60))) - set(argFound))
        # in order of increasing theta
        missingFiducialNames = ["F9", "F8", "F6", "F5", "F3", "F2", "F18", "F17", "F15", "F14", "F12", "F11"]

        outerFMeas = xyWokMeas[indNotFound, :]
        thetaMeas = numpy.degrees(numpy.arctan2(outerFMeas[:,1], outerFMeas[:,0]))
        thetaMeas = thetaMeas % 360
        asort = numpy.argsort(thetaMeas)
        outerFMeas = outerFMeas[asort,:]

        newFdf = pandas.DataFrame({
            "site": ["APO"] * len(asort),
            "id": ["F?"] * len(asort),
            "xWok": outerFMeas[:,0],
            "yWok": outerFMeas[:,1],
            "zWok": [143.1] * len(asort),
            "holeID": missingFiducialNames,
            "col": [numpy.nan] * len(asort),
            "row": [numpy.nan] * len(asort)
        })

        nfw = pandas.concat([nfw, newFdf], ignore_index=True)
        # nfw = nfw.reset_index()


        plt.figure()
        text = nfw.holeID
        for ii, (xx, yy) in enumerate(nfw[["xWok", "yWok"]].to_numpy()):
            plt.text(xx,yy,text[ii])
        plt.plot(0,0,'xr')
        plt.xlim([-350, 350])
        plt.ylim([-350, 350])

        nfw.to_csv(fiducialTableFilename)






    # fiducial fit
    # d = {}
    # d["xCMM"] = xyCMM[:,0]
    # d["yCMM"] = xyCMM[:,1]
    # d["xMeas"] = xyWokMeas[:,0]
    # d["yMeas"] = xyWokMeas[:,1]
    # d["dx"] = xyCMM[:,0] - xyWokMeas[:,0]
    # d["dy"] = xyCMM[:,1] - xyWokMeas[:,1]
    # d["keepThresh"] = [keepThresh]*len(xyCMM[:,0])
    # df = pandas.DataFrame(d)
    # df.to_csv("fiducialFit.csv")


    # dxy = xyCMM - xyWokMeas

    # plt.figure()
    # plt.hist(numpy.linalg.norm(dxy,axis=1)*1000)

    # plt.figure()
    # plt.quiver(xyCMM[:,0], xyCMM[:,1], dxy[:,0], dxy[:,1], angles="xy")
    # plt.axis("equal")
    # plt.show()


    # dfList = []


    # for img in sorted(glob.glob("apoFullImg/59520/proc*.fits")): # full range calib
    # # for img in glob.glob("apoFullImg/59518/proc*.fits"): # safe image calib
    #     print("on img", img)
    #     dfList.append(solveImage(img, firstGuessTF, zb=True, plot=False))

    # dfList = pandas.concat(dfList)

    # dfList.to_csv("apoFullMeasNotSafe.csv", index=False)


# return firstGuessTF
def generateFullMeasTable(procImgList):
    dfList = []
    for imgName in procImgList:

        f = fitsio.FITS(imgName)

        # import pdb; pdb.set_trace()

        # f = fits.open(imgName)
        measArray = f[5].read()

        keep = measArray["fibre_type"] == "Metrology"

        dfMeas = pandas.DataFrame(
            {
            "robotID": measArray["positioner_id"].byteswap().newbyteorder()[keep],
            "holeID": measArray["hole_id"].byteswap().newbyteorder()[keep],
            "xWokMeas": measArray["xwok_measured"].byteswap().newbyteorder()[keep],
            "yWokMeas": measArray["ywok_measured"].byteswap().newbyteorder()[keep],
            "xWokExpect": measArray["xwok"].byteswap().newbyteorder()[keep],
            "yWokExpect": measArray["ywok"].byteswap().newbyteorder()[keep],
            "fibre_type": measArray["fibre_type"].byteswap().newbyteorder()[keep]
            }
        )

        dfMeas["imgFile"] = [imgName] * len(dfMeas)
        dfMeas.reset_index(inplace=True)

        cmdArray = f[6].read()
        dfCmd = pandas.DataFrame({
            "robotID": cmdArray["positionerID"].byteswap().newbyteorder(),
            "cmdAlpha": cmdArray["alphaReport"].byteswap().newbyteorder(),
            "cmdBeta": cmdArray["betaReport"].byteswap().newbyteorder(),

        })


        dfOut = dfMeas.merge(dfCmd, on="robotID")

        dfOut = dfOut.drop(columns="index")
        dfList.append(dfOut)

    return pandas.concat(dfList).reset_index()

        # import pdb; pdb.set_trace()



    #         return pandas.DataFrame(
    #     {
    #         "robotID": positionerID,
    #         "cmdAlpha": cmdAlpha,
    #         "cmdBeta": cmdBeta,
    #         "xWokMeas": xyWokRobotMeas[:, 0],
    #         "yWokMeas": xyWokRobotMeas[:, 1],
    #         "xWokExpect": xExpectPos,
    #         "yWokExpect": yExpectPos,
    #         "imgFile": [imgFile]*len(positionerID),
    #     }
    # )

def plotPositionerTable(infile):
    df = pandas.read_csv(infile)
    df = pandas.merge(df, wokCoords, on="holeID").reset_index()

    mdx = numpy.mean(df.dx)
    mdy = numpy.mean(df.dy)

    plt.figure()
    plt.hist(df.dx)
    plt.title("dx")

    plt.figure()
    plt.hist(df.dy)
    plt.title("dy")

    plt.figure()
    plt.hist(df.dx - mdx)
    plt.title("dx - <dx>")

    plt.figure()
    plt.hist(df.dy - mdy)
    plt.title("dy - <dy>")

    plt.figure()
    plt.quiver(df.xWok, df.yWok, df.dx, df.dy, angles="xy")
    plt.title("positioner base offsets")
    plt.xlabel("x wok")
    plt.ylabel("y wok")

    plt.figure()
    plt.quiver(df.xWok, df.yWok, df.dx - mdx, df.dy - mdy, angles="xy")
    plt.title("mean subtracted positioner base offsets")
    plt.xlabel("x wok")
    plt.ylabel("y wok")


def safeMoveAnalysis(recompute=True):
    # from 2021-11-26 wok in plug lab

    imgRange = [2, 81]
    imgPaths = []
    for imgNum in range(imgRange[0], imgRange[1]+1):
        zf = ("%i"%imgNum).zfill(4)
        imgPath = "/Volumes/futa/apo/data/fcam/59544/proc-fimg-fvc2n-%s.fits"%zf
        imgPaths.append(imgPath)


    # determine median-combined fiducial locations


    if recompute:
        # tstart = time.time()
        # medianCombine(imgPaths, doMean=False)
        # print("median combine took %.1f"%(time.time()-tstart))
        fullMeasTable = generateFullMeasTable(imgPaths)
        fullMeasTable.to_csv("apoSafe20111127.csv")
        calibrateRobots(inFile="apoSafe20111127.csv", outfileName="positionerTableSafe20111127.csv", calibResultsFileName="calibSafe20111127.csv")



    # plot median img
    # medImg = fitsio.read("medianImg.fits")
    # plt.figure()
    # plt.imshow(medImg, origin="lower")
    # plt.figure()
    # plt.imshow(equalize_hist(medImg), origin="lower")

    # calculate new fiducial locations
    # initialFiducialTransform(keepThresh=120, fiducialTableFilename="fiducialCoords20111127.csv")  # throw out 120 micron outliers

    plotCalibResults("calibSafe20111127.csv")

    plotPositionerTable("positionerTableSafe20111127.csv")

    plt.show()


def dangerMoveAnalysis(recompute=True):
    # from 2021-11-28 wok in plug lab

    imgRange = [2, 233]
    imgPaths = []
    for imgNum in range(imgRange[0], imgRange[1]+1):
        # image 7 doesn't exist
        if imgNum == 7:
            continue
        zf = ("%i"%imgNum).zfill(4)
        imgPath = "/Volumes/futa/apo/data/fcam/59546/proc-fimg-fvc2n-%s.fits"%zf
        imgPaths.append(imgPath)


    # determine median-combined fiducial locations


    if recompute:

        fullMeasTable = generateFullMeasTable(imgPaths)
        fullMeasTable.to_csv("apoDanger20111128.csv")
        calibrateRobots(inFile="apoDanger20111128.csv", outfileName="positionerTableDanger20111128.csv", calibResultsFileName="calibDanger20111128.csv")

    # calculate new fiducial locations
    # initialFiducialTransform(keepThresh=120, fiducialTableFilename="fiducialCoords20111127.csv")  # throw out 120 micron outliers

    plotCalibResults("calibDanger20111128.csv")

    plotPositionerTable("positionerTableDanger20111128.csv")

    plt.show()


if __name__ == "__main__":
    # be sure to set the right calibration environment before running!

    # safeMoveAnalysis(True)
    dangerMoveAnalysis(False)

    ########### old
    # medianCombine()
    # initialFiducialTransform()
    # medianCombine()
    #organize(utahBasePath)
    # compileMetrology(multiprocess=True, plot=False) # plot isn't working?

    # organize(laptopBasePath) # run this only for duPont, produces medianImg.fits for duPontSparse
    # medianCombine() # produces medianImg.fits for apoFull

    # initialFiducialTransform()

    # calibrateRobots(inFile="apoFullMeasNotSafe.csv", outfileName="positionerTableAPONotSafe.csv") # use this for apoFull
    # calibrateRobots() # use this for duPont sparse

    # plotCalibResults(removeOutliers=False)
    # plt.show()

"""
process:

"""
