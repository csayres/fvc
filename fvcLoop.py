import asyncio
import os
import numpy
numpy.random.seed(0)
import datetime
from astropy.io import fits
import pandas as pd
import pickle
import datetime
import time

from jaeger import FPS, log
from jaeger.commands.trajectory import send_trajectory
from jaeger.exceptions import FPSLockedError, TrajectoryError
# log.sh.setLevel(5)
from kaiju.robotGrid import RobotGridCalib
from coordio.defaults import positionerTable, wokCoords, fiducialCoords
# from baslerCam import BaslerCamera, BaslerCameraSystem, config
import sep
from skimage.transform import SimilarityTransform, AffineTransform
from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY
import matplotlib.pyplot as plt

# from processImgs import SimTrans, argNearestNeighbor, FullTransform

# Speed = 3               # RPM at output, breakout as kaiju param?
# smoothPts = 5           # width of velocity smoothing window, breakout as kaiju param?
# collisionShrink = 0.05  # amount to decrease collisionBuffer by when checking smoothed and simplified paths

angStep = 0.1         # degrees per step in kaiju's rough path
epsilon = angStep * 2   # max error (deg) allowed in kaiju's path simplification
collisionBuffer = 2.4    # effective *radius* of beta arm in mm effective beta arm width is 2*collisionBuffer
exptime = 1.6
EXPLODEFIRST = False
UNWINDONLY = False
FORCEUNWIND = False
LED12_VALUE = 1  # metrology
LED3_VALUE = 16  # boss
LED4_VALUE = 3  # apogee

SEED = 120
escapeDeg = 20  # 20 degrees of motion to escape
use_sync_line = False
NITER = 20
DOEXP = True
SPEED = 2  # RPM at output
LEFT_HAND = False
DO_SAFE = True
DO_MDP = False

badRobots = [235, 1395, 278]

if LEFT_HAND:
    alphaHome = 360
else:
    alphaHome = 0
betaHome = 180

xCMM = fiducialCoords.xWok.to_numpy()
yCMM = fiducialCoords.yWok.to_numpy()
xyCMM = numpy.array([xCMM, yCMM]).T


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
    polids = numpy.array([0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],dtype=int)
    # polids = numpy.array([0,1,2,3,4,5,6,9,20,27,28,29,30],dtype=int) # desi terms
    def __init__(self, xyCCD, xyWok):
        # first fit a transrotscale model
        self.simTrans = SimilarityTransform()
        # self.simTrans = AffineTransform()
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

# globals
def getGrid(seed):
    rg = RobotGridCalib(angStep, collisionBuffer, epsilon, seed)

    rg.robotDict[235].setAlphaBeta(0.0076,180.0012)
    rg.robotDict[235].setDestinationAlphaBeta(0.0076,180.0012)
    rg.robotDict[235].isOffline = True

    rg.robotDict[1395].setAlphaBeta(0.0208,180.0197)
    rg.robotDict[1395].setDestinationAlphaBeta(0.0208,180.0197)
    rg.robotDict[1395].isOffline = True

    rg.robotDict[278].setAlphaBeta(0.0163, 180.0588)
    rg.robotDict[278].setDestinationAlphaBeta(0.0163, 180.0588)
    rg.robotDict[278].isOffline = True

    if LEFT_HAND:
        for robot in rg.robotDict.values():
            robot.lefthanded = True
    return rg


async def exposeFVC(exptime, stack=1):
    cmdID = 9999
    cmdStr = "%i talk expose %.4f --stack %i\n"%(cmdID, exptime, stack)
    reader, writer = await asyncio.open_connection(
        'localhost', 19996)

    while True:
        data = await reader.readline()
        data = data.decode()
        print(data)
        if "version=" in data:
            print("break!")
            break

    print(f'Send: %s'%cmdStr)
    writer.write(cmdStr.encode())
    await writer.drain()
    while True:
        data = await reader.readline()
        data = data.decode()
        print("read from fvc:", data)
        if "filename=" in data:
            filename = data.split(",")[-1].strip("\n")
        if "%i : "%cmdID in data:
            print("exp command command finished!")
            break

    print("filename: ", filename)

    print('Close the connection')
    writer.close()
    await writer.wait_closed()

    hdus = fits.open(filename)
    print("exptime", exptime, "max counts", numpy.max(hdus[1].data))
    return filename


def dataFrameToFitsRecord(df):
    df = df.copy()
    _dtypes = df.dtypes.to_dict()
    column_dtypes = {}
    for colName, dtype in _dtypes.items():
        dtypeStr = dtype.name
        if "Unnamed" in colName:
            df.drop(colName, axis=1, inplace=True)
            continue
        if colName == "index":
            df.drop(colName, axis=1, inplace=True)
            continue

        if dtypeStr == "object":
            dtypeStr = "a20"
        column_dtypes[colName] = dtypeStr
    rec = df.to_records(index=False, column_dtypes=column_dtypes)
    return rec


# def _getTemps(fps):
#     return {"temp1":1, "temp2":2, "temp3":3}


async def writeProcFITS(filePath, fps, rg, seed, expectedTargCoords, doProcess=True):
    d, oldname = os.path.split(filePath)
    newpath = os.path.join(d, "proc-" + oldname)
    f = fits.open(filePath)

    # invert columns
    f[1].data = f[1].data[:,::-1]

    if doProcess:
        processImage(f[1].data, expectedTargCoords, newpath)

    tables = [
        ("positionerTable", positionerTable),
        ("wokCoords", wokCoords),
        ("fiducialCoords", fiducialCoords)
    ]

    for name, tab in tables:
        rec = dataFrameToFitsRecord(tab)
        binTable = fits.BinTableHDU(rec, name=name)
        f.append(binTable)

    addHeaders = await getIEBData(fps)
    hdr = f[1].header
    for key, val in addHeaders.items():
        hdr[key] = val

    hdr["KAISEED"] = seed

    currPos = await updateCurrentPos(fps, rg, setKaiju=False)
    _cmdAlpha = []
    _cmdBeta = []
    _startAlpha = []
    _startBeta = []

    if len(list(rg.robotDict.values())[0].alphaPath) > 0:
        for posID in currPos.positionerID:
            robot = rg.robotDict[posID]
            _cmdAlpha.append(robot.alphaPath[0][1])
            _cmdBeta.append(robot.betaPath[0][1])
            _startAlpha.append(robot.alphaPath[-1][1])
            _startBeta.append(robot.betaPath[-1][1])

        currPos["cmdAlpha"] = _cmdAlpha
        currPos["cmdBeta"] = _cmdBeta
        currPos["startAlpha"] = _startAlpha
        currPos["startBeta"] = _startBeta

        rec = dataFrameToFitsRecord(currPos)

        binTable = fits.BinTableHDU(rec, name="posAngles")

        f.append(binTable)

    f.writeto(newpath, checksum=True)


async def getIEBData(fps):
    # wok center metal temp
    addHeaders = {}
    try:
        addHeaders["TEMPRTD2"] = (await fps.ieb.read_device("rtd2"))[0]
    except ValueError:
        addHeaders["TEMPRTD2"] = -1
    # outer wok inside air temp

    try:
        addHeaders["TEMPT3"] = (await fps.ieb.read_device("t3"))[0]
    except ValueError:
        addHeaders["TEMPT3"] = -1

    # fps air above wok center
    try:
        addHeaders["TEMPRTD3"] = (await fps.ieb.read_device("rtd3"))[0]
    except ValueError:
        addHeaders["TEMPRTD3"] = -1

    addHeaders["LED1"] = (await fps.ieb.read_device("led1"))[0]
    addHeaders["LED2"] = (await fps.ieb.read_device("led2"))[0]
    addHeaders["LED3"] = (await fps.ieb.read_device("led3"))[0]
    addHeaders["LED4"] = (await fps.ieb.read_device("led4"))[0]

    return addHeaders


async def ledOn(fps, devName, ledpower):
    # global led_state
    # led_state = ledpower

    on_value = 32 * int(1023 * (ledpower) / 100)
    # for dev in ["led1", "led2"]:
    #     print(dev, "on")
    device = fps.ieb.get_device(devName)
    await device.write(on_value)


async def ledOff(fps, devName):
    # global led_state
    # led_state = 0
    # for dev in ["led1", "led2"]:
    #     print(dev, "off")
    device = fps.ieb.get_device(devName)
    await device.write(0)


async def updateCurrentPos(fps, rg, setKaiju=True):
    """Update kaiju's robot grid to reflect the current
    state as reported by the fps
    """
    _posID = []
    _alphaReport = []
    _betaReport = []
    printOne = True
    for r in rg.robotDict.values():
        if r.isOffline:
            continue
        await fps.positioners[r.id].update_position()
        alpha, beta = fps.positioners[r.id].position
        if printOne:
            print("robot %i at %.4f, %.4f"%(r.id, alpha, beta))
            printOne = False
        _posID.append(r.id)
        _alphaReport.append(alpha)
        _betaReport.append(beta)
        if setKaiju:
            r.setAlphaBeta(alpha, beta)
            r.setDestinationAlphaBeta(alphaHome, betaHome)

    if setKaiju:
        for r in rg.robotDict.values():
            if rg.isCollided(r.id):
                print("robot ", r.id, " is collided")

    currPos = pd.DataFrame(
        {
            "positionerID": _posID,
            "alphaReport": _alphaReport,
            "betaReport": _betaReport
        }
    )

    return currPos


async def separate(fps):
    """Move the robots escapeDeg amount to drive them
    apart from eachother
    """
    rg = getGrid(seed=0)
    print("escape")
    currPosDF = await updateCurrentPos(fps, rg)
    rg.pathGenEscape(escapeDeg)
    forwardPath, reversePath = rg.getPathPair(speed=SPEED)
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    # with open("currpos.pkl", "wb") as f:
    #     pickle.dump(currPosDF, f)
    # with open("forwardPath.pkl", "wb") as f:
    #     pickle.dump(forwardPath, f)
    # with open("reversePath.pkl", "wb") as f:
    #     pickle.dump(reversePath, f)
    await fps.send_trajectory(reversePath, use_sync_line=use_sync_line)


async def unwindGrid(fps):
    """Unwind the positioners from any starting point.
    Positioners are queried for position, kaiju builds a reverse path for
    them, the fps commands it.  A runtime error is raised if a path cannot
    be found.

    """
    # overwrite the positions to the positions that the robots
    # are reporting
    rg = getGrid(seed=0)
    print("unwind grid")
    await updateCurrentPos(fps, rg)
    # path generated from current position
    if DO_MDP:
        rg.pathGenMDP(0.9, 0.1)
    else:
        rg.pathGenGreedy()
    forwardPath, reversePath = rg.getPathPair(speed=SPEED)
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    # print(forwardPath)
    # print(reversePath)
    if rg.didFail:
        print("deadlock in unwind")
        if FORCEUNWIND:
            print("doing it anyway")
            await fps.send_trajectory(reversePath, use_sync_line=use_sync_line)
    else:
        await fps.send_trajectory(reversePath, use_sync_line=use_sync_line)

    return rg.didFail


def writePath(pathdict, direction, seed):
    tnow = datetime.datetime.now().isoformat()
    fname = tnow + "_%s_%i.pkl"%(direction, seed)
    with open (fname, "wb") as f:
        pickle.dump(pathdict, f)


def getTargetCoords(rg):
    # return the desired xyWok positions for the metrology
    # fiber for each robot, move this stuff to robot grid...
    robotID = []
    xWokMetExpect = []
    yWokMetExpect = []
    xWokApExpect = []
    yWokApExpect = []
    xWokBossExpect = []
    yWokBossExpect = []


    for r in rg.robotDict.values():
        robotID.append(r.id)
        _xm, _ym, _zm = r.metWokXYZ
        xWokMetExpect.append(_xm)
        yWokMetExpect.append(_ym)

        _xa, _ya, _za = r.apWokXYZ
        xWokApExpect.append(_xa)
        yWokApExpect.append(_ya)

        _xb, _yb, _zb = r.apWokXYZ
        xWokBossExpect.append(_xb)
        yWokBossExpect.append(_yb)

    return pd.DataFrame(
        {
            "robotID": robotID,
            "xWokMetExpect": xWokMetExpect,
            "yWokMetExpect": yWokMetExpect,
            "xWokApExpect": xWokApExpect,
            "yWokApExpect": yWokApExpect,
            "xWokBossExpect": xWokBossExpect,
            "yWokBossExpect": yWokBossExpect,
        }
    )

def setRandomTargets(rg, alphaHome, betaHome, betaLim=None):
    for robot in rg.robotDict.values():
        robot.setDestinationAlphaBeta(alphaHome, betaHome)
        if robot.isOffline:
            continue
        if betaLim is not None:
            alpha = numpy.random.uniform(0, 359.99)
            beta = numpy.random.uniform(betaLim[0], betaLim[1])
            robot.setAlphaBeta(alpha, beta)
        else:
            robot.setXYUniform()

    if betaLim is not None and rg.getNCollisions() > 0:
        raise RuntimeError("betaLim specified, but collisions present")
    else:
        rg.decollideGrid()

    return getTargetCoords(rg)


def extract(imgData):
    # run source extractor,
    # do some filtering
    # return the extracted centroids
    imgData = numpy.array(imgData, dtype="float")
    bkg = sep.Background(imgData)
    bkg_image = bkg.back()
    data_sub = imgData - bkg_image
    objects = sep.extract(data_sub, 3.5, err=bkg.globalrms)
    objects = pd.DataFrame(objects)

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


def processImage(imgData, expectedTargCoords, newpath):
    centroids = extract(imgData)
    xyCCD = centroids[["x", "y"]].to_numpy()

    # just to get close enough to associate the
    # correct centroid with the correct fiducial...
    xWokExpect = numpy.concatenate([xCMM, expectedTargCoords.xWokMetExpect.to_numpy()])
    yWokExpect = numpy.concatenate([yCMM, expectedTargCoords.yWokMetExpect.to_numpy()])
    xyWokExpect = numpy.array([xWokExpect, yWokExpect]).T

    rt = RoughTransform(xyCCD, xyWokExpect)
    xyWokRough = rt.apply(xyCCD)

    # first associate fiducials and build
    # first round just use outer fiducials
    rCMM = numpy.sqrt(xyCMM[:,0]**2+xyCMM[:,1]**2)
    keep = rCMM > 310
    xyCMMouter = xyCMM[keep, :]
    # argFound, fidRoughDist = argNearestNeighbor(xyCMM, xyWokRough)
    argFound, fidRoughDist = argNearestNeighbor(xyCMMouter, xyWokRough)
    print("max fiducial rough distance", numpy.max(fidRoughDist))
    xyFiducialCCD = xyCCD[argFound]
    xyFiducialWokRough = xyWokRough[argFound]


    plt.figure(figsize=(8,8))
    plt.title("rough fiducial association")
    plt.plot(xyWokRough[:,0], xyWokRough[:,1], 'o', ms=4, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1, label="centroid")
    plt.plot(expectedTargCoords.xWokMetExpect.to_numpy(), expectedTargCoords.yWokMetExpect.to_numpy(), 'xk', ms=3, label="expected met")
    # overplot fiducials
    plt.plot(xCMM, yCMM, "D", ms=6, markerfacecolor="None", markeredgecolor="cornflowerblue", markeredgewidth=1, label="expected fid")
    for cmm, rough in zip(xyCMMouter, xyFiducialWokRough):
        plt.plot([cmm[0], rough[0]], [cmm[1], rough[1]], "-k")
    plt.axis("equal")
    plt.legend()
    plt.savefig(newpath+"roughassoc.png", dpi=350)
    plt.close()

    ft = FullTransform(xyFiducialCCD, xyCMMouter)
    print("full trans 1 bias, unbias", ft.rms*1000, ft.unbiasedRMS*1000)
    xyWokMeas = ft.apply(xyCCD, zb=False)

    plt.figure(figsize=(8,8))
    plt.title("full transform 1")
    plt.plot(xyWokMeas[:,0], xyWokMeas[:,1], 'o', ms=4, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1, label="centroid")
    plt.plot(expectedTargCoords.xWokMetExpect.to_numpy(), expectedTargCoords.yWokMetExpect.to_numpy(), 'xk', ms=3, label="expected met")
    # overplot fiducials
    plt.plot(xCMM, yCMM, "D", ms=6, markerfacecolor="None", markeredgecolor="cornflowerblue", markeredgewidth=1, label="expected fid")
    plt.axis("equal")
    plt.legend()
    plt.xlim([-350, 350])
    plt.ylim([-350,350])
    plt.savefig(newpath+"full1.png", dpi=350)
    plt.close()

    # re-associate fiducials, some could have been wrongly associated in
    # first fit but second fit should be better?
    argFound, fidRoughDist = argNearestNeighbor(xyCMM, xyWokMeas)
    print("max fiducial fit 2 distance", numpy.max(fidRoughDist))
    xyFiducialCCD = xyCCD[argFound]  # over writing
    xyFiducialWokRefine = xyWokMeas[argFound]


    plt.figure(figsize=(8,8))
    plt.title("refined fiducial association")
    plt.plot(xyWokMeas[:,0], xyWokMeas[:,1], 'o', ms=4, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1, label="centroid")
    plt.plot(expectedTargCoords.xWokMetExpect.to_numpy(), expectedTargCoords.yWokMetExpect.to_numpy(), 'xk', ms=3, label="expected met")
    # overplot fiducials
    plt.plot(xCMM, yCMM, "D", ms=6, markerfacecolor="None", markeredgecolor="cornflowerblue", markeredgewidth=1, label="expected fid")
    for cmm, rough in zip(xyCMM, xyFiducialWokRefine):
        plt.plot([cmm[0], rough[0]], [cmm[1], rough[1]], "-k")
    plt.axis("equal")
    plt.legend()
    plt.savefig(newpath+"refineassoc.png", dpi=350)
    plt.close()

    # try a new transform
    ft = FullTransform(xyFiducialCCD, xyCMM)
    print("full trans 2 bias, unbias", ft.rms*1000, ft.unbiasedRMS*1000)
    xyWokMeas = ft.apply(xyCCD) # overwrite


    xyWokMeas = ft.apply(xyCCD)
    plt.figure(figsize=(8,8))
    plt.title("full transform 2")
    plt.plot(xyWokMeas[:,0], xyWokMeas[:,1], 'o', ms=4, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1, label="centroid")
    plt.plot(expectedTargCoords.xWokMetExpect.to_numpy(), expectedTargCoords.yWokMetExpect.to_numpy(), 'xk', ms=3, label="expected met")
    # overplot fiducials
    plt.plot(xCMM, yCMM, "D", ms=6, markerfacecolor="None", markeredgecolor="cornflowerblue", markeredgewidth=1, label="expected fid")
    plt.axis("equal")
    plt.legend()
    plt.xlim([-350, 350])
    plt.ylim([-350,350])
    plt.savefig(newpath+"full2.png", dpi=350)
    plt.close()



    xyFiducialMeas = ft.apply(xyFiducialCCD)

    # transform all CCD detections to wok space

    xyExpectPos = expectedTargCoords[["xWokMetExpect", "yWokMetExpect"]].to_numpy()

    argFound, metDist = argNearestNeighbor(xyExpectPos, xyWokMeas)
    print("max metrology distance", numpy.max(metDist))
    xyWokRobotMeas = xyWokMeas[argFound]

    expectedTargCoords["xWokMetMeas"] = xyWokRobotMeas[:, 0]
    expectedTargCoords["yWokMetMeas"] = xyWokRobotMeas[:, 1]

    dx = expectedTargCoords.xWokMetExpect - expectedTargCoords.xWokMetMeas
    dy = expectedTargCoords.yWokMetExpect - expectedTargCoords.yWokMetMeas

    rms = numpy.sqrt(numpy.mean(dx**2+dy**2))
    print("rms full fit um", rms*1000)

    return expectedTargCoords


def getUnsafePath(seed):
    rg = getGrid(seed)
    setRandomTargets(rg, alphaHome, betaHome)
    # tc = getTargetCoords(rg)
    attempt = 0
    replacedRobotList = []
    for jj in range(5):
        print("attempt %i"%attempt)
        tstart = time.time()
        expectedTargCoords = getTargetCoords(rg)
        if DO_MDP:
            rg.pathGenMDP(0.9, 0.1)
        else:
            rg.pathGenGreedy()
        print("path gen took %.2f secs"%(time.time()-tstart))
        print("%s deadlocked robots"%len(rg.deadlockedRobots()))
        # plotOne(0, rg, "beg_apo%i_%i.png"%(seed, attempt), isSequence=False)
        # plotOne(rg.nSteps, rg, "end_apo%i_%i.png"%(seed, attempt), isSequence=False)
        if not rg.didFail:
            break
        if len(rg.deadlockedRobots()) > 6:
            print("too many deadlocks to resolve")
            break
        while True:
            nextReplacement = numpy.random.choice(rg.deadlockedRobots())
            if nextReplacement not in badRobots:
                break
        replacedRobotList.append(nextReplacement)
        rg = getGrid(seed)
        setRandomTargets(rg, alphaHome, betaHome)
        for robotID in replacedRobotList:
            robot = rg.robotDict[robotID]
            robot.setXYUniform()
        rg.decollideGrid()

        attempt += 1

    if rg.didFail:
        print("failed")
    else:
        print("solved!")
        # plotPaths(rg, downsample=1000, filename="apo%i.mp4"%seed)
    return rg, expectedTargCoords

def getSafePath(seed):
    rg = getGrid(seed)
    betaLim = [165, 195]
    numpy.random.seed(seed)

    expectedTargCoords = setRandomTargets(rg, alphaHome, betaHome, betaLim)
    rg.pathGenGreedy()
    return rg, expectedTargCoords



async def outAndBack(fps, seed, safe=True):
    """Move robots out and back on non-colliding trajectories
    """
    # rg = getGrid(seed)

    print("out and back safe=%s seed=%i"%(str(safe), seed))
    if safe:
        rg, expectedTargCoords = getSafePath(seed)
    else:
        rg, expectedTargCoords = getUnsafePath(seed)


    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)

    await ledOff(fps, "led1")
    await ledOff(fps, "led2")
    if not rg.didFail and rg.smoothCollisions == 0:
        forwardPath, reversePath = rg.getPathPair(speed=SPEED)
        print("sending forward path")
        try:
            await fps.send_trajectory(forwardPath, use_sync_line=use_sync_line)
        except TrajectoryError as e:
            print("trajectory failed!!!!")

            t = e.trajectory.failed_positioners

            print("%s failed on forward"%str(t))
            return

        print("forward path done")
        if DOEXP:
            await ledOn(fps, "led1", LED12_VALUE)
            await ledOn(fps, "led2", LED12_VALUE)
            await asyncio.sleep(1)
            print("exposing img 1")
            filename = await exposeFVC(exptime)
            await writeProcFITS(filename, fps, rg, seed, expectedTargCoords)

            await ledOff(fps, "led1")
            await ledOff(fps, "led2")
            await ledOn(fps, "led3", LED3_VALUE)
            await asyncio.sleep(1)
            print("exposing img 2")
            filename = await exposeFVC(exptime)
            await writeProcFITS(filename, fps, rg, seed, expectedTargCoords)

            await ledOff(fps, "led3")
            await ledOn(fps, "led4", LED4_VALUE)
            await asyncio.sleep(1)
            print("exposing img 3")
            filename = await exposeFVC(exptime)
            await writeProcFITS(filename, fps, rg, seed, expectedTargCoords)

            await ledOff(fps, "led4")
            await ledOn(fps, "led1", LED12_VALUE)
            await ledOn(fps, "led2", LED12_VALUE)
            await asyncio.sleep(1)
            print("exposing img 4")
            filename = await exposeFVC(exptime)
            await writeProcFITS(filename, fps, rg, seed, expectedTargCoords)

        else:
            print("sleeping for 60 seconds")
            await asyncio.sleep(2)

        await ledOff(fps, "led1")
        await ledOff(fps, "led2")

        print("sending reverse path")
        try:
            await fps.send_trajectory(reversePath, use_sync_line=use_sync_line)
        except TrajectoryError as e:
            print("trajectory failed!!!!")
            t = e.trajectory.failed_positioners
            print("%s failied on reverse path"%(str(t)))
            return

    else:
        print("not sending path")



async def main():
    global SEED
    if SEED is None:
        SEED = numpy.random.randint(0, 30000)

    print("seed", SEED)
    fps = FPS()
    await fps.initialise()
    # await getTemps(fps)
    # await openCamera()
    # for exptime in [0.5, 0.6, 0.7]:#, 1, 1.1, 1.2]:
    #     print("\n\n")

    # filename = await exposeFVC(exptime)
    # await asyncio.sleep(2)
    # await writeProcFITS(filename, fps)
    if EXPLODEFIRST:
        print("exploding grid")
        await separate(fps)
        print("done exploding")

    print("unwind")
    unwindFail = await unwindGrid(fps)
    print("unwound")

    if unwindFail:
        print("not all positioners are unwound, exiting")
        await fps.shutdown()
        return

    if UNWINDONLY:
        await fps.shutdown()
        return
    # print("unwound")



    # for ii in range(100):
    ii = 0
    while ii < NITER:
        ii += 1
        SEED += 1
        print("\n\niter %i\n\n"%ii)
        await outAndBack(fps, SEED, safe=DO_SAFE)


    await fps.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
