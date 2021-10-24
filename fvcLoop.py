import asyncio
import os
import numpy
import datetime
from astropy.io import fits
import pandas as pd
import pickle
import datetime

from jaeger import FPS, log
from jaeger.commands.trajectory import send_trajectory
from jaeger.exceptions import FPSLockedError, TrajectoryError
# log.sh.setLevel(5)
from kaiju.robotGrid import RobotGridCalib
from coordio.defaults import positionerTableCalib, wokCoordsCalib, fiducialCoordsCalib
# from baslerCam import BaslerCamera, BaslerCameraSystem, config
import sep
from skimage.transform import SimilarityTransform
from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY

# from processImgs import SimTrans, argNearestNeighbor, FullTransform

# Speed = 3               # RPM at output, breakout as kaiju param?
# smoothPts = 5           # width of velocity smoothing window, breakout as kaiju param?
# collisionShrink = 0.05  # amount to decrease collisionBuffer by when checking smoothed and simplified paths

angStep = 0.1          # degrees per step in kaiju's rough path
epsilon = angStep * 2   # max error (deg) allowed in kaiju's path simplification
collisionBuffer = 2.4    # effective *radius* of beta arm in mm effective beta arm width is 2*collisionBuffer
exptime = 1.2
# CONTINUOUS = False
UNWINDONLY = False
TAKE_IMGS = False
LED_VALUE = 1
alphaHome = 0
betaHome = 180
seed = None
escapeDeg = 20  # 20 degrees of motion to escape
use_sync_line = False

xCMM = fiducialCoordsCalib.xWok.to_numpy()
yCMM = fiducialCoordsCalib.yWok.to_numpy()
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
    def __init__(self, centroids, expectedTargCoords):
        # scale pixels to mm roughly
        xCCD = centroids.x.to_numpy()
        yCCD = centroids.y.to_numpy()

        xWok = numpy.concatenate(
            [expectedTargCoords.xWokMetExpect.to_numpy(), xCMM]
        )
        yWok = numpy.concatenate(
            [expectedTargCoords.yWokMetExpect.to_numpy(), yCMM]
        )
        

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


class FullTransfrom(object):
    # use fiducials to fit this
    polids = numpy.array([0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],dtype=int)

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

# globals
def getGrid(seed):
    rg = RobotGridCalib(angStep, collisionBuffer, epsilon, seed)

    rg.robotDict[1097].setAlphaBeta(0, 180.0001)
    rg.robotDict[1097].setDestinationAlphaBeta(0, 180.0001)
    rg.robotDict[1097].isOffline = True
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


async def writeProcFITS(filePath, fps, rg, seed, expectedTargCoords):
    d, oldname = os.path.split(filePath)
    newpath = os.path.join(d, "proc-" + oldname)
    f = fits.open(filePath)

    # invert columns
    f[1].data = f[1].data[:,::-1]

    processImage(f[1].data, expectedTargCoords)

    tables = [
        ("positionerTable", positionerTableCalib),
        ("wokCoords", wokCoordsCalib),
        ("fiducialCoords", fiducialCoordsCalib)
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

    return addHeaders


async def ledOn(fps, devName, ledpower=LED_VALUE):
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


# async def separate(fps):
#     """Move the robots escapeDeg amount to drive them
#     apart from eachother
#     """
#     rg = getGrid(seed=0)
#     print("escape")
#     await updateCurrentPos(fps)
#     rg.pathGenEscape(escapeDeg)
#     forwardPath, reversePath = rg.getPathPair()
#     print("didFail", rg.didFail)
#     print("smooth collisions", rg.smoothCollisions)
#     await fps.send_trajectory(forwardPath, use_sync_line=False)


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
    forwardPath, reversePath = rg.getPathPair()
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    # print(forwardPath)
    # print(reversePath)
    await fps.send_trajectory(reversePath, use_sync_line=use_sync_line)


def writePath(pathdict, direction, seed):
    tnow = datetime.datetime.now().isoformat()
    fname = tnow + "_%s_%i.pkl"%(direction, seed)
    with open (fname, "wb") as f:
        pickle.dump(pathdict, f)


def setRandomTargets(rg, alphaHome=0, betaHome=180, betaLim=None):
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
    print("got", len(objects), "centroids")

    # ecentricity
    objects["ecentricity"] = 1 - objects["b"] / objects["a"]

    # slope of ellipse (optical distortion direction)
    objects["slope"] = numpy.tan(objects["theta"] + numpy.pi/2) # rotate by 90
    # intercept of optical distortion direction
    objects["intercept"] = objects["y"] - objects["slope"] * objects["x"]

    # ignore everything less than 100 pixels
    # something changed when rick bumped things?
    #objects = objects[objects["npix"] > 100]

    # filter on most eliptic, this is an assumption!!!!
    # objects["outerFIF"] = objects.ecentricity > 0.15

    return objects


def processImage(imgData, expectedTargCoords):
    centroids = extract(imgData)
    xyCCD = centroids[["x", "y"]].to_numpy()

    # just to get close enough to associate the
    # correct centroid with the correct fiducial...
    rt = RoughTransform(centroids, expectedTargCoords)
    xyWokRough = numpy.array([rt.roughWokX, rt.roughWokY]).T


    # first associate fiducials and build
    argFound, fidRoughDist = argNearestNeighbor(xyCMM, xyWokRough)
    print("max fiducial rough distance", numpy.max(fidRoughDist))
    xyFiducialCCD = xyCCD[argFound]

    ft = FullTransfrom(xyFiducialCCD, xyCMM)
    xyFiducialMeas = ft.apply(xyFiducialCCD)

    xyCCD = centroids[["x", "y"]].to_numpy()
    # transform all CCD detections to wok space
    xyWokMeas = ft.apply(xyCCD)

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


async def outAndBack(fps, seed, safe=True):
    """Move robots out and back on non-colliding trajectories
    """
    rg = getGrid(seed)
    print("out and back safe=%s seed=%i"%(str(safe), seed))
    if safe:
        betaLim = [165, 195]
    else:
        betaLim = None

    expectedTargCoords = setRandomTargets(rg, alphaHome, betaHome, betaLim)
    forwardPath, reversePath = rg.getPathPair()

    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)

    await ledOff(fps, "led1")
    await ledOff(fps, "led2")
    if not rg.didFail and rg.smoothCollisions == 0:
        print("sending forward path")
        # writePath(forwardPath, "forward", seed)
        # await fps.send_trajectory(forwardPath, use_sync_line=True)
        try:
            await fps.send_trajectory(forwardPath, use_sync_line=use_sync_line)
            # await send_trajectory(fps, forwardPath, use_sync_line=True)
        except TrajectoryError as e:
            print("trajectory failed!!!!")
            writePath(forwardPath, "forward", seed)
            # note offending robots
            t = e.trajectory.failed_positioners
            with open("failed_positioners_forward_%i.txt"%seed, "w") as f:
                f.write(str(t))
            print("failed on forward")
            print("unwinding grid")
            await unwindGrid(fps)
            return

        print("forward path done")
        await ledOn(fps, "led1")
        await asyncio.sleep(1)
        print("exposing img 1")
        filename = await exposeFVC(exptime)
        await writeProcFITS(filename, fps, rg, seed, expectedTargCoords)
        await ledOn(fps, "led2")
        await asyncio.sleep(1)
        print("exposing img2")
        filename = await exposeFVC(exptime)
        await writeProcFITS(filename, fps, rg, seed, expectedTargCoords)
        await ledOff(fps, "led1")
        await ledOff(fps, "led2")

        print("sending reverse path")
        #writePath(reversePath, "reverse", seed)
        try:
            await fps.send_trajectory(reversePath, use_sync_line=use_sync_line)
            # await send_trajectory(fps, reversePath, use_sync_line=use_sync_line)
        except TrajectoryError as e:
            print("trajectory failed!!!!")
            writePath(reversePath, "reverse", seed)
            # note offending robots
            t = e.trajectory.failed_positioners
            with open("failed_positioners_reverse_%i.txt"%seed, "w") as f:
                f.write(str(t))
            print("unwinding grid")
            await unwindGrid(fps)
            return

        # await fps.send_trajectory(reversePath, use_sync_line=True)
    else:
        print("not sending path")


async def main():
    seed = 9
    if seed is None:
        seed = numpy.random.randint(0, 30000)

    print("seed", seed)
    fps = FPS()
    await fps.initialise()
    # await getTemps(fps)
    # await openCamera()
    # for exptime in [0.5, 0.6, 0.7]:#, 1, 1.1, 1.2]:
    #     print("\n\n")

    # filename = await exposeFVC(exptime)
    # await asyncio.sleep(2)
    # await writeProcFITS(filename, fps)
    print("unwind")
    await unwindGrid(fps)

    if UNWINDONLY:
        await fps.shutdown()
        return
    print("unwound")



    # for ii in range(100):
    ii = 0
    while ii <3:
        ii += 1
        seed += 1
        print("\n\niter %i\n\n"%ii)
        await outAndBack(fps, seed, safe=False)


    await fps.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
