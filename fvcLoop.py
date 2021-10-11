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
escapeDeg = 20 # 20 degrees of motion to escape

# globals
def getGrid(seed):
    rg = RobotGridCalib(angStep, collisionBuffer, epsilon, seed)
    return rg

    # rg.robotDict[1255].setAlphaBeta(305, 238.29)
    # rg.robotDict[1255].setDestinationAlphaBeta(305, 238.29)
    # rg.robotDict[1255].isOffline = True
    # rg.robotDict[717].setAlphaBeta(0,180)
    # rg.robotDict[717].setDestinationAlphaBeta(0,180)
    # rg.robotDict[717].isOffline = True
    # rg.robotDict[1367].setAlphaBeta(0, 164.88)
    # rg.robotDict[1367].setDestinationAlphaBeta(0, 164.88)
    # rg.robotDict[1367].isOffline = True
    # rg.robotDict[398].setAlphaBeta(0, 152.45)
    # rg.robotDict[398].setDestinationAlphaBeta(0, 152.45)
    # rg.robotDict[398].isOffline = True
    # rg.robotDict[775].setAlphaBeta(0, 180)
    # rg.robotDict[775].setDestinationAlphaBeta(0, 180)
    # rg.robotDict[775].isOffline = True
    # rg.robotDict[738].setAlphaBeta(0, 179.8121)
    # rg.robotDict[738].setDestinationAlphaBeta(0, 179.8121)
    # rg.robotDict[738].isOffline = True
    # rg.robotDict[1003].setAlphaBeta(0, 180.0502)
    # rg.robotDict[1003].setDestinationAlphaBeta(0, 180.0502)
    # rg.robotDict[1003].isOffline = True
    # rg.robotDict[981].setAlphaBeta(0, 179.7738)
    # rg.robotDict[981].setDestinationAlphaBeta(0, 179.7738)
    # rg.robotDict[981].isOffline = True
    # rg.robotDict[545].setAlphaBeta(5.5527, 180.2876)
    # rg.robotDict[545].setDestinationAlphaBeta(5.5527, 180.2876)
    # rg.robotDict[545].isOffline = True
    # rg.robotDict[688].setAlphaBeta(0.0129, 180.8187)
    # rg.robotDict[688].setDestinationAlphaBeta(0.0129, 180.8187)
    # rg.robotDict[688].isOffline = True
    # rg.robotDict[474].setAlphaBeta(0, 180)
    # rg.robotDict[474].setDestinationAlphaBeta(0, 180)
    # rg.robotDict[474].isOffline = True
    # rg.robotDict[769].setAlphaBeta(45, 180)
    # rg.robotDict[769].setDestinationAlphaBeta(45, 180)
    # rg.robotDict[769].isOffline = True
    # rg.robotDict[652].setAlphaBeta(0, 180)
    # rg.robotDict[652].setDestinationAlphaBeta(0, 180)
    # rg.robotDict[652].isOffline = True
    # rg.robotDict[703].setAlphaBeta(0, 180)
    # rg.robotDict[703].setDestinationAlphaBeta(0, 180)
    # rg.robotDict[703].isOffline = True
    # # rg.robotDict[769].setAlphaBeta(0, 180)
    # # rg.robotDict[769].setDestinationAlphaBeta(0, 180)
    # # rg.robotDict[769].isOffline = True
    # return rg

# cam = None
# led_state = None


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
        data = await reader.read(100)
        data = data.decode()
        # print(data)
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


async def appendDataToFits(filePath, fps, rg, seed):
    d, oldname = os.path.split(filePath)
    newpath = os.path.join(d, "proc-" + oldname)
    f = fits.open(filePath)
    # invert columns

    f[1].data = f[1].data[:,::-1]

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


# async def openCamera():
#     global cam
#     camID = 0
#     bcs = BaslerCameraSystem(BaslerCamera, camera_config=config)
#     sids = bcs.list_available_cameras()
#     cam = await bcs.add_camera(uid=sids[camID], autoconnect=True)


# async def expose():
#     global cam
#     tnow = datetime.datetime.now().isoformat()
#     expname = tnow + ".fits"
#     exp = await cam.expose(exptime * 1e-6, stack=nAvgImg, filename=expname)
#     await exp.write()
#     print("wrote %s"%expname, "max counts", numpy.max(exp.data))
#     return expname


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
    await fps.send_trajectory(reversePath, use_sync_line=True)

def writePath(pathdict, direction, seed):
    tnow = datetime.datetime.now().isoformat()
    fname = tnow + "_%s_%i.pkl"%(direction, seed)
    with open (fname, "wb") as f:
        pickle.dump(pathdict, f)


async def outAndBack(fps, seed, safe=True):
    """Move robots out and back on non-colliding trajectories
    """
    rg = getGrid(seed)
    print("out and back safe=%s seed=%i"%(str(safe), seed))
    if safe:
        betaLim = [165, 195]
    else:
        betaLim = None
    forwardPath, reversePath = rg.getRandomPathPair(
        alphaHome=alphaHome, betaHome=betaHome, betaLim=betaLim
    )
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    #for r in rg.robotDict.values():
        #print("beta", r.betaPath[0])
    await ledOff(fps, "led1")
    await ledOff(fps, "led2")
    if not rg.didFail and rg.smoothCollisions == 0:
        print("sending forward path")
        # writePath(forwardPath, "forward", seed)
        # await fps.send_trajectory(forwardPath, use_sync_line=True)
        try:
            await fps.send_trajectory(forwardPath, use_sync_line=True)
            # await send_trajectory(fps, forwardPath, use_sync_line=True)
        except TrajectoryError as e:
            print("trajectory error on forward.  trying to resend")
            try :
                await.fps.send_trajectory(forwardPath, use_sync_line=True)
            except TrajectoryError as e:
                print("trajectory failed twice!!!!")
                writePath(forwardPath, "forward", seed)
                # note offending robots
                t = e.trajectory.failed_positioners
                with open("failed_positioners_forward_%i.txt"%seed, "w") as f:
                    f.writeline(str(t))
                print("failed on forward, continue")
                return # dont send forward path
                print("unwinding grid")
                return
            print("second try to send trajectory worked?!??!")

        print("forward path done")
        await ledOn(fps, "led1")
        await asyncio.sleep(1)
        print("exposing img 1")
        filename = await exposeFVC(exptime)
        await appendDataToFits(filename, fps, rg, seed)
        await ledOn(fps, "led2")
        await asyncio.sleep(1)
        print("exposing img2")
        filename = await exposeFVC(exptime)
        await appendDataToFits(filename, fps, rg, seed)
        await ledOff(fps, "led1")
        await ledOff(fps, "led2")

        print("sending reverse path")
        writePath(reversePath, "reverse", seed)
        try:
            await fps.send_trajectory(reversePath, use_sync_line=True)
            # await send_trajectory(fps, reversePath, use_sync_line=True)
        except TrajectoryError as e:
            print("trajectory error on reverse.  trying to resend")
            try :
                await.fps.send_trajectory(reversePath, use_sync_line=True)
            except TrajectoryError as e:
                print("trajectory failed twice!!!!")
                writePath(reversePath, "reverse", seed)
                # note offending robots
                t = e.trajectory.failed_positioners
                with open("failed_positioners_reverse_%i.txt"%seed, "w") as f:
                    f.writeline(str(t))
                print("unwinding grid")
                await unwindGrid(fps)
                return
            print("second try to send trajectory worked?!??!")

        # await fps.send_trajectory(reversePath, use_sync_line=True)
    else:
        print("not sending path")


# async def outAndBackUnsafe(fps):
#     """Move robots out and back on potentially, but hopefully not
#     colliding trajectories
#     """
#     print("out and back UNsafe")
#     forwardPath, reversePath = rg.getRandomPathPair(
#         alphaHome=alphaHome, betaHome=betaHome, betaSafe=False
#     )

#     print("didFail", rg.didFail)
#     print("smooth collisions", rg.smoothCollisions)
#     if not rg.didFail and rg.smoothCollisions == 0:
#         await fps.send_trajectory(forwardPath, use_sync_line=False)
#         await asyncio.sleep(25)
#         await fps.send_trajectory(reversePath, use_sync_line=False)
#     else:
#         print("not sending path")

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
    # await appendDataToFits(filename, fps)
    print("unwind")
    await unwindGrid(fps)

    if UNWINDONLY:
        await fps.shutdown()
        return
    print("unwound")



    # for ii in range(100):
    ii = 0
    while ii <500:
        ii += 1
        seed += 1
        print("\n\niter %i\n\n"%ii)
        await outAndBack(fps, seed, safe=False)


    await fps.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
