import asyncio
import os
import numpy
import datetime
from astropy.io import fits
import pandas as pd
import pickle
import datetime

from jaeger import FPS, log
# log.sh.setLevel(5)
from kaiju.robotGrid import RobotGridCalib
from coordio.defaults import positionerTableCalib, wokCoordsCalib, fiducialCoordsCalib
# from baslerCam import BaslerCamera, BaslerCameraSystem, config

# Speed = 3               # RPM at output, breakout as kaiju param?
# smoothPts = 5           # width of velocity smoothing window, breakout as kaiju param?
# collisionShrink = 0.05  # amount to decrease collisionBuffer by when checking smoothed and simplified paths

angStep = 0.1          # degrees per step in kaiju's rough path
epsilon = angStep * 2   # max error (deg) allowed in kaiju's path simplification
collisionBuffer = 2.6    # effective *radius* of beta arm in mm effective beta arm width is 2*collisionBuffer
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


async def updateCurrentPos(fps, rg, setKaiju=True):
    """Update kaiju's robot grid to reflect the current
    state as reported by the fps
    """
    _posID = []
    _alphaReport = []
    _betaReport = []
    for r in rg.robotDict.values():
        if r.isOffline:
            continue
        await fps.positioners[r.id].update_position()
        alpha, beta = fps.positioners[r.id].position
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
    await fps.send_trajectory(reversePath, use_sync_line=False)

# def writePath(pathdict, direction, seed):
#     tnow = datetime.datetime.now().isoformat()
#     fname = tnow + "_%s_%i.pkl"%(direction, seed)
#     with open (fname, "wb") as f:
#         pickle.dump(pathdict, f)


async def outAndBackSafe(fps, seed):
    """Move robots out and back on non-colliding trajectories
    """
    rg = getGrid(seed)
    print("out and back safe")
    forwardPath, reversePath = rg.getRandomPathPair(
        alphaHome=alphaHome, betaHome=betaHome, betaLim=[165, 195]
    )
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    #for r in rg.robotDict.values():
        #print("beta", r.betaPath[0])

    if not rg.didFail and rg.smoothCollisions == 0:
        await asyncio.sleep(1)
        print("sending forward path")
        # writePath(forwardPath, "forward", seed)
        await fps.send_trajectory(forwardPath, use_sync_line=False)

        await asyncio.sleep(1)


        print("sending reverse path")
        await fps.send_trajectory(reversePath, use_sync_line=False)
    else:
        print("not sending path")


async def outAndBackUnsafe(fps, seed):
    """Move robots out and back on potentially, but hopefully not
    colliding trajectories
    """
    print("out and back UNsafe")
    rg = getGrid(seed)
    forwardPath, reversePath = rg.getRandomPathPair(
        alphaHome=alphaHome, betaHome=betaHome, betaLim=None
    )

    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    if not rg.didFail and rg.smoothCollisions == 0:
        await asyncio.sleep(1)
        print("sending forward path")
        await fps.send_trajectory(forwardPath, use_sync_line=False)
        await asyncio.sleep(1)
        print("sending reverse path")
        await fps.send_trajectory(reversePath, use_sync_line=False)
    else:
        print("not sending path")

async def main():
    seed = None
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

    ii = 0
    while ii < 50:
        ii += 1
        seed += 1
        print("\n\niter %i\n\n"%ii)
        await outAndBackUnsafe(fps, seed)


    await fps.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
