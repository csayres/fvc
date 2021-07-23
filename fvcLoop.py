import asyncio
import numpy

from jaeger import FPS, log
from kaiju.robotGrid import RobotGridCalib


# Speed = 3               # RPM at output
angStep = 0.05          # degrees per step in kaiju's rough path
smoothPts = 5           # width of velocity smoothing window
epsilon = angStep * 2   # max error (deg) allowed in kaiju's path simplification
collisionBuffer = 2.4    # effective *radius* of beta arm in mm effective beta arm width is 2*collisionBuffer
# collisionShrink = 0.05  # amount to decrease collisionBuffer by when checking smoothed and simplified paths
exptime = 10000 # micro seconds
nAvgImg = 20 # number of stack
CONTINUOUS = True
UNWINDONLY = False
TAKE_IMGS = False
LED_VALUE = 52
alphaHome = 0
betaHome = 180
seed = None
escapeDeg = 20 # 20 degrees of motion to escape

if seed is None:
    seed = numpy.random.randint(0, 30000)

print("seed", seed)

rg = RobotGridCalib(angStep, collisionBuffer, epsilon, seed)


async def updateCurrentPos(fps):
    for r in rg.robotDict.values():
        await fps.positioners[r.id].update_position()
        alpha, beta = fps.positioners[r.id].position
        r.setAlphaBeta(alpha, beta)
        r.setDestinationAlphaBeta(alphaHome, betaHome)

    for r in rg.robotDict.values():
        if rg.isCollided(r.id):
            print("robot ", r.id, " is collided")


async def escape(fps):
    print("escape")
    await updateCurrentPos(fps)
    rg.pathGenEscape(escapeDeg)
    forwardPath, reversePath = rg.getPathPair()
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    await fps.send_trajectory(forwardPath)


async def unwindGrid(fps):
    """Unwind the positioners from any starting point.
    Positioners are queried for position, kaiju builds a reverse path for
    them, the fps commands it.  A runtime error is raised if a path cannot
    be found.

    """
    # overwrite the positions to the positions that the robots
    # are reporting
    print("unwind grid")
    await updateCurrentPos(fps)
    forwardPath, reversePath = rg.getPathPair()
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    # print(forwardPath)
    # print(reversePath)

    await fps.send_trajectory(reversePath)


async def outAndBackSafe(fps):
    print("out and back safe")
    forwardPath, reversePath = rg.getRandomPathPair(
        alphaHome=alphaHome, betaHome=betaHome, betaSafe=True
    )
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)

    if not rg.didFail and rg.smoothCollisions == 0:
        await fps.send_trajectory(forwardPath)
        await fps.send_trajectory(reversePath)
    else:
        print("not sending path")


async def outAndBackUnsafe(fps):
    print("out and back UNsafe")
    forwardPath, reversePath = rg.getRandomPathPair(
        alphaHome=alphaHome, betaHome=betaHome, betaSafe=False
    )

    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    if not rg.didFail and rg.smoothCollisions == 0:
        await fps.send_trajectory(forwardPath)
        await fps.send_trajectory(reversePath)
    else:
        print("not sending path")

async def main():
    fps = FPS()
    await fps.initialise()
    await unwindGrid(fps)
    await outAndBackSafe(fps)
    await outAndBackUnsafe(fps)
    # for i in range(5):
    #     await escape(fps)

    await fps.shutdown()


if __name__ == "__main__":
    asyncio.run(main())