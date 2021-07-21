import asyncio

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

rg = RobotGridCalib(angStep, collisionBuffer, epsilon)
fps = FPS()


async def unwindGrid(fps):
    """Unwind the positioners from any starting point.
    Positioners are queried for position, kaiju builds a reverse path for
    them, the fps commands it.  A runtime error is raised if a path cannot
    be found.

    """
    # overwrite the positions to the positions that the robots
    # are reporting
    for r in rg.robotDict.values():
        await fps.positioners[r.id].update_position()
        alpha, beta = fps.positioners[r.id].position
        r.setAlphaBeta(alpha, beta)

    for r in rg.robotDict.values():
        if rg.isCollided(r.id):
            print("robot ", r.id, " is collided")
    forwardPath, reversePath = rg.getPathPair()
    print("didFail", rg.didFail)
    print("smooth collisions", rg.smoothCollisions)
    print(forwardPath)
    print(reversePath)

    # await fps.send_trajectory(reversePath)

async def main():
    await fps.initialise()
    await unwindGrid(fps)


if __name__ == "__main__":
    asyncio.run(main())