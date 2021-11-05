import asyncio
from jaeger import FPS
from fvcLoop import updateCurrentPos, getGrid
from kaiju.utils import plotOne
import sys

seed = 0

# baseline 1.6 sec exp for led1/2 at 1

async def main():
    rg = getGrid(0)



    fps = FPS()
    await fps.initialise()

    await updateCurrentPos(fps, rg)
    plotOne(0, rg, "plotone.png", isSequence=False)
    print("plotted current state")

if __name__ == "__main__":

    # complile some junk

    asyncio.run(main())
