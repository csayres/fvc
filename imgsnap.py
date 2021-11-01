import asyncio
from jaeger import FPS
from fvcLoop import exposeFVC, writeProcFITS, getGrid, getTargetCoords, updateCurrentPos
import sys

seed = 0

# baseline 1.6 sec exp for led1/2 at 1

async def main(exptime, led1, led2):
    rg = getGrid(0)



    fps = FPS()
    await fps.initialise()

    updateCurrentPos(fps, rg)
    expectedTargCoords = getTargetCoords(rg)

    for ledpower, devName in zip([led1, led2], ["led1", "led2"]):
        on_value = 32 * int(1023 * (ledpower) / 100)
        device = fps.ieb.get_device(devName)
        await device.write(on_value)

    filename = await exposeFVC(exptime)
    await writeProcFITS(filename, fps, rg, seed, expectedTargCoords, doProcess=False)

if __name__ == "__main__":

    # complile some junk
    exptime = float(sys.argv[1])
    led1 = float(sys.argv[2])
    led2 = float(sys.argv[3])

    asyncio.run(main(exptime, led1, led2))
