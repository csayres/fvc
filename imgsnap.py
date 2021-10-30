import asyncio
from jaeger import FPS
from fvcLoop import exposeFVC, writeProcFITS, getGrid, getTargetCoords, updateCurrentPosition
import sys

seed = 0

async def main(exptime, led1, led2):
    rg = getGrid(0)



    fps = FPS()
    await fps.initialise()

    updateCurrentPosition(fps, rg)
    expectedTargCoords = getTargetCoords(rg)

    for ledpower, devName in zip([led1, led2], ["led1", "led2"]):
        on_value = 32 * int(1023 * (ledpower) / 100)
        # for dev in ["led1", "led2"]:
        #     print(dev, "on")
        device = fps.ieb.get_device(devName)
        await device.write(on_value)

    filename = await exposeFVC(exptime)
    await writeProcFITS(filename, fps, rg, seed, expectedTargCoords, doProcess=True)

if __name__ == "__main__":

    # complile some junk
    exptime = float(sys.argv[1])
    led1 = float(sys.argv[2])
    led2 = float(sys.argv[3])

    asyncio.run(main(exptime, led1, led2))
