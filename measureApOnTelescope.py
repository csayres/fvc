### this was from the pluglab, illuminating met, ap, boss, then all toghether

# from __future__ import annotations
import coordio
from coordio.transforms import arg_nearest_neighbor, plot_fvc_assignments
from coordio.transforms import transformFromMetData, xyWokFromPosAngles, positionerToWok #RoughTransform, ZhaoBurgeTransform
import numpy
import pandas
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import glob

positionerTable = coordio.defaults.calibration.positionerTable
wokCoords = coordio.defaults.calibration.wokCoords

fiducialCoords = coordio.defaults.calibration.fiducialCoords.reset_index()
posWokCoords = pandas.merge(positionerTable, wokCoords, on="holeID").reset_index()
fiberAssignments = coordio.defaults.calibration.fiberAssignments.reset_index()

hitsTable = pandas.read_csv("posHits.csv")

# confsummary /home/sdss5/software/sdsscore/main/apo/summary_files/0000XX/confSummary-55.par

# keep = hitsTable["snr"] > 10
# hitsTable = hitsTable[keep]
# print("number of hits", len(hitsTable))
# import pdb; pdb.set_trace()


polids = numpy.array(
    [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],
    dtype=numpy.int16
)


def fitsTableToPandas(recarray):
    d = {}
    for name in recarray.names:
        d[name] = recarray[name].byteswap().newbyteorder()
    return pandas.DataFrame(d)


def _tangentToPositioner(xt, yt, xBeta, yBeta, la, alphaOffDeg, betaOffDeg):
    # new implementation
    # http://motion.pratt.duke.edu/RoboticSystems/InverseKinematics.html
    radDistT2 = xt**2 + yt**2
    radDistT = numpy.sqrt(radDistT2)
    thetaT = numpy.degrees(numpy.arctan2(yt, xt)) % 360
    la2 = la**2
    lb2 = xBeta**2 + yBeta**2
    lb = numpy.sqrt(lb2)

    # solve *exactly* account for offsets at the end
    if radDistT >= la + lb:
        # outside donut

        betaAngleRight = numpy.array([0])
        betaAngleLeft = betaAngleRight

        alphaAngleRight = numpy.array([thetaT])
        alphaAngleLeft = alphaAngleRight
        err = radDistT - (la + lb)
        # print("outside donut", err*1000)

    elif radDistT <= lb - la:
        # inside donut
        betaAngleRight = numpy.array([180])
        betaAngleLeft = betaAngleRight

        alphaAngleRight = numpy.array([(thetaT + 180) % 360])
        alphaAngleLeft = alphaAngleRight
        err = (lb - la) - radDistT
        # print("inside donut", err*1000)
    else:
        # in workspace
        c_2 = (radDistT2 - la2 - lb2) / (2*la*lb)
        betaAngleRight = numpy.arccos(c_2)
        betaAngleLeft = -1*numpy.arccos(c_2)

        alphaAngleRight = thetaT - numpy.degrees(
            numpy.arctan2(
                lb*numpy.sin(betaAngleRight),
                la + lb*numpy.cos(betaAngleRight)
            )
        )

        alphaAngleLeft = thetaT - numpy.degrees(
            numpy.arctan2(
                lb*numpy.sin(betaAngleLeft),
                la + lb*numpy.cos(betaAngleLeft)
            )
        )
        betaAngleRight = numpy.degrees(betaAngleRight) % 360
        betaAngleLeft = numpy.degrees(betaAngleLeft) % 360
        alphaAngleRight = alphaAngleRight % 360
        alphaAngleLeft = alphaAngleLeft % 360
        err = 0


    # account for offsets here
    # and slight angle off the robot arm (xyBeta)
    # fiber angular offset is the angle the fiber makes
    # with the "centerline" of the robot
    # alpha/beta should be reported with respect to the centerline
    # not the line connecting the beta axis and the fiber
    # eg fiberAngOff == 0 for a perfectly centered metrology fiber
    fibAngOff = numpy.degrees(numpy.arctan2(yBeta, xBeta))

    betaAngleRight = betaAngleRight - betaOffDeg - fibAngOff
    betaAngleLeft = betaAngleLeft - betaOffDeg - fibAngOff

    alphaAngleRight = alphaAngleRight - alphaOffDeg
    alphaAngleLeft = alphaAngleLeft - alphaOffDeg

    return [alphaAngleRight, betaAngleRight, alphaAngleLeft, betaAngleLeft, err]


def wokToPositioner(
        xWok, yWok,
        xBeta, yBeta, la,
        alphaOffDeg, betaOffDeg,
        dx, dy, b, iHat, jHat, kHat, new=True
    ):
    zWok = coordio.defaults.POSITIONER_HEIGHT
    xt, yt, zt = coordio.conv.wokToTangent(
        xWok, yWok, zWok, b, iHat, jHat, kHat,
        coordio.defaults.POSITIONER_HEIGHT,
        scaleFac=1, dx=dx, dy=dy, dz=0
    )

    if new:
        out = _tangentToPositioner(xt, yt, xBeta, yBeta, la, alphaOffDeg, betaOffDeg)
    else:
        out = coordio.conv.tangentToPositioner(xt, yt, xBeta, yBeta, la, alphaOffDeg, betaOffDeg)
    # print("out", out)
    return out[0], out[1]



def getIllumType(ff):
    metOn = ff[1].header["LED1"] > 0 and ff[1].header["LED2"] > 0
    apOn = ff[1].header["LED3"] > 0
    bossOn = ff[1].header["LED4"] > 0
    return metOn, apOn, bossOn



def measureAlphaBeta(fullTable):
    # xy wok meas must be from metrology fiber
    alpha = []
    beta = []

    for ii, posRow in fullTable.iterrows():

        b = numpy.array([posRow.xWok, posRow.yWok, posRow.zWok])
        iHat = numpy.array([posRow.ix, posRow.iy, posRow.iz])
        jHat = numpy.array([posRow.jx, posRow.jy, posRow.jz])
        kHat = numpy.array([posRow.kx, posRow.ky, posRow.kz])
        la = float(posRow.alphaArmLen)
        alphaOffDeg = float(posRow.alphaOffset)
        betaOffDeg = float(posRow.betaOffset)
        dx = float(posRow.dx)
        dy = float(posRow.dy)
        xBeta = float(posRow.metX)
        yBeta = float(posRow.metY)
        xWok = float(posRow.xWokMetMeas)
        yWok = float(posRow.yWokMetMeas)

        out = wokToPositioner(
            xWok, yWok,
            xBeta, yBeta, la,
            alphaOffDeg, betaOffDeg,
            dx, dy, b, iHat, jHat, kHat
        )

        alphaAngleRight, betaAngleRight = out

        # import pdb; pdb.set_trace()

        if numpy.isnan(alphaAngleRight[0]):
            alphaAngleRight = [-999]
        if numpy.isnan(betaAngleRight[0]):
            betaAngleRight = [-999]
        alpha.append(alphaAngleRight[0])
        beta.append(betaAngleRight[0])

    fullTable["alphaMeas"] = alpha
    fullTable["betaMeas"] = beta
    return fullTable


def predictWokAp(fullTable):
    xWokAp = []
    yWokAp = []

    for ii, posRow in fullTable.iterrows():
        b = numpy.array([posRow.xWok, posRow.yWok, posRow.zWok])
        iHat = numpy.array([posRow.ix, posRow.iy, posRow.iz])
        jHat = numpy.array([posRow.jx, posRow.jy, posRow.jz])
        kHat = numpy.array([posRow.kx, posRow.ky, posRow.kz])
        la = float(posRow.alphaArmLen)
        alphaOffDeg = float(posRow.alphaOffset)
        betaOffDeg = float(posRow.betaOffset)
        dx = float(posRow.dx)
        dy = float(posRow.dy)
        xBetaBoss = float(posRow.bossX)
        yBetaBoss = float(posRow.bossY)
        xBetaAp = float(posRow.apX)
        yBetaAp = float(posRow.apY)
        alphaDeg = float(posRow.alphaMeas)
        betaDeg = float(posRow.betaMeas)

        xw, yw, zw = positionerToWok(
            alphaDeg, betaDeg,
            xBetaAp, yBetaAp, la,
            alphaOffDeg, betaOffDeg,
            dx, dy, b, iHat, jHat, kHat
        )
        xWokAp.append(xw)
        yWokAp.append(yw)

    fullTable["xWokApExpect"] = xWokAp
    fullTable["yWokApExpect"] = yWokAp

    return fullTable


def getImgs():
    allFiles = glob.glob("/Volumes/futa/apo/data/fcam/59557/proc*.fits")
    allFiles = sorted(allFiles)

    keepFiles = []
    for f in allFiles:
        ff = fits.open(f)
        if len(ff[5].columns) > 12:
            continue
        keepFiles.append(f)

    return keepFiles



def apFiberAnalysis():
    ### met and apogee fiber on for fitting metrology vs apogee location

    imgs = getImgs()
    # randomRobotNum = 494 # for determining if confiugration has changed
    apogeeFiberTables = []
    # dxyBoss = []
    # dxyApogee = []
    # positionerID = []
    for img in imgs:
        if "proc-fimg-fvc1n-0117.fits" in img:
            break # the rest of the sequence has no apogee fiber illumation
        f = fits.open(img)
        metOn, apOn, bossOn = getIllumType(f)

        centroids = fitsTableToPandas(f[8].data)

        if metOn:
            print("got met")
            # first img in sequence
            # solve for measured alpha/beta coords for each robot
            posAngles = fitsTableToPandas(f[7].data)
            fullTable = pandas.merge(posWokCoords, posAngles, on="positionerID")
            fullTable = xyWokFromPosAngles(fullTable, "Metrology")
            ft, metrologyTable = transformFromMetData(centroids, fullTable, fiducialCoords, figPrefix="%s-metrology-"%img, polids=polids)
            # next predict apogee and boss fiber locations
            # based on measured alpha beta
            metrologyTable = measureAlphaBeta(metrologyTable)
            metrologyTable = predictWokAp(metrologyTable)
            # print("\n\n")
            # print("met img number", imgNum)
        if apOn:
            print("--got ap--")
            # print("apogee img number", imgNum)

            xyCCD = centroids[["x", "y"]].to_numpy()
            xyWokMeas = ft.apply(xyCCD)
            apogeeTable = metrologyTable.copy()
            apogeeTable = apogeeTable[apogeeTable.holeType=="ApogeeBoss"]
            positionerIDs = list(apogeeTable.positionerID)
            xyWokExpect = apogeeTable[["xWokApExpect", "yWokApExpect"]].to_numpy()
            plot_fvc_assignments(
                "%s-apogee.pdf"%img,
                xyFitCentroids=xyWokMeas,
                xyApogeeFiber=xyWokExpect,
                positionerIDs=positionerIDs,
                title="Apogee Centroids"
            )

            argFound, dist = arg_nearest_neighbor(xyWokExpect, xyWokMeas)
            print("median apogee dist um", numpy.median(dist)*1000)



            apogeeTable["xWokApogeeMeas"] = xyWokMeas[argFound, 0]
            apogeeTable["yWokApogeeMeas"] = xyWokMeas[argFound, 1]
            apogeeTable["dist"] = dist

            # throw out obvious missassociations
            apogeeTable = apogeeTable[apogeeTable.dist < 1]

            dxFromMet = numpy.array(apogeeTable.xWokApogeeMeas - apogeeTable.xWokMetMeas, dtype=numpy.float64)
            dyFromMet = numpy.array(apogeeTable.yWokApogeeMeas - apogeeTable.yWokMetMeas, dtype=numpy.float64)
            rFromMet = numpy.sqrt(dxFromMet**2 + dyFromMet**2)
            apogeeTable["rFromMet"] = rFromMet

            thetaRot = numpy.radians(numpy.array(apogeeTable.alphaOffset + apogeeTable.betaOffset + apogeeTable.alphaMeas + apogeeTable.betaMeas, dtype=numpy.float64))

            dxBeta = []
            dyBeta = []
            for dx, dy, theta in zip(dxFromMet, dyFromMet, thetaRot):
                _dxBeta = dx * numpy.cos(-theta) - dy *numpy.sin(-theta)
                dxBeta.append(_dxBeta)
                _dyBeta = dx * numpy.sin(-theta) + dy *numpy.cos(-theta)
                dyBeta.append(_dyBeta)

            dxBeta = numpy.array(dxBeta)
            dyBeta = numpy.array(dyBeta)

            xBeta = apogeeTable.metX + dxBeta
            yBeta = apogeeTable.metY + dyBeta

            apogeeTable["apX"] = xBeta
            apogeeTable["apY"] = yBeta
            apogeeFiberTables.append(apogeeTable.copy())

            # import pdb; pdb.set_trace()
    aft = pandas.concat(apogeeFiberTables)
    aft.to_csv("aftTelescope.csv")

if __name__ == "__main__":

    apFiberAnalysis()

    aft = pandas.read_csv("aftTelescope.csv")
    print("len aft tel", len(aft))
    # bft = pandas.read_csv("bft.csv")

    apX = []
    apY = []
    stdr = []
    for positionerID in positionerTable.positionerID.to_numpy():
        xx = aft[aft.positionerID == positionerID]
        if len(xx) == 0:
            # print("no apogee fiber here, use default")
            nomRow = positionerTable[positionerTable.positionerID==positionerID]
            apX.append(float(14.965))
            apY.append(float(0.376))
            stdr.append(-1)
            continue
        apX.append(numpy.median(xx.apX))
        apY.append(numpy.median(xx.apY))
        stdr.append(numpy.std(xx.rFromMet))



    # positionerTable["bossX"] = bossX
    # positionerTable["bossY"] = bossY
    # positionerTable["apX_orig"] = positionerTable["apX"]
    # positionerTable["apY_orig"] = positionerTable["apY"]

    positionerTable["apX"] = apX
    positionerTable["apY"] = apY
    positionerTable["stdr"] = stdr
    positionerTable = positionerTable.reset_index()
    positionerTable.to_csv("positionerTableUpdateSciFibersTelescope.csv")

    plt.figure()
    plt.hist(positionerTable.apX)
    plt.xlabel("apX")

    plt.figure()
    plt.hist(positionerTable.apY)
    plt.xlabel("apY")

    plt.figure()
    plt.hist(positionerTable[positionerTable.stdr >=0]["stdr"], bins=100)
    plt.xlabel("stdr")

    # plt.figure()
    # plt.hist(positionerTable.bossX)
    # plt.xlabel("bossX")

    # plt.figure()
    # plt.hist(positionerTable.bossY)
    # plt.xlabel("bossY")

    # plt.show()
    # # import pdb; pdb.set_trace()

    aftLab = pandas.read_csv("aftTelescope.csv")
    ptLab = pandas.read_csv("positionerTableUpdateSciFibers.csv")

    dx = []
    dy = []
    hitDX = []
    hitDY = []
    hitFlux = []
    for rID in positionerTable.positionerID:
        tr = positionerTable[positionerTable.positionerID == rID]
        holeID = str(tr.holeID.values[0])
        lr = ptLab[ptLab.positionerID == rID]

        _dx = float(tr["apX"] - lr["apX"])
        _dy = float(tr["apY"] - lr["apY"])
        dx.append(_dx)
        dy.append(_dy)

        # import pdb; pdb.set_trace()

        apFiberID = float(fiberAssignments[fiberAssignments.holeID==holeID]["APOGEEFiber"])

        if apFiberID in hitsTable.APOGEEFiber.to_numpy():
            print("got hit for fiber", apFiberID)
            hitDX.append(_dx)
            hitDY.append(_dy)
            flux = hitsTable[hitsTable.APOGEEFiber==apFiberID]["flux"]
            hitFlux.append(float(flux))
            # hitFlux.append(float(hitsTable[hitsTable.APOGEEFiber==apFiberID]["flux"].values[0]))

        # if numpy.isnan(tr.APOGEEFiber.to_numpy()):
        #     continue


        # apFiberID = int(tr.APOGEEFiber)
        # print("apFiberID", apFiberID)

        # if apFiberID in hitsTable.APOGEEFiber.to_numpy():
        #     print("got fiberID", apFiberID)
        # print("at holeID", holeID)
        # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()




    plt.figure()
    plt.hist(dx, 100)
    plt.xlabel("dx")

    plt.figure()
    plt.hist(dy, 100)
    plt.xlabel("dy")

    plt.figure()
    plt.plot(dx, dy, ".k", alpha=0.5)
    plt.scatter(hitDX, hitDY, s=numpy.array(hitFlux)/10, color="red")

    X = numpy.array([[1]*len(dx), dx]).T
    coeff = numpy.linalg.lstsq(X,dy)[0]
    # import pdb; pdb.set_trace()
    ymodel = coeff[0] + coeff[1]*numpy.array(dx)

    plt.figure()
    plt.plot(dx, ymodel-numpy.array(dy), '.', color="black")

    # import pdb; pdb.set_trace()


    plt.show()




    # # _nextAlpha = f[5].data[0]["alpha"]
    # # print()
    # # import pdb; pdb.set_trace()




