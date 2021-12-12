### this was from the pluglab, illuminating met, ap, boss, then all toghether

# from __future__ import annotations
import coordio
from coordio import defaults
from coordio.transforms import arg_nearest_neighbor, plot_fvc_assignments
from coordio.transforms import transformFromMetData, xyWokFromPosAngles, positionerToWok #RoughTransform, ZhaoBurgeTransform
from coordio.conv import wokToTangent
import numpy
import pandas
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist

positionerTable = coordio.defaults.calibration.positionerTable
wokCoords = coordio.defaults.calibration.wokCoords

fiducialCoords = coordio.defaults.calibration.fiducialCoords.reset_index()
posWokCoords = pandas.merge(positionerTable, wokCoords, on="holeID").reset_index()


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


def measureSciLocationFromMet(xTanSci, yTanSci, xTanMet, yTanMet, alphaLen, alphaOffDeg, alphaDeg):
    xyTanSci = numpy.array([xTanSci, yTanSci])
    xyTanMet = numpy.array([xTanMet, yTanMet])
    rot = numpy.radians(alphaDeg + alphaOffDeg)

    xTanBetaAxis = alphaLen * numpy.cos(rot)
    yTanBetaAxis = alphaLen * numpy.sin(rot)

    xyTanBetaAxis = numpy.array([xTanBetaAxis, yTanBetaAxis])

    xHatBeta = (xyTanMet - xyTanBetaAxis) / numpy.linalg.norm(xyTanMet - xyTanBetaAxis)

    dxyBetaSci = xyTanSci - xyTanMet

    dxBetaSci = dxyBetaSci @ xHatBeta

    dyBetaSci = numpy.sqrt(dxBetaSci**2 + numpy.sum(dxyBetaSci**2))

    # if not isAp:
    #     dyBetaSci *= -1

    # import pdb; pdb.set_trace()

    # plt.figure()
    # plt.plot([0, xyTanBetaAxis[0]], [0, xyTanBetaAxis[1]], '-k')
    # plt.plot([xyTanBetaAxis[0], xyTanMet[0]], [xyTanBetaAxis[1], xyTanMet[1]], '-r')
    # plt.plot([xyTanMet[0], xyTanSci[0]], [xyTanMet[1], xyTanSci[1]], ':g')


    # plt.plot([0, xHatBeta[0]], [0, xHatBeta[1]], '-k')
    # plt.plot([0, dxyBetaSci[0]], [0, dxyBetaSci[1]], '-r')

    # rotate by xHatBeta
    _theta = numpy.arctan2(xHatBeta[1], xHatBeta[0])
    _xxHatBeta = xHatBeta[0]*numpy.cos(-_theta) - xHatBeta[1]*numpy.sin(-_theta)
    _yxHatBeta = xHatBeta[0]*numpy.sin(-_theta) + xHatBeta[1]*numpy.cos(-_theta)

    _dxSci = dxyBetaSci[0]*numpy.cos(-_theta) - dxyBetaSci[1]*numpy.sin(-_theta)
    _dySci = dxyBetaSci[0]*numpy.sin(-_theta) + dxyBetaSci[1]*numpy.cos(-_theta)

    # plt.plot([0, _xxHatBeta], [0, _yxHatBeta], ':k')
    # plt.plot([0, _dxSci], [0, _dySci], ':c')

    # plt.show()

    print("dxySci %.4f, %.4f"%(_dySci, _dxSci))

    import pdb; pdb.set_trace()

    return _dxSci, _dySci


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


def predictWokApBoss(fullTable):

    xWokBoss = []
    yWokBoss = []
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

        # alphaDeg = float(posRow.alphaReport)
        # betaDeg = float(posRow.betaReport)

        # print("xyBetaBoss", xBetaBoss, yBetaBoss)
        # print("xyBetaApogee", xBetaAp, yBetaAp)

        xw, yw, zw = positionerToWok(
            alphaDeg, betaDeg,
            xBetaBoss, yBetaBoss, la,
            alphaOffDeg, betaOffDeg,
            dx, dy, b, iHat, jHat, kHat
        )
        xWokBoss.append(xw)
        yWokBoss.append(yw)

        xw, yw, zw = positionerToWok(
            alphaDeg, betaDeg,
            xBetaAp, yBetaAp, la,
            alphaOffDeg, betaOffDeg,
            dx, dy, b, iHat, jHat, kHat
        )
        xWokAp.append(xw)
        yWokAp.append(yw)

    fullTable["xWokBossExpect"] = xWokBoss
    fullTable["yWokBossExpect"] = yWokBoss
    fullTable["xWokApExpect"] = xWokAp
    fullTable["yWokApExpect"] = yWokAp

    return fullTable


def apBossFiberAnalysis():
    # using danger moves from pluglab, find where apogee
    # and boss fibers are with respect to
    imgDir = "/Volumes/futa/apo/data/fcam/59547/"
    imgNums = numpy.arange(33,332)

    # randomRobotNum = 494 # for determining if confiugration has changed
    bossFiberTables = []
    apogeeFiberTables = []
    # dxyBoss = []
    # dxyApogee = []
    # positionerID = []
    for imgNum in imgNums[:4*10]:
        if imgNum == 100:
            continue
        imgStr = ("%i" % imgNum).zfill(4)
        f = fits.open(imgDir + "proc-fimg-fvc2n-%s.fits" % imgStr)
        metOn, apOn, bossOn = getIllumType(f)

        if metOn and apOn and bossOn:
            # print("\n-----\n")
            continue

        centroids = fitsTableToPandas(f[7].data)

        if metOn:
            # first img in sequence
            # solve for measured alpha/beta coords for each robot
            posAngles = fitsTableToPandas(f[6].data)
            fullTable = pandas.merge(posWokCoords, posAngles, on="positionerID")
            fullTable = xyWokFromPosAngles(fullTable, "Metrology")
            ft, metrologyTable = transformFromMetData(centroids, fullTable, fiducialCoords, figPrefix="%i-metrology-"%imgNum, polids=polids)
            # next predict apogee and boss fiber locations
            # based on measured alpha beta
            metrologyTable = measureAlphaBeta(metrologyTable)
            metrologyTable = predictWokApBoss(metrologyTable)
            lastMetImg = imgNum
            print("\n\n")
            # print("met img number", imgNum)
        if bossOn:
            # print("boss img number", imgNum)
            if not imgNum < lastMetImg + 2:
                # skip it probably wrong association
                print("skipping a boss image")
                continue
            print("n centroids boss", len(centroids))
            xyCCD = centroids[["x", "y"]].to_numpy()
            xyWokMeas = ft.apply(xyCCD)
            bossTable = metrologyTable.copy()
            xyWokExpect = bossTable[["xWokBossExpect", "yWokBossExpect"]].to_numpy()
            positionerIDs = list(bossTable.positionerID)

            plot_fvc_assignments(
                "%i-boss.pdf"%imgNum,
                xyFitCentroids=xyWokMeas,
                xyBossFiber=xyWokExpect,
                positionerIDs = positionerIDs,
                title="Boss Centroids"
            )

            # find nearest neighbors in
            argFound, dist = arg_nearest_neighbor(xyWokExpect, xyWokMeas)


            bossTable["xWokBossMeas"] = xyWokMeas[argFound, 0]
            bossTable["yWokBossMeas"] = xyWokMeas[argFound, 1]
            bossTable["dist"] = dist

            # throw out obvious missassociations
            bossTable = bossTable[bossTable.dist < 1]

            dxFromMet = numpy.array(bossTable.xWokBossMeas - bossTable.xWokMetMeas, dtype=numpy.float64)
            dyFromMet = numpy.array(bossTable.yWokBossMeas - bossTable.yWokMetMeas, dtype=numpy.float64)
            rFromMet = numpy.sqrt(dxFromMet**2 + dyFromMet**2)
            bossTable["rFromMet"] = rFromMet

            thetaRot = numpy.radians(numpy.array(bossTable.alphaOffset + bossTable.betaOffset + bossTable.alphaMeas + bossTable.betaMeas, dtype=numpy.float64))


            dxBeta = []
            dyBeta = []
            for ii, row in bossTable.iterrows():
                dx = float(row.dx)
                dy = float(row.dy)
                b = row[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
                iHat = row[["ix", "iy", "iz"]].to_numpy().squeeze()
                jHat = row[["jx", "jy", "jz"]].to_numpy().squeeze()
                kHat = row[["kx", "ky", "kz"]].to_numpy().squeeze()
                alphaLen = float(row.alphaArmLen)
                alphaOffDeg = float(row.alphaOffset)
                alphaDeg = float(row.alphaMeas)

                print("alpha, beta angle", alphaDeg, float(row.betaMeas))


                xWokMet = float(row.xWokMetMeas)
                yWokMet = float(row.yWokMetMeas)
                xWokBoss = float(row.xWokBossMeas)
                yWokBoss = float(row.yWokBossMeas)

                xTanMet, yTanMet, zTanMet = wokToTangent(
                    xWok=xWokMet, yWok=yWokMet, zWok=defaults.POSITIONER_HEIGHT,
                    b=b, iHat=iHat, jHat=jHat, kHat=kHat,
                    elementHeight=defaults.POSITIONER_HEIGHT, scaleFac=1,
                    dx=dx, dy=dy, dz=0

                )

                xTanBoss, yTanBoss, zTanBoss = wokToTangent(
                    xWok=xWokBoss, yWok=yWokBoss, zWok=defaults.POSITIONER_HEIGHT,
                    b=b, iHat=iHat, jHat=jHat, kHat=kHat,
                    elementHeight=defaults.POSITIONER_HEIGHT, scaleFac=1,
                    dx=dx, dy=dy, dz=0

                )

                xTanMet = xTanMet[0]
                yTanMet = yTanMet[0]
                xTanBoss = xTanBoss[0]
                yTanBoss = yTanBoss[0]

                _dx, _dy = measureSciLocationFromMet(
                    xTanBoss, yTanBoss, xTanMet, yTanMet, alphaLen,
                    alphaOffDeg, alphaDeg
                )


                dxBeta.append(_dx)
                dyBeta.append(_dy)


            # dxBeta = []
            # dyBeta = []
            # for dx, dy, theta in zip(dxFromMet, dyFromMet, thetaRot):
            #     _dxBeta = dx * numpy.cos(-theta) - dy *numpy.sin(-theta)
            #     dxBeta.append(_dxBeta)
            #     _dyBeta = dx * numpy.sin(-theta) + dy *numpy.cos(-theta)
            #     dyBeta.append(_dyBeta)

            # dxBeta = numpy.array(dxBeta)
            # dyBeta = numpy.array(dyBeta)

            xBeta = bossTable.metX + dxBeta
            yBeta = bossTable.metY + dyBeta

            # import pdb; pdb.set_trace()

            # plt.figure()
            # plt.hist(xBeta)
            # plt.xlabel("boss xBeta")

            # plt.figure()
            # plt.hist(yBeta)
            # plt.xlabel("boss yBeta")
            # plt.show()

            bossTable["bossX"] = xBeta
            bossTable["bossY"] = yBeta
            bossFiberTables.append(bossTable.copy())


            # filter
            # plt.hist(metrologyTable)
            # import pdb; pdb.set_trace()
            # print("imgNum", imgNum)
            print("median boss dist um", numpy.median(dist)*1000)
            # plt.hist(dist*1000, bins=numpy.linspace(0,1000, 50))
            # # plt.title("boss %i"%imgNum)
            # # print("max boss dist mm", numpy.max(dist))
            # print("\n")
            # import pdb; pdb.set_trace()

        if apOn:
            # print("apogee img number", imgNum)
            if not imgNum < lastMetImg + 3:
                # skip it probably wrong association
                print("skipping an apogee image")
                continue
            print("n centroids ap", len(centroids))
            xyCCD = centroids[["x", "y"]].to_numpy()
            xyWokMeas = ft.apply(xyCCD)
            apogeeTable = metrologyTable.copy()
            apogeeTable = apogeeTable[apogeeTable.holeType=="ApogeeBoss"]
            positionerIDs = list(apogeeTable.positionerID)
            xyWokExpect = apogeeTable[["xWokApExpect", "yWokApExpect"]].to_numpy()
            plot_fvc_assignments(
                "%i-apogee.pdf"%imgNum,
                xyFitCentroids=xyWokMeas,
                xyApogeeFiber=xyWokExpect,
                positionerIDs=positionerIDs,
                title="Apogee Centroids"
            )

            print("imgNum", imgNum)
            argFound, dist = arg_nearest_neighbor(xyWokExpect, xyWokMeas)
            print("median apogee dist um", numpy.median(dist)*1000)
            # plt.hist(dist*1000, bins=numpy.linspace(0,1000,50))
            # plt.title("apogee %i"%imgNum)
            # plt.show()
            # print("max apogee dist mm", numpy.max(dist))


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
            # thetaRot = numpy.radians(numpy.array(apogeeTable.betaOffset + apogeeTable.betaMeas, dtype=numpy.float64))


            dxBeta = []
            dyBeta = []
            for ii, row in apogeeTable.iterrows():
                dx = float(row.dx)
                dy = float(row.dy)
                b = row[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
                iHat = row[["ix", "iy", "iz"]].to_numpy().squeeze()
                jHat = row[["jx", "jy", "jz"]].to_numpy().squeeze()
                kHat = row[["kx", "ky", "kz"]].to_numpy().squeeze()
                alphaLen = float(row.alphaArmLen)
                alphaOffDeg = float(row.alphaOffset)
                alphaDeg = float(row.alphaMeas)

                print("alpha, beta angle", alphaDeg, float(row.betaMeas))


                xWokMet = float(row.xWokMetMeas)
                yWokMet = float(row.yWokMetMeas)
                xWokApogee = float(row.xWokApogeeMeas)
                yWokApogee = float(row.yWokApogeeMeas)

                xTanMet, yTanMet, zTanMet = wokToTangent(
                    xWok=xWokMet, yWok=yWokMet, zWok=defaults.POSITIONER_HEIGHT,
                    b=b, iHat=iHat, jHat=jHat, kHat=kHat,
                    elementHeight=defaults.POSITIONER_HEIGHT, scaleFac=1,
                    dx=dx, dy=dy, dz=0

                )

                xTanApogee, yTanApogee, zTanApogee = wokToTangent(
                    xWok=xWokApogee, yWok=yWokApogee, zWok=defaults.POSITIONER_HEIGHT,
                    b=b, iHat=iHat, jHat=jHat, kHat=kHat,
                    elementHeight=defaults.POSITIONER_HEIGHT, scaleFac=1,
                    dx=dx, dy=dy, dz=0

                )

                xTanMet = xTanMet[0]
                yTanMet = yTanMet[0]
                xTanApogee = xTanApogee[0]
                yTanApogee = yTanApogee[0]

                _dx, _dy = measureSciLocationFromMet(
                    xTanApogee, yTanApogee, xTanMet, yTanMet, alphaLen,
                    alphaOffDeg, alphaDeg
                )


                dxBeta.append(_dx)
                dyBeta.append(_dy)

                # import pdb; pdb.set_trace()



            # dxBeta = []
            # dyBeta = []
            # for dx, dy, theta in zip(dxFromMet, dyFromMet, thetaRot):
            #     _dxBeta, _dyBeta = measureSciLocationFromMet(xWokSci, yWokSci, xWokMet, yWokMet, xWokBetaAxis, yWokBetaAxis, isAp=True)
            #     theta = theta % (numpy.pi * 2)
            #     _dxBeta = dx * numpy.cos(-1*theta) - dy *numpy.sin(-1*theta)
            #     dxBeta.append(_dxBeta)
            #     _dyBeta = dx * numpy.sin(-1*theta) + dy *numpy.cos(-1*theta)
                #dyBeta.append(_dyBeta)

            dxBeta = numpy.array(dxBeta)
            dyBeta = numpy.array(dyBeta)

            xBeta = apogeeTable.metX + dxBeta
            yBeta = apogeeTable.metY + dyBeta

            # import pdb; pdb.set_trace()

            # import pdb; pdb.set_trace()

            # plt.figure()
            # plt.hist(xBeta)
            # plt.xlabel("apogee xBeta")

            # plt.figure()
            # plt.hist(yBeta)
            # plt.xlabel("apogee yBeta")
            # plt.show()

            apogeeTable["apX"] = xBeta
            apogeeTable["apY"] = yBeta
            apogeeFiberTables.append(apogeeTable.copy())

            # import pdb; pdb.set_trace()
    # bft = pandas.concat(bossFiberTables)
    # bft.to_csv("bftLab.csv")

    aft = pandas.concat(apogeeFiberTables)
    aft.to_csv("aftLab.csv")

if __name__ == "__main__":
    apBossFiberAnalysis()
    aft = pandas.read_csv("aftLab.csv")
    print("len aft labl", len(aft))
    bft = pandas.read_csv("bftLab.csv")

    bossX = []
    bossY = []
    apX = []
    apY = []
    stdr = []
    for positionerID in positionerTable.positionerID.to_numpy():
        xx = aft[aft.positionerID == positionerID]
        if len(xx) == 0:
            # print("no apogee fiber here, use default")
            nomRow = positionerTable[positionerTable.positionerID==positionerID]
            apX.append(float(nomRow.apX))
            apY.append(float(nomRow.apY))
            stdr.append(-1)
            continue
        apX.append(numpy.median(xx.apX))
        apY.append(numpy.median(xx.apY))
        stdr.append(numpy.std(xx.rFromMet))

    for positionerID in positionerTable.positionerID.to_numpy():
        xx = bft[bft.positionerID == positionerID]

        # plt.figure()
        # plt.hist(xx.bossX)
        # plt.show()
        # import pdb; pdb.set_trace()

        bossX.append(numpy.median(xx.bossX))
        bossY.append(numpy.median(xx.bossY))

        # import pdb; pdb.set_trace()

    positionerTable["bossX"] = bossX
    positionerTable["bossY"] = bossY
    positionerTable["apX"] = apX
    positionerTable["apY"] = apY
    positionerTable["stdr"] = stdr
    positionerTable = positionerTable.reset_index()
    positionerTable.to_csv("positionerTableUpdateSciFibers.csv")

    plt.figure()
    plt.hist(positionerTable.apX, bins=100)
    plt.xlabel("apX")

    plt.figure()
    plt.hist(positionerTable.apY, bins=100)
    plt.xlabel("apY")

    plt.figure()
    plt.hist(positionerTable[positionerTable.stdr >= 0]["stdr"], bins=100)
    plt.xlabel("stdr")

    plt.figure()
    plt.hist(positionerTable.bossX, bins=100)
    plt.xlabel("bossX")

    plt.figure()
    plt.hist(positionerTable.bossY, bins=100)
    plt.xlabel("bossY")

    plt.show()
    # import pdb; pdb.set_trace()



    # _nextAlpha = f[5].data[0]["alpha"]
    # print()
    # import pdb; pdb.set_trace()




