import os
import fitsio
import sep
import numpy
import glob
import time
from multiprocessing import Pool

import pandas as pd

import coordio

from coordio.defaults import positionerTable, wokCoords, fiducialCoords

# configDir = os.path.dirname(coordio.__file__) + "/etc/uwMiniWok"
# positionerTable = pd.read_csv(configDir + "/positionerTable.csv")
# wokCoords = pd.read_csv(configDir + "/wokCoords.csv")
# fiducialCoords = pd.read_csv(configDir + "/fiducialCoords.csv")


def positionerToWok(
        positionerID, alphaDeg, betaDeg,
        xBeta=None, yBeta=None, la=None,
        alphaOffDeg=None, betaOffDeg=None,
        dx=None, dy=None #, dz=None
    ):
    posRow = positionerTable[positionerTable.positionerID == positionerID]

    if xBeta is None:
        xBeta = posRow.metX
    if yBeta is None:
        yBeta = posRow.metY
    if la is None:
        la = posRow.alphaArmLen
    if alphaOffDeg is None:
        alphaOffDeg = posRow.alphaOffset
    if betaOffDeg is None:
        betaOffDeg = posRow.betaOffset
    if dx is None:
        dx = posRow.dx
    if dy is None:
        dy = posRow.dy
    # if dz is None:
    #     dz = posRow.dz

    xt, yt = coordio.conv.positionerToTangent(
        alphaDeg, betaDeg, xBeta, yBeta,
        la, alphaOffDeg, betaOffDeg
    )

    if hasattr(xt, "__len__"):
        zt = numpy.zeros(len(xt))
    else:
        zt = 0

    # import pdb; pdb.set_trace()

    wokRow = wokCoords[wokCoords.holeID == posRow.holeID.values[0]]

    b = wokRow[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
    iHat = wokRow[["ix", "iy", "iz"]].to_numpy().squeeze()
    jHat = wokRow[["jx", "jy", "jz"]].to_numpy().squeeze()
    kHat = wokRow[["kx", "ky", "kz"]].to_numpy().squeeze()

    xw, yw, zw = coordio.conv.tangentToWok(
        xt, yt, zt, b, iHat, jHat, kHat,
        elementHeight=coordio.defaults.POSITIONER_HEIGHT, scaleFac=1,
        dx=dx, dy=dy, dz=0

    )

    return xw, yw, zw, xt, yt, b


def getCentroids(imgdata): #, peakThresh=1000):
    objects = sep.extract(imgdata, 2.5)
    # objects = objects[objects["peak"] > peakThresh]
    return objects["x"], objects["y"]


def solveImg(imgfile):
    # find the cmd file based on the imgfile name
    cmdfile = imgfile.strip("_on.fits") + ".csv"
    if not os.path.exists(cmdfile):
        return

    imgdata = fitsio.read(imgfile)
    cmddata = pd.read_csv(cmdfile)

    xCCD, yCCD = getCentroids(imgdata)

    # do rough conversion to wok coords to match
    fvc = coordio.conv.FVCUW()
    xWokCentroid, yWokCentroid = fvc.fvcToWok(xCCD, yCCD)

    # find fiducials in image
    xFidNom = fiducialCoords.xWok.to_numpy()
    yFidNom = fiducialCoords.yWok.to_numpy()
    xFidMeas = []
    yFidMeas = []
    errx = []
    erry = []
    for _xfid, _yfid in zip(xFidNom, yFidNom):
        dx = xWokCentroid - _xfid
        dy = yWokCentroid - _yfid
        norm = numpy.sqrt(dx**2+dy**2)
        amin = numpy.argmin(norm)
        errx.append(dx[amin])
        erry.append(dy[amin])
        xFidMeas.append(xCCD[amin])
        yFidMeas.append(yCCD[amin])

    # refit the image model
    fvc.fit(xFidMeas, yFidMeas, xFidNom, yFidNom)
    xWokCentroid, yWokCentroid = fvc.fvcToWok(xCCD, yCCD)


    _robotID = []
    _xCCD = []
    _yCCD = []
    _xWokExpect = []
    _yWokExpect = []
    _xTanExpect = []
    _yTanExpect = []
    _bx = []
    _by = []
    _xWokMeas = []
    _yWokMeas = []
    _cmdAlpha = []
    _cmdBeta = []
    _reportAlpha = []
    _reportBeta = []
    _err = []
    for ii, row in cmddata.iterrows():
        xw, yw, zw, xt, yt, b = positionerToWok(
            row.robotID, row.cmdAlpha, row.cmdBeta
        )

        dx = xWokCentroid - xw
        dy = yWokCentroid - yw
        norm = numpy.sqrt(dx**2+dy**2)
        amin = numpy.argmin(norm)
        err = norm[amin]
        if err > 1.5:
            print("no centroid found", row.robotID)
            continue
        _robotID.append(row.robotID)
        _err.append(err)
        _xWokExpect.append(xw)
        _yWokExpect.append(yw)
        _xTanExpect.append(xt)
        _yTanExpect.append(yt)
        _bx.append(b[0])
        _by.append(b[1])
        _xCCD.append(xCCD[amin])
        _yCCD.append(yCCD[amin])
        _xWokMeas.append(xWokCentroid[amin])
        _yWokMeas.append(yWokCentroid[amin])
        _cmdAlpha.append(row.cmdAlpha)
        _cmdBeta.append(row.cmdBeta)
        _reportAlpha.append(row.reportAlpha)
        _reportBeta.append(row.reportBeta)

    nElements = len(_xCCD)
    _fvcScale = [fvc.tform.scale] * nElements
    _fvcRot = [fvc.tform.rotation] * nElements
    _fvcTransX = [fvc.tform.translation[0]] * nElements
    _fvcTransY = [fvc.tform.translation[1]] * nElements
    _imgFile = [imgfile] * nElements
    _folded = ["_targ_" not in imgfile] * nElements
    _nDetections = [len(xCCD)] * nElements
    _nAssociations = [len(_xCCD)] * nElements

    d = {}
    d["robotID"] = _robotID
    d["xCCD"] = _xCCD
    d["yCCD"] = _yCCD
    d["xWokExpect"] = _xWokExpect
    d["yWokExpect"] = _yWokExpect
    d["xWokMeas"] = _xWokMeas
    d["yWokMeas"] = _yWokMeas
    d["xTanExpect"] = _xTanExpect
    d["yTanExpect"] = _yTanExpect
    d["bx"] = _bx
    d["by"] = _by
    d["err"] = _err
    d["cmdAlpha"] = _cmdAlpha
    d["cmdBeta"] = _cmdBeta
    d["reportAlpha"] = _reportAlpha
    d["reportBeta"] = _reportBeta
    d["fvcScale"] = _fvcScale
    d["fvcRot"] = _fvcRot
    d["fvcTransX"] = _fvcTransX
    d["fvcTransY"] = _fvcTransY
    d["imgFile"] = _imgFile
    d["folded"] = _folded
    d["nDetect"] = _nDetections
    d["nAssoc"] = _nAssociations

    df = pd.DataFrame(d)

    outfilename = "proc/" + cmdfile.split("/")[-1].strip(".csv") + "_proc.csv"
    df.to_csv(outfilename, index=False)
    return df
    # write data to file

    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    ons = glob.glob("data/*_on.fits")
    p = Pool(11)
    tstart = time.time()
    dfs = p.map(solveImg, ons)
    tend = time.time()
    print("took %.2f seconds for %i images"%(tend-tstart, len(ons)))

    # condcat all results
    allOut = pd.concat(dfs, ignore_index=True)
    allOut.to_csv("proc/all.csv", index=False)
