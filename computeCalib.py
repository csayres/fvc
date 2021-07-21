import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform
from scipy.optimize import minimize
from coordio import defaults
from coordio.defaults import positionerTable

from extract import positionerToWok

import time
import os

df = pd.read_csv("proc/all.csv")
# red = df.groupby("imgFile").first().reset_index()

# plt.hist(red.fvcTransX)
# plt.show()

# plt.hist(red.fvcTransY)
# plt.show()



# rIDs = list(set(df.robotID))

rIDs = positionerTable.positionerID
# print(rIDs)
# print(df.columns)


def forwardModel(x, robotID, alpha, beta):
    xBeta, la, alphaOff, betaOff, dx, dy = x
    xw, yw, zw, xt, yt, b = positionerToWok(
        robotID, alpha, beta,
        xBeta=xBeta, la=la,
        alphaOffDeg=alphaOff, betaOffDeg=betaOff,
        dx=dx, dy=dy
    )
    return xw, yw


def minimizeMe(x, robotID, alpha, beta, xWok, yWok):
    xw, yw = forwardModel(x, robotID, alpha, beta)
    return numpy.sum((xw-xWok)**2 + (yw-yWok)**2)


def fitCalibs():
    df = pd.read_csv("proc/all.csv")
    dfTarg = df[df.folded == 0]
    calibs = []
    for rID in rIDs:
        dfR = dfTarg[dfTarg.robotID==rID]
        xExpect = dfR.xWokExpect.to_numpy()
        yExpect = dfR.yWokExpect.to_numpy()
        xMeas = dfR.xWokMeas.to_numpy()
        yMeas = dfR.yWokMeas.to_numpy()

        #minimize
        x0 = numpy.array([
            defaults.MET_BETA_XY[0], defaults.ALPHA_LEN,
            0, 0, 0, 0
        ])
        args = (rID, dfR.cmdAlpha, dfR.cmdBeta, xMeas, yMeas)
        tstart = time.time()
        out = minimize(minimizeMe, x0, args, method="Powell")
        # print(out.x - x0)
        print("alpha arm len", out.x[1])
        tend = time.time()
        print("took %.2f"%(tend-tstart))

        xFit, yFit = forwardModel(out.x, rID, dfR.cmdAlpha, dfR.cmdBeta)



        # fit translation rotation scale
        # tf = EuclideanTransform()
        # xyMeas = numpy.array([xMeas, yMeas]).T
        # xyExpect = numpy.array([xExpect, yExpect]).T
        # tf.estimate(xyMeas, xyExpect)
        # xyFit = tf(xyMeas)
        dx = (xMeas-xFit)*1000
        dy = (yMeas-yFit)*1000
        sqErr = dx**2 + dy**2

        print(rID, numpy.sqrt(numpy.sum(sqErr)/len(dx)))
        calibs.append(out.x)

    return calibs

def updatePositionerTable(calibs):
    _alphaArmLen = []
    _metX = []
    _apX = []
    _bossX = []
    _alphaOffset = []
    _betaOffset = []
    _dx = []
    _dy = []
    for calib, (ii, row) in zip(calibs, positionerTable.iterrows()):
        xBeta, la, alphaOff, betaOff, dx, dy = calib
        print(row)
        # adjust the fiber positions based on xBeta solution
        dxbeta = float(row.metX) - xBeta
        # row.metx - _dx = xbeta
        _metX.append(xBeta)
        _apX.append(float(row.apX) - dxbeta)
        _bossX.append(float(row.bossX) - dxbeta)
        _alphaArmLen.append(la)
        # invert the offsets solved
        _alphaOffset.append(-1*alphaOff)
        _betaOffset.append(-1*betaOff)
        _dx.append(-1*dx)
        _dy.append(-1*dy)

    # overwrite file
    positionerTable["alphaArmLen"] = _alphaArmLen
    positionerTable["metX"] = _metX
    positionerTable["apX"] = _apX
    positionerTable["bossX"] = _bossX
    positionerTable["alphaOffset"] = _alphaOffset
    positionerTable["betaOffset"] = _betaOffset
    positionerTable["dx"] = _dx
    positionerTable["dy"] = _dy

    positionerTable.to_csv(os.environ["WOKCALIB_DIR"] + "/positionerTable.csv", index=False)

if __name__ == "__main__":
    calibs = fitCalibs()
    updatePositionerTable(calibs)


    # print("robot", rID, len(xExpect))
    # plt.figure()
    # plt.hist(dx)
    # plt.xlim([-500, 500])
    # plt.title("dx")

    # plt.figure()
    # plt.hist(dy)
    # plt.xlim([-500, 500])
    # plt.title("dy")

    # plt.figure()
    # plt.hist(numpy.sqrt(sqErr))
    # plt.title("err microns")


    # plt.figure()
    # plt.quiver(xExpect, yExpect, dx, dy, angles="xy")
    # plt.title("%i"%rID)
    # plt.axis("equal")
    # plt.show()




# sns.scatterplot(x="xWokExpect", y="yWokExpect", hue="err", data=dfTarg)

# plt.axis("equal")
# plt.show()
# import pdb; pdb.set_trace()