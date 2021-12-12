import pandas
import numpy
import time
import matplotlib.pyplot as plt
from coordio.utils import radec2wokxy
from coordio.conv import wokToTangent, tangentToWok, tangentToGuide, guideToTangent
from coordio.defaults import calibration, POSITIONER_HEIGHT
from coordio.transforms import arg_nearest_neighbor
import sep
from astropy.io import fits
from astropy.time import Time
import pandas
from skimage.exposure import equalize_hist
from matplotlib.patches import Ellipse
from skimage.transform import SimilarityTransform, EuclideanTransform

gaiaEpoch = 2457206
gfaCoords = calibration.gfaCoords.reset_index()
sigDetect = 3.5


def rescaleImgData(imgData):
    origShape = imgData.shape
    fl = imgData.flatten()
    aw = numpy.argwhere(fl < 0)
    fl[aw] = fl[aw] + 2*2**15
    imgData = fl.reshape(origShape)
    return imgData


def wokToGFA(xWok, yWok, gfaID):
    gfaRow = gfaCoords[gfaCoords.id==gfaID]
    b = gfaRow[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
    # b = b - numpy.array([-3.464, 0.079, 0])
    iHat = gfaRow[["ix", "iy", "iz"]].to_numpy().squeeze()
    jHat = gfaRow[["jx", "jy", "jz"]].to_numpy().squeeze()
    kHat = gfaRow[["kx", "ky", "kz"]].to_numpy().squeeze()

    xt, yt, zt = wokToTangent(xWok, yWok, [POSITIONER_HEIGHT]*len(xWok), b, iHat, jHat, kHat)

    xPix, yPix = tangentToGuide(xt, yt)

    return xPix, yPix


def GFAToWok(xPix, yPix, gfaID):
    gfaRow = gfaCoords[gfaCoords.id==gfaID]
    b = gfaRow[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
    # b = b - numpy.array([-3.464, 0.079, 0])
    iHat = gfaRow[["ix", "iy", "iz"]].to_numpy().squeeze()
    jHat = gfaRow[["jx", "jy", "jz"]].to_numpy().squeeze()
    kHat = gfaRow[["kx", "ky", "kz"]].to_numpy().squeeze()

    xt, yt = guideToTangent(xPix, yPix)
    zt = 0

    xWok, yWok, zWok = tangentToWok(xt, yt, [zt]*len(xt), b, iHat, jHat, kHat)

    return xWok, yWok


def plotSEP(imgData, ax):
    bkg = sep.Background(imgData)
    imgBack = bkg.back()
    imgDataSub = imgData - imgBack
    centroids = sep.extract(imgDataSub, sigDetect, err=bkg.globalrms)
    # centroids = pandas.DataFrame(centroids)
    keep = centroids["npix"] > 200
    centroids = centroids[keep]
    # import pdb; pdb.set_trace()
    ax.imshow(equalize_hist(imgData), origin="lower")

    for cent in centroids:
        e = Ellipse(xy=(cent["x"], cent["y"]),
            width=6*cent["a"],
            height=6*cent["b"],
            angle=cent["theta"] * 180. / numpy.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)

    return centroids


dtypes = {
    "solution_id": numpy.int64,
    "source_id": numpy.int64,
    "ra": numpy.float64,
    "dec": numpy.float64,
    "mag": numpy.float64,
    "parallax": numpy.float64,
    "pmra": numpy.float64,
    "pmdec": numpy.float64
}

names = list(dtypes.keys())

tstart = time.time()
gaia = pandas.read_csv("gaiaFieldsAll.txt", sep="|", skipinitialspace=True, names=names, dtype=dtypes)
print("load csv took", (time.time()-tstart))

# raDecs = []

# for ra in numpy.arange(0,360,15):
#     raDecs.append([ra, 20])

# raDecs.append([90, 42])
# raDecs.append([105, 42])
# raDecs.append([135, 42])
# raDecs.append([330, 45])
# raDecs.append([22.5*15, 60])
# raDecs.append([10*15, 42])
# raDecs.append([22*15, 20])
# raDecs.append([22*15, 25])
# raDecs.append([6*15, 25])

# start with ra=120, dec=20

# just keep stars in the GFA annulus centered at ra=120, dec=20

ARCSEC_PER_PIXEL = 0.216

raCen = 120
decCen = 20
raRadCen = numpy.radians(raCen)
decRadCen = numpy.radians(decCen)

tstart = time.time()

raRad = numpy.radians(gaia.ra.to_numpy())
decRad = numpy.radians(gaia.dec.to_numpy())

distFromCenter = numpy.arccos(
    numpy.sin(decRadCen)*numpy.sin(decRad) + numpy.cos(decRadCen)*numpy.cos(decRad)*numpy.cos(raRadCen-raRad)
)

distFromCenter = numpy.degrees(distFromCenter)

gaia["dDeg"] = distFromCenter


print("before cut", len(gaia))

gaia = gaia[gaia.dDeg < 1.55]
gaia = gaia[gaia.dDeg > 1.4]
# gaia = gaia[gaia.mag < 16]
print("print after cut", len(gaia))

#########################
# start with rot angle = 0 imgs
gfaList = [2,3,4,5,6]
imgBundle = {}
obsTimeJD = None
img5offset = 20
imgBaseDir = "/Users/csayres/fpscommis/apoData/59552/caroselRot/"
for gfaid in gfaList:
    imgNumber = 293 # rot=0 orientation.
    # imgNumber = 294 # rot=60 orientation
    if gfaid == 5:
        imgNumber += 20
    imgNumStr = ("%i"%imgNumber).zfill(4)
    filename = imgBaseDir + "gimg-gfa%in-%s.fits"%(gfaid, imgNumStr)
    f = fits.open(filename)
    print("exptime", f[1].header["EXPTIME"])
    if obsTimeJD is None:
        dObs = f[1].header["DATE-OBS"]
        astroTime = Time(dObs, format="iso", scale="tai")
        obsTimeJD = astroTime.jd
    # 2**16 - value in negative pixel
    imgData = numpy.array(f[1].data, dtype=numpy.float64)
    imgData = rescaleImgData(imgData)
    imgBundle[gfaid] = imgData


tstart = time.time()
output = radec2wokxy(
    ra=gaia.ra.to_numpy(),
    dec=gaia.dec.to_numpy(),
    coordEpoch=gaiaEpoch,
    waveName="GFA",
    raCen=raCen,
    decCen=decCen,
    obsAngle= 0.414,
    obsSite="APO",
    obsTime=obsTimeJD,
    pmra=gaia.pmra.to_numpy(),
    pmdec=gaia.pmdec.to_numpy(),
    parallax=gaia.parallax.to_numpy(),
    radVel=None,
    pressure=None,
    relativeHumidity=0.5,
    temperature=10
)
# print("conversion took %.2f"%(time.time()-tstart))

xWok, yWok, fieldWarn, hourAngle, positionAngle = output
print("hour angle", hourAngle)
xyWok = numpy.array([xWok, yWok]).T

dfList = []

for gfaID, imgData in imgBundle.items():

    # if gfaID != 6:
    #     continue

    xCCD, yCCD = wokToGFA(xWok, yWok, gfaID)



    # only keep pixels reasonably in the range
    keep = (xCCD > -1000) & (xCCD < 2048 + 1000) & (yCCD > -1000) & (yCCD < 2048 + 1000)
    xCCD = xCCD[keep]
    yCCD = yCCD[keep]
    _xWokGaia = xWok[keep]
    _yWokGaia = yWok[keep]
    _mag = gaia.mag.to_numpy()[keep]
    _flux = numpy.exp(_mag/-2.5) * 10000


    # if gfaID == 6:
    #     yCCD = yCCD - 975
    #     xCCD = xCCD - 160

    # find magical pixShift for matching at rotAngle = 0
    # if gfaID == 6:
    #     xOff = 1413-1237
    #     yOff = 1189 - 1230
    #     xCCD = xCCD - xOff
    #     yCCD = yCCD - yOff
    # if gfaID == 5:
    #     xOff = 848 - 696
    #     yOff = 947 - 964
    #     xCCD = xCCD - xOff
    #     yCCD = yCCD - yOff
    # if gfaID == 4:
    #     xOff = 871 - 715
    #     yOff = 1565 - 1550
    #     xCCD = xCCD - xOff
    #     yCCD = yCCD - yOff
    # if gfaID == 3:
    #     xOff = 634 - 460
    #     yOff = 709 - 680
    #     xCCD = xCCD - xOff
    #     yCCD = yCCD - yOff
    # if gfaID == 2:
    #     xOff = 1064 - 863
    #     yOff = 0
    #     xCCD = xCCD - xOff
    #     yCCD = yCCD - yOff





    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_title("GFA %i"%gfaID)
    ax.set_facecolor("black")
    centroids = plotSEP(imgData, ax)
    ax.imshow(equalize_hist(imgData), origin="lower")
    ax.scatter(xCCD, yCCD, s=_flux, facecolor="none", edgecolor="white")
    # for ii, (x, y) in enumerate(zip(xCCD, yCCD)):
    #     _magStr = "%.2f"%_mag[ii]
    #     plt.text(x,y,_magStr, color="cyan")


    # perform matching, for each detected centroid find the nearest
    # neighbor in gaia
    xCent = centroids["x"]
    yCent = centroids["y"]
    xyCent = numpy.array([xCent, yCent]).T
    # xyCCD are hacked to be close for these images, match them
    # hacked then set them back
    xyGaia = numpy.array([xCCD, yCCD]).T

    argFound, dist = arg_nearest_neighbor(xyCent, xyGaia)
    # throw out everything > 50 pixels away
    xyGaiaFound = xyGaia[argFound,:]
    _xWokGaia = _xWokGaia[argFound]
    _yWokGaia = _yWokGaia[argFound]
    keep = dist < 30
    xyGaiaFound = xyGaiaFound[keep, :]
    xyCent = xyCent[keep, :]
    _xWokGaia = _xWokGaia[keep]
    _yWokGaia = _yWokGaia[keep]

    # shift pixels back and convert to wok coords
    xCent = xyCent[:,0]
    yCent = xyCent[:,1]

    xWokCent, yWokCent = GFAToWok(xCent, yCent, gfaID)

    df = pandas.DataFrame({
        "xWokCent": xWokCent,
        "yWokCent": yWokCent,
        "xWokGaia": _xWokGaia,
        "yWokGaia": _yWokGaia,
        "gfaID": [gfaID] * len(xWokCent)
    })

    dfList.append(df)
    # import pdb; pdb.set_trace()
    # plt.figure()
    # plt.hist(numpy.linalg.norm(xyCent-xyGaiaFound, axis=1))
    # plt.title("GFA%i"%gfaID)
    # plt.show()

    # now that associations

    # plt.savefig("gfa%i.png"%gfaID)

    # # write csv's for rick
    # xCentroidPix = centroids["x"]
    # yCentroidPix = centroids["y"]
    # fluxCentroid = centroids["flux"]
    # centDF = pandas.DataFrame({
    #     "xPixCent": xCentroidPix,
    #     "yPixCent": yCentroidPix,
    #     "fluxCent": fluxCentroid
    # })
    # centDF.to_csv("GFA%i_centroids.csv"%gfaID)

    # gaiaDF = pandas.DataFrame({
    #     "xPixGaia": xCCD,
    #     "yPixGaia": yCCD,
    #     "gMagGaia": _mag

    # })

    # gaiaDF.to_csv("GFA%i_gaia.csv"%gfaID)

    # break

# plt.show()

df = pandas.concat(dfList)
df["xerr"] = df.xWokCent - df.xWokGaia
df["yerr"] = df.yWokCent - df.yWokGaia
plt.figure()
plt.quiver(df.xWokGaia, df.yWokGaia, df.xerr, df.yerr, angles="xy")

plt.figure()
plt.hist(numpy.sqrt(df.xerr**2+df.yerr**2))
plt.xlabel("mm err")

### transform the centroid locations to gaia
# xyGaia = df[["xWokGaia", "yWokGaia"]].to_numpy()
# xyCent = df[["xWokCent", "yWokCent"]].to_numpy()
# simTrans = SimilarityTransform()
# # simTrans = EuclideanTransform()
# simTrans.estimate(xyCent, xyGaia)

# # Apply the model to the data
# xySimTransFit = simTrans(xyCent)
# print("translation", simTrans.translation)
# print("rotation", numpy.degrees(simTrans.rotation))
# print("scale", simTrans.scale)

# err = xySimTransFit - xyGaia

# plt.figure()
# plt.quiver(xyGaia[:,0], xyGaia[:,1], err[:,0], err[:,1], angles="xy")


# plt.figure()
# plt.hist(numpy.sqrt(numpy.sum(err**2, axis=1)))
# plt.xlabel("mm err")


plt.show()

# import pdb; pdb.set_trace()


# import pdb; pdb.set_trace()

