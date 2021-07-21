import coordio.defaults
from coordio.defaults import designRef, POSITIONER_HEIGHT
import fitsio
import glob
import matplotlib.pyplot as plt
import numpy
from skimage.exposure import equalize_hist
import sep
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform
import pandas as pd

# estimate img scale # from dot image
# c1 =  numpy.array([2595.6355, 1796.1706])
# c2 =  numpy.array([2932.208, 1797.8451])
# c1c2dist = 10 # mm
# imgScale = c1c2dist / numpy.linalg.norm(c1-c2)
# print("imgScale", imgScale)

imgScale = 0.026926930620282834 ## found from best fit similarity transform


# fiducialIDs = [0, 1, 7, 8, 16, 17] # just ordered by discovery
fiducialIDs = [8, 17, 16, 7, 0, 1]  # ordered by ccw position
column = [16,14,12,12,12,14]
row = [0,2,2,0,-2,-2]
fiducialXCCD = []
fiducialYCCD = []

# make a anti clockwise hex for the hexagon points
rHex = 22.4 * 2
thetas = numpy.radians(numpy.array([0, 60, 2*60, 3*60, 4*60, 5*60]))
modfxmm = rHex * numpy.cos(thetas)
modfymm = rHex * numpy.sin(thetas)

onFold = glob.glob("data/*fold*on.fits")
onTarg = glob.glob("data/*targ*on.fits")

data = fitsio.read(onTarg[0])
shape = data.shape

allStack = numpy.zeros(shape)

for onImg in onFold[:50]:
    allStack += fitsio.read(onImg)

allStack = allStack / len(onTarg)
plt.imshow(equalize_hist(allStack), origin="lower")

objects = sep.extract(allStack, 4.5)
objects = objects[objects["peak"] > 1000]

# for ii, obj in enumerate(objects):
for ii in fiducialIDs:
    fiducialXCCD.append(objects[ii]["x"])
    fiducialYCCD.append(objects[ii]["y"])
    plt.text(objects[ii]["x"], objects[ii]["y"], "%i"%ii)
    # print("%i: peak=%.2f, x=%.f, erry=%.8f"%(obj["peak"], obj["errx2"], obj["erry2"]))
print("len objs", len(objects))
plt.show()

fiducialXCCD = numpy.array(fiducialXCCD)
fiducialYCCD = numpy.array(fiducialYCCD)
fiducialXYCCD = numpy.array([fiducialXCCD, fiducialYCCD]).T

# measfxmm = fiducialXCCD * imgScale
# measfymm = fiducialYCCD * imgScale
# measfxymm = numpy.array([measfxmm,measfymm]).T

modfxymm = numpy.array([modfxmm, modfymm]).T

tform = SimilarityTransform()
tform.estimate(fiducialXYCCD, modfxymm)

xymm = tform(fiducialXYCCD)
err = xymm - modfxymm

print("err", err*1000)

plt.figure()
plt.quiver(xymm[:,0], xymm[:,1], err[:,0], err[:,1], angles="xy")
plt.show()

mtform =  numpy.array(
    [[ 2.69265448e-02,  1.44153787e-04, -7.56960605e+01],
     [-1.44153787e-04,  2.69265448e-02, -4.86856127e+01],
     [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
)

rotMat = numpy.array([
    [ 2.69265448e-02,  1.44153787e-04],
     [-1.44153787e-04,  2.69265448e-02]
])

txy = numpy.array([-7.56960605e+01, -4.86856127e+01])

"""
from skimage docs
X = a0 * x - b0 * y + a1 =
  = s * x * cos(rotation) - s * y * sin(rotation) + a1

Y = b0 * x + a0 * y + b1 =
  = s * x * sin(rotation) + s * y * cos(rotation) + b1

"""

_xymm = (rotMat @ fiducialXYCCD.T).T + txy
print(_xymm)

print(_xymm/xymm)


# import pdb; pdb.set_trace()
# estimate physical xy mm of fiducials

# build the fiducial table
_fid = []
_xWok = []
_yWok = []
_zWok = []
_holeID = []
_col = []
_row = []
for ii in range(len(fiducialIDs)):
    _fid.append("F%i"%fiducialIDs[ii])
    _xWok.append(xymm[ii][0])
    _yWok.append(xymm[ii][1])
    _zWok.append(POSITIONER_HEIGHT)
    c = column[ii]
    r = row[ii]
    if r <= 0:
        holeID = "R%iC%i"%(r,c)
    else:
        holeID = "R+%iC%i"%(r,c)
    _col.append(c)
    _row.append(r)
    _holeID.append(holeID)

d = {}
d["id"] = _fid
d["xWok"] = _xWok
d["yWok"] = _yWok
d["zWok"] = _zWok
d["holeID"] = _holeID
d["col"] = _col
d["row"] = _row


df = pd.DataFrame(d)

df.to_csv("fiducialCoords.csv", index=False)

# build the positioner table
_pid = [734, 428, 561, 594, 649, 497, 484, 704, 524, 645, 457, 705, 566]
_row = [2, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -2]
_col = [13, 12, 13, 14, 15, 13, 14, 15, 12, 13, 14, 15, 13]
_holeID = []
nItems = len(_pid)

for ii in range(nItems):
    c = _col[ii]
    r = _row[ii]
    if r <= 0:
        holeID = "R%iC%i"%(r,c)
    else:
        holeID = "R+%iC%i"%(r,c)
    _holeID.append(holeID)
wokName = "uwMiniWok"
d = {}
d["positionerID"] = _pid
d["robotailID"] = ["FTO%i"%x for x in range(nItems)]
d["wokID"] = [wokName] * nItems
d["holeID"] = _holeID
d["apSpecID"] = list(range(nItems))
d["bossSpecID"] = list(range(nItems))
d["alphaArmLen"] = [7.4] * nItems
d["metX"] = [coordio.defaults.MET_BETA_XY[0]] * nItems
d["metY"] = [coordio.defaults.MET_BETA_XY[1]] * nItems
d["apX"] = [coordio.defaults.AP_BETA_XY[0]] * nItems
d["apY"] = [coordio.defaults.AP_BETA_XY[1]] * nItems
d["bossX"] = [coordio.defaults.BOSS_BETA_XY[0]] * nItems
d["bossY"] = [coordio.defaults.BOSS_BETA_XY[1]] * nItems
d["alphaOffset"] = [0] * nItems
d["betaOffset"] = [0] * nItems
d["dx"] = [0] * nItems
d["dy"] = [0] * nItems

df = pd.DataFrame(d)

df.to_csv("positionerTable.csv", index=False)

# build wok coord table
fiducialIDs = [8, 17, 16, 7, 0, 1]  # ordered by ccw position
fidcolumn = [16,14,12,12,12,14]
fidrow = [0,2,2,0,-2,-2]

_holeID = []
_holeType = []
_x = []
_y = []
_hexRow = []
_hexCol = []

for ii in range(len(fidrow)):
    r = fidrow[ii]
    c = fidcolumn[ii]
    if r <= 0:
        holeID = "R%iC%i"%(r,c)
    else:
        holeID = "R+%iC%i"%(r,c)
    _holeID.append(holeID)
    _holeType.append("Fiducial")
    _hexRow.append(r)
    _hexCol.append(c)
    tabRow = designRef[designRef.holeName==holeID]
    _x.append(float(tabRow.xWok))
    _y.append(float(tabRow.yWok))

for ii in range(nItems):
    r = _row[ii]
    c = _col[ii]
    if r <= 0:
        holeID = "R%iC%i"%(r,c)
    else:
        holeID = "R+%iC%i"%(r,c)
    _holeID.append(holeID)
    _holeType.append("ApogeeBoss")
    _hexRow.append(r)
    _hexCol.append(c)
    tabRow = designRef[designRef.holeName==holeID]
    _x.append(float(tabRow.xWok))
    _y.append(float(tabRow.yWok))

nItems = len(_holeID)

iHat = [0,-1,0]
jHat = [1,0,0]
kHat = [0,0,1]

d = {}
d["wokID"] = [wokName] * nItems
d["holeID"] = _holeID
d["holeType"] = _holeType
d["hexRow"] = _hexRow
d["hexCol"] = _hexCol
d["xWok"] = _x
d["yWok"] = _y
d["zWok"] = [0] * nItems
d["ix"] = [iHat[0]] * nItems
d["iy"] = [iHat[1]] * nItems
d["iz"] = [iHat[2]] * nItems
d["jx"] = [jHat[0]] * nItems
d["jy"] = [jHat[1]] * nItems
d["jz"] = [jHat[2]] * nItems
d["kx"] = [kHat[0]] * nItems
d["ky"] = [kHat[1]] * nItems
d["kz"] = [kHat[2]] * nItems

df = pd.DataFrame(d)
df.to_csv("wokCoords.csv", index=False)

# import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()


