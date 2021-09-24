import coordio.defaults
from coordio.defaults import designRef, POSITIONER_HEIGHT, IHAT, JHAT, KHAT
from coordio.defaults import wokCoords
from coordio.conv import tangentToWok
import fitsio
import glob
import matplotlib.pyplot as plt
import numpy
from skimage.exposure import equalize_hist
import sep
from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform
import pandas as pd
import os

# estimate img scale # from dot image
# c1 =  numpy.array([2595.6355, 1796.1706])
# c2 =  numpy.array([2932.208, 1797.8451])
# c1c2dist = 10 # mm
# imgScale = c1c2dist / numpy.linalg.norm(c1-c2)
# print("imgScale", imgScale)

class CSVBuilder(object):
    wokName = "default"

    def __init__(self):
        self.genFiducials()
        self.genPositionerTable()
        self.genWokCoords()

        # get rid of index columns if they exist
        if "index" in self.positionerTable.columns:
            self.positionerTable.drop(columns=["index"], inplace=True)
        if "index" in self.wokCoords.columns:
            self.wokCoords.drop(columns=["index"], inplace=True)
        if "index" in self.fiducialCoords.columns:
            self.fiducialCoords.drop(columns=["index"], inplace=True)

    def write(self, outdir):
        self.fiducialCoords.to_csv(os.path.join(outdir, "fiducialCoords.csv"))
        self.wokCoords.to_csv(os.path.join(outdir, "wokCoords.csv"))
        self.positionerTable.to_csv(os.path.join(outdir, "positionerTable.csv"))


class UWMiniWok(CSVBuilder):
    wokName = "uwMiniWok"

    def genFiducials(self):
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
        self.fiducialCoords = df.reset_index()
        # df.to_csv("fiducialCoords.csv", index=False)

    def genPositionerTable(self):
        # build the positioner table
        _pid = [734, 428, 561, 594, 649, 497, 484, 704, 524, 645, 457, 705, 566]
        _row = [2, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -2]
        _col = [13, 12, 13, 14, 15, 13, 14, 15, 12, 13, 14, 15, 13]
        self._row = _row
        self._col = _col
        _holeID = []
        self.nItems = len(_pid)

        for ii in range(self.nItems):
            c = _col[ii]
            r = _row[ii]
            if r <= 0:
                holeID = "R%iC%i"%(r,c)
            else:
                holeID = "R+%iC%i"%(r,c)
            _holeID.append(holeID)
        # wokName = "uwMiniWok"
        d = {}
        d["positionerID"] = _pid
        d["robotailID"] = ["FTO%i"%x for x in range(self.nItems)]
        d["wokID"] = [self.wokName] * self.nItems
        d["holeID"] = _holeID
        d["apSpecID"] = list(range(self.nItems))
        d["bossSpecID"] = list(range(self.nItems))
        d["alphaArmLen"] = [7.4] * self.nItems
        d["metX"] = [coordio.defaults.MET_BETA_XY[0]] * self.nItems
        d["metY"] = [coordio.defaults.MET_BETA_XY[1]] * self.nItems
        d["apX"] = [coordio.defaults.AP_BETA_XY[0]] * self.nItems
        d["apY"] = [coordio.defaults.AP_BETA_XY[1]] * self.nItems
        d["bossX"] = [coordio.defaults.BOSS_BETA_XY[0]] * self.nItems
        d["bossY"] = [coordio.defaults.BOSS_BETA_XY[1]] * self.nItems
        d["alphaOffset"] = [0] * self.nItems
        d["betaOffset"] = [0] * self.nItems
        d["dx"] = [0] * self.nItems
        d["dy"] = [0] * self.nItems

        df = pd.DataFrame(d)
        self.positionerTable = df.reset_index()
        # df.to_csv("positionerTable.csv", index=False)

    def genWokCoords(self):
        # wokName = "uwMiniWok"
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

        for ii in range(self.nItems):
            r = self._row[ii]
            c = self._col[ii]
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
        d["wokID"] = [self.wokName] * nItems
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
        self.wokCoords = df.reset_index()
        # df.to_csv("wokCoords.csv", index=False)


class FlatWokNominal(CSVBuilder):
    wokName = "flatNominal"

    def genFiducials(self):
        fdf = designRef[(designRef.fType == "Fiducial") | (designRef.fType == "GFA-Fiducial") ]
        nItems = len(fdf)
        _fid = ["F%i"%x for x in range(nItems)]
        _zWok = [POSITIONER_HEIGHT]*nItems
        d = {}
        d["id"] = _fid
        d["xWok"] = fdf["xWok"]
        d["yWok"] = fdf["yWok"]
        d["zWok"] = _zWok
        d["holeID"] = fdf["holeName"]
        d["col"] = fdf["col"]
        d["row"] = fdf["row"]


        df = pd.DataFrame(d)
        self.fiducialCoords = df.reset_index()
        # df.to_csv("fiducialCoords.csv", index=False)

    def genPositionerTable(self):

        fdf = designRef[(designRef.fType == "BA") | (designRef.fType == "BOSS")]

        nItems = len(fdf)
        # fake positioner ids, robotail ids
        _pid = numpy.arange(nItems)


        d = {}
        d["positionerID"] = _pid
        d["robotailID"] = ["FTO%i"%x for x in range(nItems)]
        d["wokID"] = [self.wokName] * nItems
        d["holeID"] = fdf["holeName"]
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
        self.positionerTable = df.reset_index()
        # df.to_csv("positionerTable.csv", index=False)

    def genWokCoords(self):

        fdf = designRef[
        (designRef.fType == "BA") | (designRef.fType == "BOSS") |
        (designRef.fType == "Fiducial") | (designRef.fType == "GFA-Fiducial") |
        (designRef.fType == "Aux")
        ]
        nItems = len(fdf)

        fdf["wokID"] = [self.wokName] * nItems
        fdf = fdf.rename(columns={
            "holeName": "holeID", "row": "hexRow", "col": "hexCol"
        })

        holeType = []
        ftypes = fdf["fType"].to_list()
        for ftype in ftypes:
            if "Fiducial" in ftype:
                holeType.append("Fiducial")
            elif ftype == "BA":
                holeType.append("ApogeeBoss")
            else:
                holeType.append("Boss")

        # import pdb; pdb.set_trace()
        fdf["holeType"] = holeType
        # rename the Boss and Apogees differently
        fdf["zWok"] = [0] * nItems
        fdf["ix"] = IHAT[0]
        fdf["iy"] = IHAT[1]
        fdf["iz"] = IHAT[2]
        fdf["jx"] = JHAT[0]
        fdf["jy"] = JHAT[1]
        fdf["jz"] = JHAT[2]
        fdf["kx"] = KHAT[0]
        fdf["ky"] = KHAT[1]
        fdf["kz"] = KHAT[2]

        # put things in right order
        fdf = fdf[
            ["wokID","holeID","holeType","hexRow","hexCol","xWok",
            "yWok","zWok","ix","iy","iz","jx","jy","jz","kx","ky","kz"]
        ]

        self.wokCoords = fdf.reset_index()
        # fdf.to_csv("wokCoords.csv", index=False)


class CurveWokNominal(FlatWokNominal):

    # note the curved wok coords (coordio.wokCoords)
    # has outdated assignments, use assignments
    # from designRef, but the coordinates from
    # wokCoords
    def __init__(self, wokType):
        self.df = wokCoords[wokCoords.wokType==wokType]
        self.wokName = wokType
        wcHoleNames = numpy.array(["F%i"%x for x in range(1,19)])
        designRefHoleNames = numpy.roll(wcHoleNames[::-1], 10)
        self.dr2wc = {}
        self.wc2dr = {}
        for wcName, drName in zip(wcHoleNames, designRefHoleNames):
            self.dr2wc[drName] = wcName
            self.wc2dr[wcName] = drName
        super().__init__()

    def genFiducials(self):
        fc = FlatWokNominal().fiducialCoords  # built with design ref
        # use new map for outer ring of fiducials
        # named F1-F18 starting from boss end and
        # incrementing clockwise, to match the design reference table
        #  The wokCoords
        # table started at apogee and increased
        # counter cloclwise

        # use the design reference to get assignments
        # because the 3D coordio.wokCoords table was generated
        # with an out of date assignment
        nItems = len(fc)
        _xWok = []
        _yWok = []
        _zWok = []
        _holeID = []
        for hid in fc.holeID.to_numpy():
            if  hid.startswith("F"):  # outer ring is backwards
                hidwc = self.dr2wc[hid]
            else:
                hidwc = hid
            row = self.df[self.df.holeID==hidwc]
            b = [row.x, row.y, row.z]
            i = [row.ix, row.iy, row.iz]
            j = [row.jx, row.jy, row.jz]
            k = [row.kx, row.ky, row.kz]
            xWok, yWok, zWok = tangentToWok(
                0, 0, 0, b, i, j, k
            )
            _xWok.append(xWok)
            _yWok.append(yWok)
            _zWok.append(zWok)
            _holeID.append(hid)

        d = {}
        d["id"] = ["F%i"%x for x in range(nItems)]
        d["xWok"] = _xWok
        d["yWok"] = _yWok
        d["zWok"] = _zWok
        d["holeID"] = _holeID
        d["col"] = fc.col
        d["row"] = fc.row

        df = pd.DataFrame(d)
        self.fiducialCoords = df.reset_index()

    def genPositionerTable(self):
        # this one doesn't change!
        super().genPositionerTable()

    def genWokCoords(self):

        df = self.df.copy()

        df = df.rename(columns={
            "x": "xWok", "y": "yWok", "z": "zWok", "row": "hexRow", "col": "hexCol"
        })

        # replace the names for holeID for outer fiducial ring
        _holeID = []

        for holeID in df.holeID:
            if holeID.startswith("F"):
                holeID = self.wc2dr[holeID]
            _holeID.append(holeID)

        df["holeID"] = _holeID
        df["wokID"] = "curvNom" + self.wokName

        df = df[
            ["wokID","holeID","holeType","hexRow","hexCol","xWok",
            "yWok","zWok","ix","iy","iz","jx","jy","jz","kx","ky","kz"]
        ]


        self.wokCoords = df.reset_index()


class CurveWokNominalAPO(CurveWokNominal):

    def __init__(self):
        super().__init__("APO")


class CurveWokNominalLCO(CurveWokNominal):

    def __init__(self):
        super().__init__("LCO")

#####################  hexagon cmm meas ###########################
# class CMMFiducialMeasTable(object):


def getCMMFidMeas(filename, flat=False):
    """If flat, then model wok as a flat surface
    do this when using a FlatWokNominal
    """
    df = pd.read_excel(filename)
    df = df.rename(columns={
        "WokPosition": "holeID", "FIF_SN": "id",
        "Xcore": "xWok", "Ycore": "yWok", "Ztactile": "zWok"
    })

    _col = []
    _row = []
    for holeID in df.holeID:
        if holeID.startswith("F"):
            _col.append(None)
            _row.append(None)
            continue

        x = holeID.strip("R")
        x = x.strip("+")
        r, c = x.split("C")
        _col.append(int(c))
        _row.append(int(r))

    df["row"] = _row
    df["col"] = _col

    df = df[["id", "xWok", "yWok", "zWok", "holeID", "col", "row"]]
    if flat:
        df["zWok"] = POSITIONER_HEIGHT
    df.reset_index(inplace=True)
    df.drop(columns=["index"], inplace=True)
    return df


def getCMMFidNom(filename, flat=False):
    """If flat, then model wok as a flat surface
    do this when using a FlatWokNominal
    """
    df = pd.read_excel(filename)
    df = df.rename(columns={
        "WokPosition": "holeID", "FIF_SN": "id",
        "Xnominal": "xWok", "Ynominal": "yWok", "Znominal": "zWok"
    })

    _col = []
    _row = []
    for holeID in df.holeID:
        if holeID.startswith("F"):
            _col.append(None)
            _row.append(None)
            continue

        x = holeID.strip("R")
        x = x.strip("+")
        r, c = x.split("C")
        _col.append(int(c))
        _row.append(int(r))

    df["row"] = _row
    df["col"] = _col

    df = df[["id", "xWok", "yWok", "zWok", "holeID", "col", "row"]]
    if flat:
        df["zWok"] = POSITIONER_HEIGHT
    df.reset_index(inplace=True)
    df.drop(columns=["index"], inplace=True)
    return df

def getPositionerTable(wokName):

    nItems = len(designRef)
    # fake positioner ids, robotail ids
    _pid = numpy.arange(nItems)


    d = {}
    d["positionerID"] = _pid
    d["robotailID"] = ["FTO%i"%x for x in range(nItems)]
    d["wokID"] = [wokName] * nItems
    d["holeID"] = designRef["holeName"]
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
    return df.reset_index()

def assign(csvObj, assignmentFile):
    """Assign robots to holes, modifies csvObj
    """
    df = pd.read_csv(assignmentFile)

    positionerTable = getPositionerTable(csvObj.wokName)

    # import pdb; pdb.set_trace()
    usedHoles = []
    for ii, row in df.iterrows():
        devID = row.Device
        if devID.startswith("P"):
            pid = int(devID.strip("P"))
            holeID = row.Row + row.Column
            usedHoles.append(holeID)
            # before = csvObj.positionerTable[csvObj.positionerTable.holeID==holeID].positionerID
            # print("before", before)

            positionerTable.loc[(positionerTable.holeID == holeID),'positionerID']= pid

            # import pdb; pdb.set_trace()

            # after = positionerTable[positionerTable.holeID==holeID].positionerID
            # print("after", after)


    # remove default robots not present in assignmentFile
    allHoles = positionerTable.holeID
    removeHoles = list(set(allHoles) - set(usedHoles))

    for removeMe in removeHoles:
        idx = positionerTable.loc[positionerTable.holeID == removeMe].index
        positionerTable.drop(index=idx, inplace=True)

    positionerTable.reset_index(inplace=True)
    positionerTable.drop(columns=["index", "level_0"], inplace=True)
    csvObj.positionerTable = positionerTable

    # import pdb; pdb.set_trace()

# class AssignmentAPO(Assignment):

#     def __init__(self):
#         super().__init__("SloanFPS_HexArray_2021July23.csv")


if __name__ == "__main__":
    # csvObj = FlatWokNominal()
    # apoCSV = assign(csvObj, "SloanFPS_HexArray_2021July23.csv")
    # apoCSV.write("/users/csayres/wokCalib/sloanFlatNom")

    # csvObj = CurveWokNominalAPO()
    # cmmFid = CMMFiducialMeasAPO()
    # apoCSV = assign(csvObj, "SloanFPS_HexArray_2021July23.csv", cmmFid)
    # apoCSV.write("/users/csayres/wokCalib/sloanCurveCMM")


    # duPont sparse wok
    # csvObj = FlatWokNominal()
    # print(len(csvObj.positionerTable))
    # cmmMeasTable = getCMMFidMeas("FPS_DuPont_CMM_20210504.xlsx", flat=True)
    # assign(csvObj, "LCO-sparse-2021Sep20.csv")
    # csvObj.fiducialCoords = cmmMeasTable
    # # print(len(csvObj.positionerTable))
    # csvObj.write("/users/csayres/wokCalib/duPontSparse")


    # osu mini wok
    csvObj = FlatWokNominal()
    assign(csvObj, "miniwok_OSU_2021Sep20.csv")
    # no fiducials in the osu miniwok
    csvObj.fiducialCoords = csvObj.fiducialCoords[0:0]
    # print(len(csvObj.positionerTable))
    # import pdb; pdb.set_trace()
    csvObj.write("/Users/csayres/wokCalib/osuMiniWok")




# import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()


