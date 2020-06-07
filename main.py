import nst.data_loader
import nst.nst
from nst.plotter import Plotter
data_loader = nst.data_loader.DataLoader("sample")
data_loader.PlotData()
NST = nst.nst.NST(data_loader)
Plotter().ShowOneImage(NST.Run())