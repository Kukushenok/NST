import nst.data_loader
import nst.nst
from nst.plotter import Plotter

TASK_NAME = "sample_2"

data_loader = nst.data_loader.DataLoader(TASK_NAME)
data_loader.PlotData()
NST = nst.nst.NST(data_loader)
Plotter().SaveImage(NST.Run(),TASK_NAME+"_output.png")