import os
import win32com.client as win32
from xlsxwriter.utility import xl_cell_to_rowcol
import time

class simulation_driver:
    def __init__(self, hy_filename,  cols_mapping, x_cols, y_cols, resultindict=False, incaseofnooutput = None):
        self.hy_filename = hy_filename
        self.cols_mapping = cols_mapping
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.HyApp = None
        self.HyCase = None
        self.HyOperations = None
        self.HySolver = None
        self.resultindict = resultindict
        self.incaseofnooutput = incaseofnooutput
        self.debug = False

    def load_model(self):
        hy_visible = True
        hyFilePath = os.path.abspath(self.hy_filename)

        # 02 Initialize  Aspen Hysys application
        if self.debug:
            print(' # Connecting to the Aspen Hysys App in debug mode ... ')
        HyApp = win32.Dispatch('HYSYS.Application')
        
        HyCase = HyApp.SimulationCases.Open(hyFilePath)
        # HyCase = HyApp.ActiveDocument
        HyCase.Visible = hy_visible
        
        # 05 Aspen Hysys File Name
        HySysFile = HyCase.Title.Value
        
        # 06 Aspen Hysys Fluid Package Name
        package_name = HyCase.Flowsheet.FluidPackage.PropertyPackageName
        
        # 07 Main Aspen Hysys Document Objects
        HySolver = HyCase.Solver  # Access to Hysys Solver
        # HyFlowsheet = HyCase.Flowsheet                   # Access to main Flowsheet
        HyOperations = HyCase.Flowsheet.Operations  # Access to the Unit Operations
        #HyMaterialStream = HyCase.Flowsheet.MaterialStreams  # Access to the material streams
        #HyEnergyStream = HyCase.Flowsheet.EnergyStreams  # Access to the energy streams

        self.HyApp = HyApp
        self.HyCase = HyCase
        self.HyOperations = HyOperations
        self.HySolver = HySolver

        if self.debug:
            print(' ')
            print('HySys File Loaded: -----  ', HySysFile)
            print('HySys Fluid Package: ---  ', package_name)
            print(' ')

    def sim_write(self, data=dict()):
        self.HySolver.CanSolve = False
        retd = False
        spreadsheet = list(self.cols_mapping.keys())[0]
        # write data to Simulation
        for k, v in self.cols_mapping[spreadsheet].items():
            (rows, cols) = xl_cell_to_rowcol(v)
            hycellin = self.HyOperations.Item(spreadsheet).Cell(cols, rows)
            
            if k in data.keys():
                hycellin.CellValue = data[k]
            else:
                raise ValueError(f"Key '{k}' not found in input data.")
        retd = True
        
        return retd

    def sim_run(self):
        self.HySolver.CanSolve = True
        xx = 0
        while self.HySolver.IsSolving:
                xx += 1
        
        solved = True
        return solved

    def sim_read(self):
        # read data from Simulation
        spreadsheet = list(self.cols_mapping.keys())[1]
        uomsname = list(self.cols_mapping.keys())[2]
        uomsdict = self.cols_mapping[uomsname]

        output = []
        for k, v in self.cols_mapping[spreadsheet].items():
            (rows, cols) = xl_cell_to_rowcol(v)
            hycellout = self.HyOperations.Item(spreadsheet).Cell(cols, rows)
            uom = uomsdict[k]
            if uom:
                output.append(hycellout.Variable.GetValue(uom))
            else:
                output.append(hycellout.CellValue)
        
        return output

    def close(self):
        self.HyCase.Close()
        
    def predict(self, data=dict()):
        output = []
        
        retd = self.sim_write(data)
        
        if retd:
            solved = self.sim_run()
        
        if retd and solved:
            try:
                output = self.sim_read()
            except:
                Warning('Simulation read error, trying to reload model...')
                self.close()
                time.sleep(5)
                self.load_model()
                time.sleep(10)
                output = self.incaseofnooutput              
        
        if output and self.resultindict:
            pred_dict = {}
            for k, v in zip(self.y_cols, output):
                pred_dict[k] = v
            
            output = pred_dict

        return output