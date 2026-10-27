# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:26:38 2021

@author: lenovo
"""

import numpy as np

from myClass.myBICFile import boudaryAndInitial2D
from myClass.myTrainingPointFile import TrainingPoints2D
from myClass.myAuxiPointsFile import AuxiMultiGL2D
from myClass.myPredictPointsFile import predictionPoin2D
from myClass.myNormalizingPointFile import NormalizingPoints2D
from myClass.myCostPointFile import CostPoints2D

class myDataSet2D():
    def __init__(self,parameters,targetPDF):
        #######################################################################
        self.IBCdata=boudaryAndInitial2D(
            parameters
            )
        #######################################################################
        self.Trainingdata=TrainingPoints2D(
            parameters
            )
        #######################################################################        
        self.Predictdata = predictionPoin2D(parameters)
        #######################################################################
        self.Auxidata = AuxiMultiGL2D(parameters)
        #######################################################################
        self.Normalizingdata = NormalizingPoints2D(self.Auxidata)        
        #######################################################################
        self.Costdata = CostPoints2D(self.Auxidata,targetPDF)        
