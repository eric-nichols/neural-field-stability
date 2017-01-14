#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use( 'WXAgg' )
matplotlib.interactive( True )
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import wx
import numpy as np



# import sys, wxversion
# print "Python version:", (sys.version)
# print "\nWX version:", wxversion.getInstalled()
# print "\nMatplotlib version:", matplotlib.__version__ 
    
class findRoots(wx.Frame):
    

    def __init__(self, *args, **kw):

        super(findRoots, self).__init__(*args, **kw) 

        # isStable is the stability state of the system
        # -2: unstable and prior calculation unstable
        # -1: unstable and prior calculation   stable
        #  0:  not yet initialized
        #  1:   stable and prior calculation unstable
        #  2:   stable and prior calculation   stable
        self.isStable     = 1 # assume we are starting stable
        

        self.pi2          = 2 * np.pi
        numVals           = 50
        self.Krange       = np.array(range(0, numVals)) / 25.0 - 1
        self.xAxis        = np.linspace(-2.0,2.0,50) 
        self.data         = [i-2 for i in range(0, numVals)]  # [(i-2, i-2) for i in range(0, numVals)]  
        self.windowWidth  = 350
        self.windowHeight = 800 #350
        self.InitUI()
        

    def InitUI(self):   

        self.SetBackgroundColour("white")
        self.c     = 1.84
        self.theta = 3.0 #100.0 #3.0
        self.s0    = 1 #600
        self.ae = 5.7
        self.ai = 1.0
        self.I0 = 0.0
        self.s  = 1.9
        self.Vdsk = 20000
        self.Vmin = 0.0 
        self.Vmax = 7.0
        self.plotNum = 1 # 1:curve relative to 0, 2: linear V and Sigmoid slope
        self.SPrime = None
        
        # *********************************************************************
        # our main window
        mainBox           = wx.BoxSizer(wx.HORIZONTAL)        

        # our 2 panels (sections)
        self.panelRoot = wx.Panel(self, 7, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER) # Get root panel
        self.panelFK   = wx.Panel(self, 7, wx.DefaultPosition, wx.DefaultSize, wx.RAISED_BORDER) # F(K) panel    
        
        # main box inside the self.panelFK
        self.boxfK   = wx.BoxSizer(wx.VERTICAL)   # vertical box to hold all sigmoid elements  
        
        # values inside self.boxfK
        self.boxFKeq      = wx.BoxSizer(wx.VERTICAL)   # f(K) vertical box to hold all f(K) elements
        self.SprimeBox    = wx.StaticBoxSizer(wx.StaticBox(self.panelFK, label="Sigmoidal firing rate derivative"), wx.HORIZONTAL)
        self.V0Box        = wx.StaticBoxSizer(wx.StaticBox(self.panelFK, label=u'Root V\u2080'), wx.HORIZONTAL)
        
        # s, ae and ai values inside self.boxfK
        self.fKvarsBox    = wx.StaticBoxSizer(wx.StaticBox(self.panelFK, label="Range of spacial inhibition"), wx.HORIZONTAL)
        sbox              = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the S      variable

        
        # Sigmoidal firing rate box
        self.SigmoidBox    = wx.StaticBoxSizer(wx.StaticBox(self.panelRoot, label="Sigmoidal firing rate"), wx.VERTICAL)
        self.SigmoidEqBox  = wx.BoxSizer(wx.HORIZONTAL) # Box to view the Sigmoid equation and output
        self.SigmoidVarBox = wx.BoxSizer(wx.HORIZONTAL)
        cbox               = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the c     variable
        thetabox           = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the theta variable        
        s0box              = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the S_0   variable 
        
        

        
        # ae and ai values inside stationary state
        self.stationaryBox    = wx.StaticBoxSizer(wx.StaticBox(self.panelRoot, label="Stationary state"), wx.VERTICAL)
        self.stationaryEqBox  = wx.BoxSizer(wx.HORIZONTAL) 
        self.stationaryVarBox = wx.BoxSizer(wx.HORIZONTAL)
        self.aebox            = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the ae     variable
        self.aibox            = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the ai     variable
        self.I0box            = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the I0    variable


        # V values inside self.boxfK
        self.vBox     = wx.StaticBoxSizer(wx.StaticBox(self.panelRoot, label="Voltage minimum, maximum and discrete units"), wx.HORIZONTAL)
        self.vMin     = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the ae     variable
        self.vMax     = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the ai     variable
        self.vDiscr   = wx.BoxSizer(wx.HORIZONTAL) # Box to modify the ai     variable


        # boxes inside the self.panelRoot
        self.boxGetRoot   = wx.BoxSizer(wx.VERTICAL)   # f(K) vertical box to hold all f(K) elements
        self.runButtonbox = wx.BoxSizer(wx.HORIZONTAL) # Box to hold the root button
        # *********************************************************************





        # *********************************************************************
        # The values for self.panelFK
        # *********************************************************************
        
        # Current file path location
        import os
        __location__  = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # The f(K) equation images
#         self.fkEqRED = wx.StaticBitmap(self.panelFK, wx.ID_ANY, wx.Bitmap(os.path.join(__location__, 'pics/fKred.tif'))) 
#         self.fkEqRED.Hide()
#         self.fkEqBLUE = wx.StaticBitmap(self.panelFK, wx.ID_ANY, wx.Bitmap(os.path.join(__location__, 'pics/fKblue.tif'))) 
#         
        
        # initialize
        self.fkEqRED = wx.Image(os.path.join(__location__, 'pics/fKred.tif'), wx.BITMAP_TYPE_TIF).Rescale(252, 63.3)  # 3
        self.fkEqRED = wx.StaticBitmap(self.panelFK, wx.ID_ANY, wx.Bitmap(self.fkEqRED)) # 1
        self.fkEqRED.Hide()
        
        self.fkEqBLUE = wx.Image(os.path.join(__location__, 'pics/fKblue.tif'), wx.BITMAP_TYPE_TIF).Rescale(252, 63.3)  # 3
        self.fkEqBLUE = wx.StaticBitmap(self.panelFK, wx.ID_ANY, wx.Bitmap(self.fkEqBLUE)) # 1
        
        #self.boxFKeq.AddSpacer(10) 
        #self.boxFKeq.Add((1,20))
        
        
        # *********************************************************************
        # Sigmoid prime equation box
        self.SprimeEq     = wx.StaticBitmap(self.panelFK, wx.ID_ANY, wx.Bitmap(os.path.join(__location__, 'pics/Sprime.png'))) 
        self.SPrimeEqText = wx.StaticText(  self.panelFK, label='                  ') #, pos=(20, 90))   
        self.SprimeBox.Add(self.SprimeEq, flag=wx.EXPAND) #, border=5)  # show equation as static bitmap
        self.SprimeBox.Add(self.SPrimeEqText, flag=wx.CENTER)
   
        # *********************************************************************
        # Root V0 box
        self.V0text = wx.StaticText(self.panelFK, -1, u'V\u2080 =')
        self.V0textAns = wx.StaticText(self.panelFK, -1, '               ')
        self.V0Box.Add(self.V0text, flag=wx.RIGHT, border=0)    # add stext to s box
        self.V0Box.Add(self.V0textAns, flag=wx.RIGHT, border=0)    # add stext to s box


        
        # *********************************************************************
        # Range of spacial Inhibition, s
        # s value inside self.boxfK

        # The values for s ************************
        # declare and initialize s values
        self.sVar = wx.TextCtrl(self.panelFK, value=str(self.s), size=(50, -1)) 
        sbox.Add(wx.StaticText(self.panelFK, -1, "s ="), flag=wx.RIGHT, border=0)    # add stext to s box
        sbox.Add(self.sVar, flag=wx.LEFT, border=0) # add self.sVar to s box
        
        # Add variables for sto fKvarsBox ************************
        self.fKvarsBox.Add((25, -1))
        self.fKvarsBox.Add(sbox, flag=wx.ALIGN_CENTRE_HORIZONTAL) #, border=10)
        # *********************************************************************
        
        




        # *********************************************************************
        # The f(K) plot
        # *********************************************************************
        self.figure = Figure(None)
        self.figure.set_facecolor('white')
        self.canvas  = FigureCanvasWxAgg( self.panelFK, -1, self.figure )
        self.subplot = self.figure.add_subplot( 111, axisbg='white' )
        #self.subplot.set_xlabel('K', fontsize=12)
        self.subplot.set_xlim(-2.0, 2.0)
        self.subplot.set_ylabel('f (k)', fontsize=12)
        self.subplot.set_xlabel('wave vector, k', fontsize=12)


        #x, y = self.canvas.GetSize()
        #self.canvas.SetSize((x, y-30))

        # *********************************************************************
        # Add the boxSigmoid elements
      
        self.boxfK.Add(self.fkEqBLUE, flag=wx.ALIGN_CENTRE_HORIZONTAL)
        self.boxfK.Add((-1, 20))
        self.boxfK.Add(self.fKvarsBox, flag=wx.ALIGN_CENTRE_HORIZONTAL)
        self.boxfK.Add((-1, 10))
        
        
        self.boxfK.Add(self.V0Box, flag=wx.ALIGN_CENTRE_HORIZONTAL)
        
        self.boxfK.Add((-1, 10))
        self.boxfK.Add(self.SprimeBox, flag=wx.ALIGN_CENTRE_HORIZONTAL)
        self.boxfK.Add(self.canvas) #, proportion=1) #, border=0, flag=wx.ALL) # | wx.EXPAND)
        
        
        # Add boxSigmoid to panelSigmoid         
        self.panelFK.SetSizer(self.boxfK)
        self.panelFK.SetMinSize((self.windowWidth, self.windowHeight))   
        # *********************************************************************







        # *********************************************************************
        # The values for self.panelRoot
        # *********************************************************************

        # Our Get Root button *************************** 
        self.getRootBtn = wx.ToggleButton(self.panelRoot, -1, 'Get Root', (120, 20))
        self.runButtonbox.Add(self.getRootBtn, flag=wx.EXPAND|wx.Left|wx.RIGHT|wx.TOP, border=5)

        # *********************************************************************
        # Add variables for stationary state ************************
        # *********************************************************************
        # Excitatory (ae) && inhibitory (ai) gain factor and external input I0

        # Sigmoid prime equation
        self.stationaryEq     = wx.StaticBitmap(self.panelRoot, wx.ID_ANY, wx.Bitmap(os.path.join(__location__, 'pics/stationary.png'))) 
        self.stationaryEqBox.Add(self.stationaryEq, flag=wx.EXPAND) 
        
        
        # The values for ae ************************
        # declare and initialize ae values
        aetext     = wx.StaticText(self.panelRoot, -1, u'ae =')
        self.aeVar = wx.TextCtrl(self.panelRoot, value=str(self.ae), size=(45, -1)) 
        self.aebox.Add(aetext, flag=wx.RIGHT, border=0)    # add aetext to aebox
        self.aebox.Add(self.aeVar, flag=wx.LEFT, border=0) # add self.aeVar to aebox
        # ******************************************

        # The values for ai ************************
        # declare and initialize ai values
        aitext     = wx.StaticText(self.panelRoot, -1, u'ai =')
        self.aiVar = wx.TextCtrl(self.panelRoot, value=str(self.ai), size=(45, -1)) 
        self.aibox.Add(aitext, flag=wx.RIGHT, border=0)    # add aitext to aibox
        self.aibox.Add(self.aiVar, flag=wx.LEFT, border=0) # add self.aiVar to aibox
        # ****************************************** 

        # The values for I0 ************************
        # declare and initialize I0 values
        I0text     = wx.StaticText(self.panelRoot, -1, u'I\u2080 =')
        self.I0Var = wx.TextCtrl(self.panelRoot, value=str(self.I0), size=(45, -1)) 
        self.I0box.Add(I0text, flag=wx.RIGHT, border=0)    # add aitext to aibox
        self.I0box.Add(self.I0Var, flag=wx.LEFT, border=0) # add self.aiVar to aibox
        # ****************************************** 


        # Add variables to stationaryVarBox ************************
        self.stationaryVarBox.Add(self.aebox, flag=wx.ALIGN_CENTRE_HORIZONTAL, border=10)
        self.stationaryVarBox.Add((10, -1))
        self.stationaryVarBox.Add(self.aibox, flag=wx.ALIGN_CENTRE_HORIZONTAL, border=10)
        self.stationaryVarBox.Add((10, -1))
        self.stationaryVarBox.Add(self.I0box, flag=wx.ALIGN_CENTRE_HORIZONTAL, border=10)
        
        
        # Add the equation and variable boxes to the stationary box
        self.stationaryBox.Add(self.stationaryEqBox, flag=wx.ALIGN_CENTRE_HORIZONTAL) #flag=wx.EXPAND|wx.Left|wx.RIGHT|wx.TOP, border=5)
        self.stationaryBox.Add((-1, 20))
        self.stationaryBox.Add(self.stationaryVarBox, flag=wx.ALIGN_CENTRE_HORIZONTAL)  
        # *********************************************************************


        # *********************************************************************
        # Add variables for Sigmoidal firing rate ************************
        # *********************************************************************
        # Sigmoid equation
        self.SEq     = wx.StaticBitmap(self.panelRoot, wx.ID_ANY, wx.Bitmap(os.path.join(__location__, 'pics/S.png'))) 
        self.SigmoidEqBox.Add(self.SEq, flag=wx.EXPAND) #, border=5)  # show equation as static bitmap
   
        # The values for c ************************
        # declare and initialize c values
        self.cText = wx.StaticText(self.panelRoot, -1, "c =")
        self.cVar  = wx.TextCtrl(self.panelRoot, value=str(self.c), size=(45, -1)) 
        cbox.Add(self.cText, flag=wx.RIGHT, border=0) # add self.cText to c box
        cbox.Add(self.cVar, flag=wx.LEFT, border=0)   # add self.cVar to c box
        # ******************************************

        # The values for theta ************************
        # declare and initialize theta values
        thetaText     = wx.StaticText(self.panelRoot, -1, u'\u0398 =')
        self.thetaVar = wx.TextCtrl(self.panelRoot, value=str(self.theta), size=(45, -1)) 
        thetabox.Add(thetaText, flag=wx.RIGHT, border=0)    # add thetaText to theta box
        thetabox.Add(self.thetaVar, flag=wx.LEFT, border=0) # add self.thetaVar to theta box
        # ******************************************

        # The values for S_0box ************************
        # declare and initialize S0 values
        s0Text     = wx.StaticText(self.panelRoot, -1, u'S\u2080 =')
        self.s0Var = wx.TextCtrl(self.panelRoot, value=str(self.s0), size=(45, -1)) 
        s0box.Add(s0Text, flag=wx.RIGHT, border=0)    # add s0Text S0 box
        s0box.Add(self.s0Var, flag=wx.LEFT, border=0) # add self.s0Var S0 box
        # ******************************************     


        # Add variables for S' to SprimeVarBox ************************
        self.SigmoidVarBox.Add(cbox, flag=wx.ALIGN_CENTRE_HORIZONTAL, border=10) 
        self.SigmoidVarBox.Add((20, -1))
        self.SigmoidVarBox.Add(thetabox, flag=wx.ALIGN_CENTRE_HORIZONTAL, border=10) 
        self.SigmoidVarBox.Add((20, -1))
        self.SigmoidVarBox.Add(s0box, flag=wx.ALIGN_CENTRE_HORIZONTAL) 

        
        # Add the equation and variable boxes to the S' box
        self.SigmoidBox.Add(self.SigmoidEqBox, flag=wx.ALIGN_CENTRE_HORIZONTAL) #flag=wx.EXPAND|wx.Left|wx.RIGHT|wx.TOP, border=5)
        self.SigmoidBox.Add((-1, 20))
        self.SigmoidBox.Add(self.SigmoidVarBox, flag=wx.ALIGN_CENTRE_HORIZONTAL)
        # *********************************************************************


        # *********************************************************************
        # Add variables for the voltage min, max & discretize  ****************
        # *********************************************************************

        # The values for the maximum V ************************
        # declare and initialize ae values
        #vmn     = wx.StaticText(self.panelRoot, -1, u'Vmin =')
        self.vMinTxt = wx.TextCtrl(self.panelRoot, value=str(self.Vmin), size=(45, -1)) 
        self.vMin.Add(wx.StaticText(self.panelRoot, -1, u'min ='), flag=wx.RIGHT, border=0)  
        self.vMin.Add(self.vMinTxt, flag=wx.LEFT, border=0) 
        # ******************************************

        # The values for the minimum V ************************
        # declare and initialize theta values
        #vmax     = wx.StaticText(self.panelRoot, -1, u'Vmax =')
        self.vMaxTxt = wx.TextCtrl(self.panelRoot, value=str(self.Vmax), size=(45, -1)) 
        self.vMax.Add(wx.StaticText(self.panelRoot, -1, u'max ='), flag=wx.RIGHT, border=0)  
        self.vMax.Add(self.vMaxTxt, flag=wx.LEFT, border=0) 
        # ****************************************** 

        # The values for the number of discretized units of V *****************
        # declare and initialize theta values
        #vmax     = wx.StaticText(self.panelRoot, -1, u'Vmax =')
        self.vDskTxt = wx.TextCtrl(self.panelRoot, value=str(self.Vdsk), size=(55, -1)) 
        self.vDiscr.Add(wx.StaticText(self.panelRoot, -1, u'units ='), flag=wx.RIGHT, border=0)  
        self.vDiscr.Add(self.vDskTxt, flag=wx.LEFT, border=0) 
        # ****************************************** 
        
        self.vBox.Add(self.vMin, flag=wx.ALIGN_CENTRE_HORIZONTAL, border=10)
        self.vBox.Add((20, -1))
        self.vBox.Add(self.vMax, flag=wx.ALIGN_CENTRE_HORIZONTAL, border=10)
        self.vBox.Add((20, -1))
        self.vBox.Add(self.vDiscr, flag=wx.ALIGN_CENTRE_HORIZONTAL, border=10)
        # *********************************************************************








        # *********************************************************************
        # The root plot
        
        self.figureRoot = Figure(None)
        self.figureRoot.set_facecolor('white')
        self.canvasRoot  = FigureCanvasWxAgg( self.panelRoot, -1, self.figureRoot )
        self.subplotRoot = self.figureRoot.add_subplot( 111, axisbg='white' )

        
        

        self.boxGetRoot.Add((-1, 10))
        self.boxGetRoot.Add(self.runButtonbox, flag=wx.CENTER) #, border=5)  # show equation as static bitmap
        self.boxGetRoot.Add((-1, 10))
        self.boxGetRoot.Add(self.stationaryBox, flag=wx.ALIGN_CENTRE_HORIZONTAL)
        

        self.boxGetRoot.Add((-1, 10))
        self.boxGetRoot.Add(self.SigmoidBox, flag=wx.ALIGN_CENTRE_HORIZONTAL)

        self.boxGetRoot.Add((-1, 10))
        self.boxGetRoot.Add(self.vBox, flag=wx.ALIGN_CENTRE_HORIZONTAL)


        self.boxGetRoot.Add(self.canvasRoot)
        
        self.panelRoot.SetSizer(self.boxGetRoot)
        self.panelRoot.SetMinSize((self.windowWidth, self.windowHeight))        
        # *********************************************************************




        # *********************************************************************
        # Events
        # *********************************************************************
        self.getRootBtn.Bind(wx.EVT_TOGGLEBUTTON, self.digRoot)        # bind get root button event
        self.aeVar.Bind(     wx.EVT_TEXT,         self.ae_Changed)     # bind ae event
        self.aiVar.Bind(     wx.EVT_TEXT,         self.ai_Changed)     # bind ae event
        self.I0Var.Bind(     wx.EVT_TEXT,         self.I0_Changed)     # bind ae event
        self.cVar.Bind(      wx.EVT_TEXT,         self.c_Changed)      # bind c event
        self.thetaVar.Bind(  wx.EVT_TEXT,         self.theta_Changed)  # bind theta event
        self.s0Var.Bind(     wx.EVT_TEXT,         self.S0_Changed)     # bind S0 event
        self.vMinTxt.Bind(   wx.EVT_TEXT,         self.min_Changed)    # bind min V event
        self.vMaxTxt.Bind(   wx.EVT_TEXT,         self.max_Changed)    # bind max V event
        self.vDskTxt.Bind(   wx.EVT_TEXT,         self.discrV_Changed) # bind V discretize event

        
        self.sVar.Bind(    wx.EVT_TEXT, self.s_Changed)     # bind S event
        
        
        #self.canvasRoot.Bind(wx.EVT_LEFT_DOWN, self.canvas_click)    # bind ae event







        # ***************************************************************************************************
        # Now put all the pieces together
        mainBox.Add(self.panelRoot, 2, wx.EXPAND)
        mainBox.Add(self.panelFK,   2, wx.ALL)
        
        
        self.SetSizer(mainBox)
        self.Layout()
        self.SetSize((1050, self.windowHeight))
        self.Centre()
        self.Show(True) 
        # ***************************************************************************************************        
        







    def digRoot(self, e):
        '''Get our root points and plot them '''
        
        self.subplotRoot.clear()
        self.getRootBtn.Hide()  # .HideWithEffect(5)
        dV   = (self.Vmax-self.Vmin)/float(self.Vdsk)
        Vold = self.Vmin
        V    = np.zeros(self.Vdsk)
        self.rootsX = [] # x axis roots
        self.rootsY = [] # y axis roots
        
        # our stationary states
        self.stationary = np.zeros(self.Vdsk) # ordinary
        
        #self.Vmin -= self.I0
        #self.Vmax -= self.I0
        
        
        for i in range(int(self.Vdsk)):
            Vnew = self.Vmin +float(i)*dV       # linear increase in V

            V[i] = Vnew
            self.stationary[i] = self.getStationaryState(Vnew) 

            # neg to pos OR pos to neg
            if (self.stationary[i]-Vnew) * (self.stationary[(i-1)]-Vold) < 0: 
                self.rootsX.append(((Vnew +Vold) / 2.0))     # got root
                self.rootsY.append(Vnew)
                #print 'root at V=%f' %(((Vnew +Vold) / 2.0)) # print root

            Vold=Vnew

        self.subplotRoot.set_xlabel(u'stationary field $V_0$', fontsize=12)
        self.subplotRoot.set_ylabel(u'($a_e$- $a_i$) $S$($V_0$) + $I_0$', fontsize=12)

        if self.Vmin < self.stationary.min():
            ymin = self.Vmin
        else:
            ymin = self.stationary.min()
            
        if self.Vmax > self.stationary.max():
            ymax = self.Vmax
        else:
            ymax = self.stationary.max()            
        
        self.subplotRoot.plot(V, V, 'r-')
        self.subplotRoot.plot(V, self.stationary, 'b-')
        self.subplotRoot.set_ylim(ymin-self.I0-.1,ymax+.1)
        self.subplotRoot.set_xlim(self.Vmin,self.Vmax)
        self.subplotRoot.plot(self.rootsX, self.rootsY, 'gD')
        eighty = (V.max() - V.min()) * .8
        self.numRoots = len(self.rootsY)
        for i in range(0, self.numRoots):
            self.subplotRoot.annotate(("%.7g" % round(self.rootsY[i],7)), xy=(self.rootsX[i], self.rootsY[i]), xytext=(self.rootsX[i], eighty-i), color='green', arrowprops=dict(facecolor='green', shrink=0.05))

        self.canvasRoot.Show()
        self.canvasRoot.draw()
        

        if self.numRoots > 1:
            selectRoot(self).Show()

        elif self.numRoots == 1:
            self.gotRoot(self.rootsY[0], ("%.7g" % round(self.rootsY[0],7)), True)
        else:
            self.gotRoot(None, '               ',   False)
        

        
    # ---------------------------------------------------------------   



    def gotRoot(self, root, stringRoot, findFK):
        self.V0 = root
        self.V0textAns.Label =  stringRoot 
        self.sigmoidPrime(root)
        self.fK(findFK)
        self.getRootBtn.Show() #WithEffect(5)

   

    # ---------------------------------------------------------------   


    def sigmoidPrime(self, V0=None): # sigmoid function
        '''S prime function '''

        if V0 == None:
            self.SPrime = None
            self.SPrimeEqText.SetLabel('          ')
            
        else:
            #V0 = 9.835341
            
            self.expo = np.exp(-self.c*(V0-self.theta))
            self.SPrime = (self.s0 * self.c * self.expo) / (1 + self.expo)**2    # S prime: the gradient of the curve
            self.SPrimeEqText.SetLabel(' ' +str(round(self.SPrime, 8)))

            print("alpha:", self.c)
            print("theta:", self.theta)
            print("S0:", self.s0)
            print("Sprime():", self.SPrime)
        

    # ---------------------------------------------------------------   
  
    def getStationaryState(self, V): # Stationary state
        
        def S(V):
            '''Sigmoid function '''
            return self.s0 / ( 1 + np.exp(-self.c*(V-self.theta))) # self.k is threshold
        
        return (self.ae-self.ai) * S(V) + self.I0


    # ---------------------------------------------------------------









    def fK(self, go=True):
        ''' Calculate: function of K then update the f(K) plot '''

        # Erase the previous graph
        self.subplot.clear() 
        
        if go:
            
            # Our f(K) data
            self.data = []
            num1 = (self.SPrime * self.ae)               # numerator 1 is a constant through K
            num2 = (self.SPrime * self.ai * self.s**3.0) # numerator 2 is a constant through K
            for k in self.Krange:
                div1 = num1 / np.sqrt(1.0 +k**2.0)**3
                div2 = num2 / np.sqrt(self.s**2.0 +k**2.0)**3
                self.data.append(div1 - div2)
 


            # set up the y axis
            miniDat = min(self.data)
            maxiDat = max(self.data)
            if miniDat < 0.1:
                if maxiDat < 1.0:
                    self.subplot.set_ylim(miniDat-0.2, 1.2)
                else:
                    self.subplot.set_ylim(miniDat-0.2, maxiDat+0.2)
            elif maxiDat < 1.0:
                self.subplot.set_ylim(0.0, 1.2)
            else:
                self.subplot.set_ylim(0.0, maxiDat+0.2)


    


            # Plot the graph based on stability
            if maxiDat > 1.0: # if unstable
                if self.isStable  == -1:  # was unstable only once before
                    #self.isStable  = -2   # now unstable more than once before
                    self.subplot.plot( self.xAxis, self.data, color='red')
                    
                elif self.isStable > -1:  # was not unstable last epoc
                    self.isStable  = -1   # first time unstable
                    self.figure.set_facecolor('red')
                    self.subplot.plot( self.xAxis, self.data, color='red')
                    self.fkEqBLUE.Hide()
                    self.boxFKeq.Replace(self.fkEqBLUE, self.fkEqRED)
                    self.fkEqRED.Center(wx.HORIZONTAL)
                    self.fkEqRED.Show()
            
            else:                       # if stable
                if self.isStable < 1:   # was not stable last epoc
                    self.isStable = 1  # first time stable
                    self.figure.set_facecolor('white')
                    self.subplot.plot( self.xAxis, self.data, color='blue')
                    self.fkEqRED.Hide() 
                    self.boxFKeq.Replace(self.fkEqRED, self.fkEqBLUE)
                    self.fkEqBLUE.Show()
                    
                elif self.isStable >= 1:  # was stable only once before
                    self.isStable  = 2    # now stable more than once before
                    self.subplot.plot( self.xAxis, self.data, color='blue')
    
    
            # plot the 2pi line and annotations
            self.subplot.plot( [-2.0, 2.0], [1.0, 1.0], color='purple')


        self.subplot.set_ylabel('f (k)', fontsize=12)
        self.subplot.set_xlabel('wave vector, k', fontsize=12)
        
        self.canvas.draw()










    # *********************************************************************
    # Events
    # *********************************************************************
    def ae_Changed(self, e):
        try:
            self.ae = float(e.GetEventObject().GetValue())   
        except ValueError:
            pass
        
    def ai_Changed(self, e):
        try:        
            self.ai = float(e.GetEventObject().GetValue()) 
        except ValueError:
            pass  

    def I0_Changed(self, e):
        try:          
            self.I0 = float(e.GetEventObject().GetValue()) 
        except ValueError:
            pass 

    def c_Changed(self, e):
        try:
            self.c = float(e.GetEventObject().GetValue())
        except ValueError:
            pass

    def theta_Changed(self, e):
        try:
            self.theta = float(e.GetEventObject().GetValue())
        except ValueError:
            pass

    def S0_Changed(self, e):
        try:
            self.s0 = float(e.GetEventObject().GetValue())
        except ValueError:
            pass

    def min_Changed(self, e):
        try:
            self.Vmin = float(e.GetEventObject().GetValue())
        except ValueError:
            pass

    def max_Changed(self, e):
        try:
            self.Vmax = float(e.GetEventObject().GetValue())
        except ValueError:
            pass

    def discrV_Changed(self, e):
        try:
            self.Vdsk = float(e.GetEventObject().GetValue())
        except ValueError:
            pass

    def s_Changed(self, e):
        try:
            inpt = float(e.GetEventObject().GetValue())
            if inpt != 0.0:
                self.s = inpt
                if self.SPrime is not None:
                    self.fK()      # update our f(K)
        except ValueError:
            pass









class selectRoot(wx.Frame):

    def __init__(self, parent):
        """Constructor"""
        wx.Frame.__init__(self, None, wx.ID_ANY, "Select a root")
        self.parent = parent
        
        
        panel = wx.Panel(self)
 
        msg = "Your " + str(int(parent.numRoots)) + " roots are:"
        message = wx.StaticText(panel, label=msg)
        
        #r1String = "  stable: "+str(parent.rootsX[0]) 
        r1String = str(parent.rootsY[0])
        self.root1Btn = wx.Button(panel, label=r1String)
        self.root1Btn.Bind(wx.EVT_BUTTON, self.onRoot1)
        
        #r2String = "unstable: "+str(parent.rootsX[1])
        r2String = str(parent.rootsY[1])
        self.root2Btn = wx.Button(panel, label=r2String)
        self.root2Btn.Bind(wx.EVT_BUTTON, self.onRoot2)
        
        if parent.numRoots > 2:
            #r3String = "  stable: "+str(parent.rootsX[2])
            r3String = str(parent.rootsY[2])
            self.root3Btn = wx.Button(panel, label=r3String)
            self.root3Btn.Bind(wx.EVT_BUTTON, self.onRoot3)
 
        sizer = wx.BoxSizer(wx.VERTICAL)
        flags = wx.ALL|wx.CENTER

        sizer.Add(message,       0, flags, 15)
        sizer.Add(self.root1Btn, 0, flags, 15)
        sizer.Add(self.root2Btn, 0, flags, 15)
        if parent.numRoots > 2:
            sizer.Add(self.root3Btn, 0, flags, 15)

        panel.SetSizer(sizer)


    def onRoot1(self, event):
        self.parent.gotRoot(self.parent.rootsY[0], ("%.7g" % round(self.parent.rootsY[0],7)), True)
        self.Close()
 
    def onRoot2(self, event):
        self.parent.gotRoot(self.parent.rootsY[1], ("%.7g" % round(self.parent.rootsY[1],7)), True)
        self.Close()
        
    def onRoot3(self, event):
        self.parent.gotRoot(self.parent.rootsY[2], ("%.7g" % round(self.parent.rootsY[2],7)), True)
        self.Close()
         


                     
def main():
    
    ex = wx.App()
    findRoots(None)    
    ex.MainLoop()    

if __name__ == '__main__':
    main()  