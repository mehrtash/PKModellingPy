from __future__ import division
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import math
import SimpleITK as sitk
from scipy.optimize import curve_fit
from scipy import stats

# Signal to concentration converter
class SignalToConcentrationConverter:
    '''
    This is a class to convert the MR signal to bolus concentration curve.
    Usage
    
    References
    [1]
    '''
    def __init__ (self):
        self.signal = []
        self.TR = None # Repetition Time
        self.FA = None # Flip Angle
        self.T10 = None # T1 before injection of CA
        self.Relaxivity = None
        self.BAT = None

    def convert(self):
        c = []
        if len(self.signal) > 0:
            # TODO: For real signals replace with averaging
            s0 = self.signal[0]
            s0 = mean(self.signal[:self.BAT])
            if s0 < 1 :
                s0 = 1
                print 's0 is <1, s0 was assigned as 1'
                #print self.signal

            for s in self.signal:
                exp_TR_BloodT1 = exp(-self.TR/self.T10)
                alpha = self.FA * pi/180
                cos_alpha = cos(alpha)
                constB = (1- exp_TR_BloodT1)/(1-cos_alpha*exp_TR_BloodT1)
                constA = s/s0
                value = (1 - constA*constB)/(1- constA*constB*cos_alpha)
                if value < 0:
                    concentration = 0
                else:
                    log_value = log(value)
                    ROft = (-1/self.TR)*log_value
                    Cb = (ROft  - (1/self.T10))/ self.Relaxivity
                    if (Cb < 0):
                        Cb = 0
                    concentration = Cb
                c.append(concentration)
        else:
            print 'you must provide an appropriate signal'
        return np.array(c)

class PKSolver(object):
    '''
    This is a class to solve the PK Modelling problem with Tofts model.
    Usage
    
    References
    [1]
    '''
    samplingRate = None
    hematocrit= None
    rSquarred = 0

    def __init__(self):
        self.__AIF = []
        self.__BATAIF = 0
        self.__BATTRF = 0
        self.__TRF = []
        self.__AIFShifted = []
        self.__TRFShifted = []
        self.__ktrans = -1
        self.__ve = -1

    def getAIF(self):
        return self.__AIF

    def setAIF(self, val):
        self.__AIF = val
        self.__setBAT()

    AIF = property(getAIF, setAIF)

    def getTRF(self):
        return self.__TRF

    def setTRF(self, val):
        self.__TRF = val 
        self.__setBAT()

    TRF = property(getTRF, setTRF)

    def getBATAIF(self):
        return self.__BATAIF

    def setBATAIF(self, val):
        self.__BATAIF = val
        self.__setBAT()

    BATAIF = property(getBATAIF, setBATAIF)

    def getBATTRF(self):
        return self.__BATTRF
    
    def setBATTRF(self, val):
        self.__BATTRF = val
        self.__setBAT()

    BATTRF = property(getBATTRF, setBATTRF)

    def func(self,t,Ktrans,ve):
        result = None
        if self.mode == 'shifted':
            p = len(self.__AIFShifted)
            t = linspace(0, self.samplingRate*(p-1),p)/60
            delta= t[1] - t[0]
            result = (1/(1-self.hematocrit))*Ktrans*(
                convolve(self.__AIFShifted,exp(-Ktrans/ve*t))*delta)[0:len(t)]
        else:
            p = len(self.__AIF)
            t = linspace(0, self.samplingRate*(p-1),p)/60
            delta= t[1] - t[0]
            result = (1/(1-self.hematocrit))*Ktrans*(
                    convolve(self.__AIF,exp(-Ktrans/ve*t))*delta)[0:len(t)]   
        return result

    def solve(self):
        p = len(self.__AIFShifted)
        t = linspace(0, self.samplingRate*(p-1),p)
        #t = self.__t[:]/60  # ktrans is in min-1
        self.mode = 'shifted'
        try:
            popt, pcov= curve_fit(self.func, t, self.__TRFShifted,np.array([0.1,1]))
            if popt[1] < 0 or popt[1]>10:
              popt, pcov= curve_fit(self.func, t, self.__TRFShifted,np.array([0.1,0.1]))
            if popt[1] < 0 or popt[1]>10:
              popt, pcov= curve_fit(self.func, t, self.__TRFShifted,np.array([0.1,0.5]))
            if popt[1] < 0 or popt[1]>10:
              popt, pcov= curve_fit(self.func, t, self.__TRFShifted,np.array([0.1,0.9]))
            if popt[1] < 0 or popt[1]>10:
              popt, pcov= curve_fit(self.func, t, self.__TRFShifted,np.array([1,0.8]))
            if popt[1] < 0 or popt[1]>10:
              popt, pcov= curve_fit(self.func, t, self.__TRFShifted,np.array([1,0.2]))
            if popt[1] < 0 or popt[1]>10:
              popt, pcov= curve_fit(self.func, t, self.__TRFShifted,np.array([1,0.4]))
            if popt[1] < 0 or popt[1]>10:
              popt[1] = -1
        except Exception, err:
          print Exception, err
          #print 'shifted TRF is: ', self.__TRFShifted
          popt = [-1, -1]
          pcov = [-1,-1]

        #perr = np.sqrt(np.diag(pcov))
        perr = [-1, -1]
        [self.__ktrans,self.__ve] = popt
        [self.__ktransError,self.__veError] = perr
        '''
        if self.__ve < 0:
          self.__ve = -1
        if self.__ktrans < 0:
          self.__ktrans = -1
        '''
        if self.__ve == -1 or self.__ktrans == -1:
          self.__ve = -1
          self.__ktrans = -1
        return np.array([self.__ktrans, self.__ve,self.__ktransError,self.__veError])

    def calculateGoodnessOfFit(self):
        if self.__ktrans != -1 and self.__ve != -1:                    
            p = len(self.__TRFShifted)
            t = linspace(0, self.samplingRate*(p-1),p)/60
            observed = self.__TRFShifted
            #self.mode = 'original'
            self.mode = 'shifted'
            expected = self.func(t,self.__ktrans,self.__ve)
            self.rSquarred = 0
            slope, intercept, r_value, p_value, std_err = stats.linregress(observed,expected) 
            self.rSquarred = r_value**2
        else:
            self.rSquarred = -1
        return self.rSquarred

    def plotResults(self):
        if self.__ktrans != -1:
            p = len(self.__TRFShifted)
            t = linspace(0, self.samplingRate*(p-1),p)/60
            fig, ax= plt.subplots()
            fig.set_size_inches(16,4)
            ax1 = plt.subplot(1,2,1)
            ax1.grid(True)
            plot(t , self.__AIFShifted,':o' ,mfc='w',mec='b',ms=6 ,label='AIF shifted')
            plot(t , self.__TRFShifted,':o' ,mfc='w',mec='r',ms=6 ,label='TRF*3 shifted')
            plt.legend(loc=0,prop={'size':14})
            ax2 = plt.subplot(1,2,2)
            ax2.grid(True)
            p = len(self.__TRFShifted)
            t = linspace(0, self.samplingRate*(p-1),p)/60
            plot(t , self.__TRFShifted,'o' ,mfc='w',mec='b',ms=6 ,label='observed')
            self.mode = 'shifted'
            fitted = self.func(t , self.__ktrans, self.__ve)
            plot(t, fitted, 'r:',label='fitted')
            plt.legend(loc=0,prop={'size':14})
            rsquared = self.calculateGoodnessOfFit()
            title('ktrans: ' + "%.2f"%self.__ktrans +  
                  ', ve: ' + "%.2f"%self.__ve + 
                  ', R2: '+ "%.2f"%self.rSquarred)
        else:
            print 'plot error: you must run solver first'

    def __setBAT(self):
        # Apply fractional-shift if BAT is non-integer
        diff = self.__BATTRF - int(self.__BATTRF) 
        if diff != 0 :
            TRF = self.__applyFractionalShift(self.__TRF, diff)
            BATTRF = int(self.__BATTRF)
        else:
            TRF = self.__TRF
            BATTRF = self.__BATTRF

        diff = self.__BATAIF - int(self.__BATAIF) 
        if diff != 0 :
            print 'warning: non integer shift for aif'
            AIF = self.__applyFractionalShift(self.__AIF, diff)
            BATAIF = int(self.__BATAIF)
        else:
            AIF = self.__AIF
            BATAIF = int(self.__BATAIF)

        # Applying the integer part of the shift
        if BATTRF > len(self.__TRF) or BATAIF > len(self.__AIF):
            print 'Error: You must assign smaller BAT or assigng appropriate concentration signals first!'
            print 'BATTRF is:', BATTRF
            print 'len TRF is:', len(self.__TRF)
            print 'BATAIF is:', BATAIF
            print 'len AIF is:', len(self.__AIF)

        else:
            if BATTRF > BATAIF:
                padding = BATTRF - BATAIF
                self.__TRFShifted = TRF[BATTRF:]
                self.__AIFShifted = AIF[BATAIF:-padding]
            elif BATTRF < BATAIF:
                padding = BATAIF - BATTRF
                self.__TRFShifted = TRF[BATTRF:-padding]
                self.__AIFShifted = AIF[BATAIF:]
            else:
                self.__TRFShifted = TRF[BATTRF:]
                self.__AIFShifted = AIF[BATAIF:]

    def __applyFractionalShift(self,c, diff):
        l = len(c)
        if l> 0:
          cDelayed = zeros(l)
          for i in xrange(0,l-1):
             cDelayed[i] = (1-diff)*c[i] + diff*c[i+1]
          # TODO: check if the following line is correct
          cDelayed[l-1] = c[l-1]
        else:
          cDelayed = c
        return cDelayed
