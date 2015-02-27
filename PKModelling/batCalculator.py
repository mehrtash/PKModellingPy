from __future__ import division
import SimpleITK as sitk
from pylab import *
import numpy as np
from numpy.linalg import inv
import scipy.optimize as optimization

class BATCalculator(object):
    ''' 
    This is a class for calculating the bolus arrival time
    with different proposed mehtods in the literature, including
    L-L model(Linear-Linear model [1]), L-Q (Linear-Quadratic model [1])
    P-L ( Piecewise-Linear model [2]) and our propsed mehtod [3].
    ----------------------------------------------------------------
    Usage:
    
    ----------------------------------------------------------------
    References
    [1] Cheong, L. H., Koh, T. S., & Hou, Z. (2003). An automatic 
    approach for estimating bolus arrival time in dynamic contrast 
    MRI using piecewise continuous regression models. 
    Physics in Medicine and Biology, 48(5), N83–8.
    [2] Singh, A., Rathore, R. K. S., Haris, M., Verma, S. K., Husain
    , N., & Gupta, R. K. (2009). Improved bolus arrival time and arterial
    input function estimation for tracer kinetic analysis in DCE-MRI. 
    Journal of Magnetic Resonance Imaging : JMRI, 29(1), 166–76. 
    doi:10.1002/jmri.21624
    '''     
    def __init__ (self, method = 'L-L'):
        self.method = method
        self.c = []
        self.BSIterationNumber = 10
    def calculate(self):
        output = -1, []
        if len(self.c) < 2:
            print "Error: Assign a concentration curve!"
        else:
            if self.method == 'LL':
                output  = self.__calculateBATLLMethod(self.c)
            elif self.method == 'LQ':
                output  = self.__calculateBATLQMethod(self.c)
            elif self.method == 'PL':
                output  = self.__calculateBATPLMethod(self.c)
            elif self.method == 'PLBS':
                output  = self.__calculateBATPLBSMethod(self.c)
            elif self.method == 'XY': #TODO: Assign an appropriate name
                output  = self.__calculateBATXYMethod(self.c)
            elif self.method == 'XYBS': #TODO: Assign an appropriate name
                output  = self.__calculateBATXYBSMethod(self.c)
            elif self.method == 'PG': #TODO: Assign an appropriate name
                output  = self.__calculateBATPGMethod(self.c)
            else:
                print 'Error: Methods are LL, LQ, PL, XY and XYBS'
            return output

    def __calculateBATPGMethod (self, c):
      i = -1
      bat = -1
      maxIndex = -1
      c = c.astype('float')
      cprime = np.diff(c)
      # finding the peak of gradient
      maxValue = np.amax(cprime)
      maxIndex = np.where(cprime == maxValue)[0][0]
      skip1 = 0
      thresh = maxValue / 10.0
      for i in xrange(maxIndex, skip1,-1):
          if(cprime[i] < thresh):
              break
      if i != -1:
        bat = i+1;
      return bat, maxIndex

    def __calculateBATLLMethod(self,c):
        bat, beta, p = self.calculateBATLLModel(c)
        cExpected = self.LLFunction(bat, beta, p)
        return bat, cExpected

    def __calculateBATLQMethod(self,c):
        bat, beta, p = self.calculateBATLQModel(c)
        cExpected = self.LQFunction(bat, beta, p)
        return bat, cExpected
    
    def __calculateBATPLMethod(self,c):
        n = len(c)
        alpha, beta, h = self.calculateBATPLModel(c)
        cExpected = self.PLFunction(alpha, beta, h,n) 
        return alpha, cExpected
     
    def __calculateBATPLBSMethod(self,c):
        n = len(c)
        alpha, beta, h = self.calculateBATPLModel(c)
        alpha, beta = self.midPointIterationsPL(c,alpha,beta,self.BSIterationNumber)
        cExpected = self.PLFunction(alpha, beta, h,n) 
        return alpha, cExpected
    
    def __calculateBATXYMethod(self,c):
        bat, beta, a,b = self.calculateBATXYModel(c)
        cExpected = self.LLFunction(bat, beta, b)
        return bat, cExpected

    def __calculateBATXYBSMethod(self,c):
        bat, beta, a,b = self.calculateBATXYModel(c)
        bat = self.midPointIterationsXY(c[a:b],bat-a,self.BSIterationNumber)+a
        cExpected = self.LLFunction(bat, beta, b)
        return bat, cExpected

    def calculateBATLLModel (self,c):
        q = np.where(c == max(c))
        p = q[0][0]
        beta = [-1,-1]
        if p > 1 :
            c = c[0:p]
            c.reshape(p,1)        
            # find the index of maximum element in array
            sse = []
            betaValues = []
            #Step 0: Let p be the index to the maximum element in array C.
            #Step 1: For k = 1to p − 1, do steps 1.1 to 1.3.
            for k in xrange(0,p-1):
                sse_k , beta = self.calculateSSEkLLModel(c,k,p) 
                betaValues.append(beta)
                sse.append(sse_k)
            #Step 2: Let kmin be the index to the minimum element in array SSE, then the estimated BAT is t[kmin].2.2.
            #print sse
            #print "Bolus arrival time is:", 
            sse_min = np.amin(sse)
            #print sse_min
            bat = np.where(sse == sse_min)[0][0]
            beta = betaValues[bat]
            return bat, beta, p
        else:
            #print 'Calculating LL max index is less than 1'
            bat = -1
            return bat,beta,p

    def calculateSSEkLLModel( self, c, k, p):
        #1.1. Construct X.
        x = np.zeros((p,2))
        for i in xrange(0,p):
            x[i,0] = 1
            if i<= k:
                x[i,1] = 0
            else:
                x[i,1] = i-k
        #1.2. Solve equation (4)for βlsq (using eithermatrix inversion or constrained least- squares methods).
        #print x
        #print c
        beta = np.dot(inv(np.dot(np.transpose(x),x)),np.dot(np.transpose(x),c))
        # print "beta is:", beta
        #1.3. Compute SSE [k].
        sse_k = np.dot(np.transpose((c-np.dot(x,beta))),(c-np.dot(x,beta)))
        return sse_k, beta      

    def LLFunction (self, bat, beta, p):
        x = np.zeros((p,2))
        for i in xrange(0,p):
            x[i,0] = 1
            if i <= bat:
                x[i,1] = 0
            else:
                x[i,1] = i-bat
        return np.dot(x,beta)

    def LLCostFunction (self, beta, x, c):
        return c-np.dot(x,beta)
  
    def calculateBATLLModel_LSQ (self, c):
        p = np.where(c == max(c))[0][0]
        if p > 2:
            c = c[0:p]
            c.reshape(p,1)
            # find the index of maximum element in array
            sse = []
            #Step 0: Let p be the index to the maximum element in array C.
            #Step 1: For k = 1to p − 1, do steps 1.1 to 1.3.
            for k in xrange(0,p-1):
                #1.1. Construct X.
                x = np.zeros((p,2))
                for i in xrange(0,p):
                    x[i,0] = 1
                    if i<= k:
                        x[i,1] = 0
                    else:
                        x[i,1] = i-k
                #1.2. Solve equation (4)for βlsq (using eithermatrix inversion or constrained least- squares methods).
                beta0 = np.ones(2)
                beta0.reshape(2,1)
                beta = optimization.leastsq(self.LLCostFunction, beta0, args=(x, c))[0]
                beta.reshape(2,1)
                #print beta
                #print "beta is:", beta
                #1.3. Compute SSE [k].
                sse_k = np.dot(np.transpose((c-np.dot(x,beta))),(c-np.dot(x,beta)))
                #print sse_k
                #print "SSE is:", sse_k[0,0]
                sse.append(sse_k)
            #Step 2: Let kmin be the index to the minimum element in array SSE, then the estimated BAT is t[kmin].2.2.
            #print sse
            #print "Bolus arrival time is:", 
            sse_min = np.amin(sse)
            #print sse_min
            bat = np.where(sse == sse_min)[0][0]
            return bat, beta, p
        else:
            #print 'concentration curve is not appropriate'
            bat = -1     
            return bat

    def LQFunction (self, bat, beta, p):
        x = np.zeros((p,3))
        for i in xrange(0,p):
            x[i,0] = 1
            if i <= bat:
                x[i,1] = 0
                x[i,2] = 0
            else:
                x[i,1] = i-bat
                x[i,2] = square(i-bat)
        return np.dot(x,beta)

    def LQCostFunction (self, beta, x, c):
        return c-np.dot(x,beta)

    def calculateBATLQModel (self, c):
        p = np.where(c == max(c))[0][0]
        beta = [-1,-1,-1]
        if p >2 :
            c = c[0:p]
            c.reshape(p,1)
            betaValues = []
            sse = []
            for k in xrange(0,p-1):
                x = np.zeros((p,3))
                for i in xrange(0,p):
                    x[i,0] = 1
                    if i<= k:
                        x[i,1] = 0
                        x[i,2] = 0
                    else:
                        x[i,1] = i-k
                        x[i,2] = square(i-k)
                beta0 = np.ones(3)
                beta0.reshape(3,1)
                # Without constraints
                #'''
                beta = optimization.leastsq(self.LQCostFunction, beta0, args=(x, c))[0]
                #print beta

                '''
                if beta[1]< 0: 
                    print "negative beta1"
                if  beta[2]<0:
                    #print "negative beta2"
                    
                '''
                '''
                bounds = np.array([[-inf,inf], [0, inf], [0,inf]])
                #print bounds
                #With constraints
                beta =leastsq_bounds( LQFunction, beta0, bounds, 10, args=(x, c))[0]
                #print beta
                '''
                beta.reshape(3,1)
                sse_k = np.dot(np.transpose((c-np.dot(x,beta))),(c-np.dot(x,beta)))
                #print sse_k
                #print "SSE is:", sse_k[0,0]
                sse.append(sse_k)
                betaValues.append(beta)
            #Step 2: Let kmin be the index to the minimum element in array SSE, then the estimated BAT is t[kmin].2.2.
            #print sse
            #print "Bolus arrival time is:", 
            sse_min = np.amin(sse)
            #print sse_min
            minIndex = np.where(sse == sse_min)[0]
            if minIndex:
              bat = minIndex[0]
            else:
              bat = -1
            beta = betaValues[bat]
            return bat, beta, p
        else:
            print 'concentration curve is not appropriate'
            bat = -1
            return bat,beta,p

    def PLFunction (self, alpha, beta , h, n):
        x = np.zeros((n,3))
        for i in xrange(0,n):
            x[i,0] = 1
            if i <= alpha:
                x[i,1] = 0
                x[i,2] = 0
            elif i > alpha and i <= beta:
                x[i,1] = i-alpha
                x[i,2] = 0
            elif i> beta:
                x[i,1] = beta-alpha
                x[i,2] = i - beta
        return np.dot(x,h)

    def PLCostFunction (self, h, x, c):
        return c-np.dot(x,h)

    def calculateBATPLModel (self, c):
        n = len(c)
        if n >2 :
            c.reshape(n,1)
            params = []
            sse = []
            for alpha in xrange(0,n-1):
                for beta in xrange(alpha,n):
                    sse_k,h = self.calculateSSEkPLModel(c, alpha, beta,n)
                    sse.append(sse_k)
                    params.append([alpha, beta, h])
            #Step 2: Let kmin be the index to the minimum element in array SSE, then the estimated BAT is t[kmin].2.2.
            #print sse
            #print "Bolus arrival time is:", 
            sse_min = np.amin(sse)
            #print sse_min
            sse_minIndex = np.where(sse == sse_min)[0][0]
            return params[sse_minIndex]
        else:
            print 'concentration curve is not appropriate'
            bat = -1 
            return bat
      
    def calculateSSEkPLModel(self, c, alpha, beta,n):
        x = np.zeros((n,3))
        for i in xrange(0,n):
            x[i,0] = 1
            if i <= alpha:
                x[i,1] = 0
                x[i,2] = 0
            elif i > alpha and i <= beta:
                x[i,1] = i-alpha
                x[i,2] = 0
            elif i> beta:
                x[i,1] = beta-alpha
                x[i,2] = i - beta

        h0 = np.ones(3)
        h0.reshape(3,1)
        h = optimization.leastsq(self.PLCostFunction, h0 , args=(x, c))[0]
        '''
        #print beta
        if h[1]< 0 or h[2]<0:
            print "negative beta1 or beta2"
        bounds = np.array([[-inf,inf], [0, inf], [0,inf]])
        #print bounds
        #With constraints
        beta =leastsq_bounds( LQFunction, beta0, bounds, 10, args=(x, c))[0]
        #print beta
        '''
        h.reshape(3,1)
        sse_k = np.dot(np.transpose((c-np.dot(x,h))),(c-np.dot(x,h)))
        return sse_k, h
 
    def midPointIterationsPL(self, c, alpha, beta, iterationNumber):
        #n is the number of iterations
        n = len(c)
        currentAlpha = float(alpha)
        currentBeta = beta
        currentStepSize = 1
        for i in xrange(0,iterationNumber):
            # for alpha
            rightShift = float(currentAlpha + currentStepSize/2)
            leftShift = float(currentAlpha - currentStepSize/2)
            rightError = self.calculateSSEkPLModel(c, rightShift,currentBeta,n)[0]
            leftError = self.calculateSSEkPLModel(c, leftShift, currentBeta,n)[0]
            if rightError < leftError:
                currentAlpha = rightShift
            else:
                currentAlpha = leftShift
            # for beta
            rightShift = float(currentBeta) + currentStepSize/2
            leftShift = float(currentBeta) - currentStepSize/2
            rightError = self.calculateSSEkPLModel(c, currentAlpha, rightShift,n)[0]
            leftError = self.calculateSSEkPLModel(c, currentAlpha, leftShift, n)[0]
            if rightError < leftError:
                currentBeta = rightShift
            else:
                currentBeta = leftShift
            currentStepSize = currentStepSize/2
        return currentAlpha, currentBeta
 
    def calculateBATXYModel(self,c):
        c = c.astype('float')
        p = len(c)
        x = linspace(0,p-1,p)
        normalizedC = (c- min(c))/(max(c)-min(c))
        normalizedCMean = np.mean(normalizedC)
        # 
        # Calculate Derrivative of c and normalization
        #
        cprime = np.diff(c)
        # finding the peak of gradient
        peakGradientIndex = np.where(cprime == max(cprime))[0]
        '''
        normalizedCPrime = (cprime- min(cprime))/(max(cprime)-min(cprime))
        normalizedCPrimeMean = np.mean(normalizedCPrime)
        vicinity_index = -1
        movingAvC = c[0]
        maximumC = np.max(c)
        minimumC = np.min(c)
        #
        # Calculating vicinity_index (an estimation of BAT) using peak gradient and jump from moving average
        #
        for index,element in enumerate(normalizedCPrime):
            movingAvC = (movingAvC + c[index])/2
            if ((c[index] - movingAvC) > 0.05*(maximumC-minimumC) and  element > normalizedCPrimeMean *1.2): 
                vicinity_index = index        
                break
        # Making the vicinity window
        '''
        windowSize =  int(len(c)/5)
        #print 'window size is: ', windowSize
        vicinity_index = peakGradientIndex
        #print 'vicinity index is: ', vicinity_index
        if (vicinity_index - windowSize)<0:
            a = 0
        else:
            a = vicinity_index - windowSize
        b = int(vicinity_index + windowSize/2)
        #print b
        #
        # Create the new signal in the vicinity of BAT (cBATVicinity)
        #
        cBATVicinity = c[a:b]
        maxCBATVic = max(cBATVicinity)
        # remove declining values
        maxIndex = np.where(cBATVicinity == maxCBATVic)[0][0]
        b = a + maxIndex + 1
        cBATVicinity = c[a:b]
        #print cBATVicinity
        if len(cBATVicinity) > 5:
            # Using the Linear-Linear model to calculate the BAT which minimizes the SSE
            [calcualtedBAT, beta, p]  = self.calculateBATLLModel(cBATVicinity)
        else:
            print "cBATVicinity has less than 5 samples"
            [calcualtedBAT, beta, p]  = [0,0,0]
        bat = a + calcualtedBAT
        return bat,beta, a, b

    def midPointIterationsXY(self, c, bat, iterationNumber):
        #n is the number of iterations
        n = len(c)
        currentBAT = bat
        currentStepSize = 1
        for i in xrange(0,iterationNumber):
            rightShift = float(currentBAT + currentStepSize/2)
            leftShift = float(currentBAT - currentStepSize/2)
            rightError = self.calculateSSEkLLModel(c, rightShift,n)[0]
            leftError = self.calculateSSEkLLModel(c, leftShift, n)[0]

            if rightError < leftError:
                currentBAT = rightShift
            else:
                currentBAT = leftShift
            currentStepSize = currentStepSize/2
        return currentBAT
