from __future__ import division
from matplotlib import gridspec
from IPython.html.widgets import interact
from IPython.html import widgets
from pkModelling import *
from IPython.display import clear_output, display, HTML

class InteractivePKModelling:
    def __init__(self,ndaImage, aif, converterFA,converterTR,converterT10,converterBAT,
                 converterRelaxitivity,solverHematocrit,solverSamplingRate,solverInitBATAIF, solverInitBATTRF):
        self.solver = PKSolver()
        self.converter = SignalToConcentrationConverter()
        self.ndaImage = ndaImage
        self.aif = aif
        self.converterFA = converterFA
        self.converterTR = converterTR
        self.converterT10 = converterT10
        self.converterBAT = converterBAT
        self.converterRelaxitivity = converterRelaxitivity
        self.solverHematocrit = solverHematocrit
        self.solverSamplingRate = solverSamplingRate
        self.solverInitBATAIF = solverInitBATAIF
        self.solverInitBATTRF = solverInitBATTRF
        
    def toftsModel(self,Ktrans, ve, cp, samplingRate, p, hematocrit):
        result = None
        t = linspace(0, samplingRate*(p-1),p)/60
        delta= t[1] - t[0]
        result = (1/(1-hematocrit))*Ktrans*(
                convolve(cp,exp(-Ktrans/ve*t))*delta)[0:len(t)]   
        return result

    def browseImages(self):
        m,n,p,q = np.shape(self.ndaImage)
        # Case for single slice images like qiba DRO
        if m==1:
          interact(self.viewImage, 
                       frame=(0,int(q)-1),
                       zoom =(1,10,1),
                       i = (0,int(n)-1),
                       j = (0,int(p)-1),
                       k=0, 
                       batAIF = widgets.FloatSlider(min=0, max=int(q)-1, step=1, value=self.solverInitBATAIF),
                       batTRF = widgets.FloatSlider(min=0, max=int(q)-1, step=0.5, value=self.solverInitBATTRF),
                       PK_Analysis= True)

        else:
          interact(self.viewImage, 
                       frame=(0,int(q)-1),
                       zoom =(0,10,1),
                       i = (0,int(n)-1),
                       j = (0,int(p)-1), 
                       k = (0,int(m)-1), 
                       batAIF = widgets.FloatSlider(min=0, max=int(q)-1, step=1, value=self.solverInitBATAIF),
                       batTRF = widgets.FloatSlider(min=0, max=int(q)-1, step=0.5, value=self.solverInitBATTRF),
                       PK_Analysis= True)

    def setAxisColor(self,ax):
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.spines['right'].set_color('w')
    
    def viewImage(self,frame, zoom, i, j, k, batAIF, batTRF, PK_Analysis):
            m,n,p,q = np.shape(self.ndaImage)
            x = []
            labels = []
            samplingRate = self.solverSamplingRate
            totalTime = int(int(q)*samplingRate/60)
            for h in xrange(0,totalTime+1):
                x.append(int((60/samplingRate)*h))
                labels.append(h)
            fig, ax = plt.subplots()
            fig.set_size_inches(16,16)
            rect = fig.patch
            rect.set_facecolor('0.15')
            rect.set_edgecolor('w')
            gs = gridspec.GridSpec(3, 3)
            gs.update(wspace=0.2, hspace=0.3) # set the spacing between axes. 
            ax1 = plt.subplot(gs[0, 0])
            self.setAxisColor(ax1)
            plt.plot([j],[i],'+' ,mfc='w',mec='r',ms=18 )
            plt.imshow(self.ndaImage[k,:,:,frame], cmap=plt.cm.gray, 
                       interpolation='nearest' ,vmin=20, vmax=1500)
            plt.title('Time :'+str(samplingRate*frame) + ' sec',color= 'w')
            #ax2 = plt.subplot(2,2,2)
            ax2 = plt.subplot(gs[0, 1:3],axisbg='0.15')
            st = self.ndaImage[k,i,j,:]
            converter = self.converter
            converter.signal = st
            converter.FA = self.converterFA
            converter.TR = self.converterTR
            converter.T10 = self.converterT10
            converter.BAT = self.converterBAT
            converter.Relaxivity = self.converterRelaxitivity
            TRF = converter.convert()
            plt.plot(self.aif,'go-' ,mfc='g',mec='g',ms=5 ,label='AIF')
            plt.plot(TRF*5,'yo-' ,mfc='y',mec='y',ms=5 ,label='TRF*5')
            plt.axvline(batAIF, color='c', linestyle='dashed',label= 'BAT AIF')
            plt.axvline(batTRF, color='m', linestyle='dashed',label= 'BAT TRF')
            plt.axvline(frame , color='r', linestyle='dashed',label= 'current Frame')
            leg2 = ax2.legend(prop={'size':12})
            texts = leg2.get_texts()
            frame2 = leg2.get_frame()
            frame2.set_edgecolor('w')
            for text in texts:
                text.set_color('w')
            self.setAxisColor(ax2)
            #ax2.grid(color='w',linewidth='1')
            ax2.set_title('AIF and TRF')
            plt.xticks(x, labels)
            plt.xlabel('Time [min]',color='w')
            plt.ylabel('concentration [mmol/l]',color = 'w')
            ax3 = plt.subplot(gs[1, 0])
            self.setAxisColor(ax3)
            extent = 1/zoom*100
            plt.plot([extent],[extent],'+' ,linewidth=5,mec='r',ms=28 )
            plt.imshow(self.ndaImage[k,i-extent:i+extent,j-extent:j+extent,frame], cmap=plt.cm.gray, 
                       interpolation='nearest' ,vmin=20, vmax=1500)
            ax4 = plt.subplot(gs[1, 1:3])
                        
            if PK_Analysis:
               # SOLVER Parameters
                solver = self.solver
                solver.AIF = self.aif
                solver.TRF = TRF
                solver.BATAIF = batAIF 
                solver.BATTRF = batTRF
                solver.samplingRate = self.solverSamplingRate
                solver.hematocrit = self.solverHematocrit
                result = solver.solve()
                aifShifted = self.aif[batAIF:]
                p = len(aifShifted)
                if result[0] != -1 and result[1] != -1:
                    fitted = self.toftsModel(result[0],result[1],aifShifted, samplingRate,p,solver.hematocrit)
                    plt.plot(fitted,'r-' ,mfc='w',mec='r',ms=6 ,label=('Fitted' +', Ktrans: '+
                                    str("%0.2f"%result[0])+' ve: ' + str("%0.2f"%result[1])),linewidth =1.5)
            ax4.grid(color='w',linewidth='1')
            plt.plot(TRF[int(batTRF):],'go-' ,mfc='g',mec='g',ms=6 ,label='Observed (Shifted)')
            plt.ylabel('concentration [mmol/l]',color='w')
            ax4.set_title('TRF',color='w')
            leg4 = ax4.legend(loc = 0, prop={'size':12})
            texts = leg4.get_texts()
            frame = leg4.get_frame()
            frame.set_edgecolor('w')
            for text in texts:
                text.set_color('w')
            self.setAxisColor(ax4)
            plt.show()
            #show_args(ktrans="%0.2f"%result[0], ve="%0.2f"%result[1])
