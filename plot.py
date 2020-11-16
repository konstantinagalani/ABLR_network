import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Plot_():

    def __init__(self,T, inc_all_ABLR_simple,inc_all_ABLR_tr, inc_all_ABLR_cont_tr,inc_all_rand, inc_all_gp,transfer=True):
        self.inc_all_ABLR_simple = inc_all_ABLR_simple
        self.mean_all_ABLR_simple = np.mean(inc_all_ABLR_simple, axis=0)
        self.std_simple = np.std(inc_all_ABLR_simple, axis=0)
        
        self.inc_all_ABLR_tr = inc_all_ABLR_tr
        self.mean_all_ABLR_tr = np.mean(inc_all_ABLR_tr, axis=0)
        self.std_tr = np.std(inc_all_ABLR_tr, axis=0)

        self.inc_all_ABLR_cont_tr = inc_all_ABLR_cont_tr
        self.mean_all_ABLR_cont_tr = np.mean(inc_all_ABLR_cont_tr, axis=0)
        self.std_cont_tr = np.std(inc_all_ABLR_cont_tr, axis=0)
        
        self.inc_all_rand = inc_all_rand
        self.mean_all_rand = np.mean(inc_all_rand, axis=0)
        self.std_rand = np.std(inc_all_rand, axis=0)
        
        self.inc_all_gp = inc_all_gp
        self.mean_all_gp = np.mean(inc_all_gp, axis=0)
        self.std_gp = np.std(inc_all_gp, axis=0)
        self.transfer = transfer # True if transfer cases where applied


    def plot_simple(self,m=True):
        """
        Plot for branin and quadratic function
        For the function plot minimum (simple and log scale)
        """

        epochs = [i for i in range(50)]
        
        if m :
            self.inc_all_rand = self.mean_all_rand
            self.inc_all_gp = self.mean_all_gp
            self.inc_all_ABLR_simple = self.mean_all_ABLR_simple
            self.inc_all_ABLR_tr = self.mean_all_ABLR_tr
            self.inc_all_ABLR_cont_tr = self.mean_all_ABLR_cont_tr


        f_log = plt.figure()
        ax1 = f_log.add_subplot(1, 1, 1)
        ax1.plot(epochs, np.log10(self.inc_all_rand), label='Random')
        ax1.fill_between(epochs, np.log10(self.inc_all_rand) - np.log10(self.std_rand), np.log10(self.inc_all_rand) + np.log10(self.std_rand), alpha=0.3)
        ax1.plot(epochs, np.log10(self.inc_all_gp), label='GP Plain')
        ax1.fill_between(epochs, np.log10(self.inc_all_gp) - np.log10(self.std_gp), np.log10(self.inc_all_gp) + np.log10(self.std_gp), alpha=0.3)
        ax1.plot(epochs, np.log10(self.inc_all_ABLR_simple), label='ABLR')
        ax1.fill_between(epochs, np.log10(self.inc_all_ABLR_simple) - np.log10(self.std_simple), np.log10(self.inc_all_ABLR_simple) + np.log10(self.std_simple), alpha=0.3)

        f_simple  = plt.figure()
        ax2 = f_simple.add_subplot(1, 1, 1)
        ax2.plot(epochs, self.inc_all_rand, label='Random')
        ax2.fill_between(epochs, self.inc_all_rand - self.std_rand, self.inc_all_rand + self.std_rand, alpha=0.3)
        ax2.plot(epochs, self.inc_all_gp, label='GP Plain')
        ax2.fill_between(epochs, self.inc_all_gp - self.std_gp, self.inc_all_gp + self.std_gp, alpha=0.3)
        ax2.plot(epochs, self.inc_all_ABLR_simple, label='ABLR')
        ax2.fill_between(epochs, self.inc_all_ABLR_simple - self.std_simple, self.inc_all_ABLR_simple + self.std_simple, alpha=0.3)

        if self.transfer:
         
            ax1.plot(epochs, np.log10(self.inc_all_ABLR_tr) , label='ABLR transfer')
            ax1.fill_between(epochs, np.log10(self.inc_all_ABLR_tr) - np.log10(self.std_tr),  np.log10(self.inc_all_ABLR_tr) + np.log10(self.std_tr), alpha=0.3)
            ax1.plot(epochs, np.log10(self.inc_all_ABLR_cont_tr) , label='ABLR transfer context')
            ax1.fill_between(epochs, np.log10(self.inc_all_ABLR_cont_tr) - np.log10(self.std_cont_tr), np.log10(self.inc_all_ABLR_cont_tr) + np.log10(self.std_cont_tr), alpha=0.3)

            ax2.plot(epochs, self.inc_all_ABLR_tr , label='ABLR transfer')
            ax2.fill_between(epochs, self.inc_all_ABLR_tr - self.std_tr, self.inc_all_ABLR_tr + self.std_tr, alpha=0.3)
            ax2.plot(epochs, self.inc_all_ABLR_cont_tr , label='ABLR transfer context')
            ax2.fill_between(epochs,self.inc_all_ABLR_cont_tr - self.std_cont_tr,self.inc_all_ABLR_cont_tr + self.std_cont_tr, alpha=0.3)

        ax2.legend(loc='upper right')
        if m : ax2.set_title("Minimum of function")
        else :    ax2.set_title("ADTM")
        
        ax1.legend(loc='upper right')
        if m : ax1.set_title("Minimum of function log10")
        else :    ax1.set_title("ADTM log10")
        #plt.show()
        f_log.savefig(str(datetime.now()) + ".png")
        f_simple.savefig(str(datetime.now()) + ".png")

    def plot_adtm(self,ymin,ymax):
        """
        For the dataset plot Average Distance to Minimum (simple and log scale)
        """
        ymin = np.tile(ymin,50)
        ymax = np.tile(ymax,50)
        adtm_rand = (self.inc_all_rand - ymin)/(ymax-ymin)
        self.inc_all_rand = np.mean(adtm_rand,axis=0)
        self.std_rand = np.std(adtm_rand,axis=0)
        
        adtm_gp = (self.inc_all_gp - ymin)/(ymax-ymin)
        self.inc_all_gp = np.mean(adtm_gp,axis=0)
        self.std_gp = np.std(adtm_gp,axis=0)

        adtm_ABLR_simple = (self.inc_all_ABLR_simple - ymin)/(ymax-ymin)
        self.inc_all_ABLR_simple = np.mean(adtm_ABLR_simple,axis=0)
        self.std_simple = np.std(adtm_ABLR_simple,axis=0)

        adtm_ABLR_tr = (self.inc_all_ABLR_tr - ymin)/(ymax-ymin)
        self.inc_all_ABLR_tr = np.mean(adtm_ABLR_tr,axis=0)
        self.std_tr = np.std(adtm_ABLR_tr,axis=0)

        adtm_ABLR_cont_tr = (self.inc_all_ABLR_cont_tr - ymin)/(ymax-ymin)
        self.inc_all_ABLR_cont_tr = np.mean(adtm_ABLR_cont_tr,axis=0)
        self.std_ABLR_cont_tr = np.std(adtm_ABLR_cont_tr,axis=0)

        self.plot_simple(m=False)