#Python 3.9

import tkinter
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import curve_fit
import glob,os
from sympy import *
from sympy.abc import x,a,b,c,d,e,f,g,h


class View(object):
    """Test docstring. """
    def __init__(self, root):
        self.fig = Figure()
        self.ax = self.fig.add_subplot()
        self.ax.set_xlabel("x", fontsize=14)
        self.ax.set_ylabel("y", fontsize=14)
        self.ax.tick_params(direction='out', length=5, width=1)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)  # A tk.DrawingArea.
        self.canvas.draw()
        x_example=np.linspace(1, 12, num=100)
        y_example=np.sin(x_example)*10+15
        self.xydata= np.column_stack((x_example,y_example))
        self.x_fitting=np.zeros(5)
        self.y_fitting=np.zeros(5)
        self.header=''
        toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        toolbar.update()
        
        frame = tkinter.Frame(master=root)
        ii=0
        tkinter.Label(master=frame,text='Select data .txt file (x,y):',font=('', 12)).grid(row=ii,column=0)
        self.unitlist = ['Example']
        for j in glob.glob('*.txt'):
            self.unitlist.append(j.split('\\')[-1])
        self.combobox = ttk.Combobox(master=frame, width=25)
        self.combobox['values'] = self.unitlist
        self.combobox.current(0)
        self.combobox.grid(row=ii,column=1)
        self.combobox.bind("<<ComboboxSelected>>",lambda e: self.selectfile())

        self.checklog = tkinter.IntVar()
        ttk.Checkbutton(master=frame,text="LogScale",variable=self.checklog,onvalue=1, offvalue=0).grid(row=ii, column=2)
        self.checklog.set(0)
        
        ii+=1
        tkinter.Label(master=frame,text='Equation for fitting f(x) =',font=('', 12)).grid(row=ii,column=0)
        self.models = tkinter.StringVar()
        ttk.Entry(master=frame, textvariable=self.models).grid(row=ii, column=1)
        self.models.set('a*sin(x)+b')
        tkinter.Button(master=frame, text="Least Squares Regression", width=20, command=self.datafit).grid(row=ii,column=2)

        ii+=1
        tkinter.Label(master=frame,text='Math: +-*/,10^x,exp(x),ln(x),log(x),sin(x)...Coefficients:a,b,c,d,e,f,g,h',font=( '', 12) ).grid(row=ii,column=0,columnspan=3)

    
        ii+=1
        tkinter.Button(master=frame, text="Save Fitting Curve", width=20, command=self.clipboard).grid(row=ii,column=1)

        frame.pack(side=tkinter.BOTTOM)
        
        toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.plotxy()

    def clipboard(self):
        if self.combobox.get() == 'Example':
            return
        np.savetxt('fit_%s'%self.combobox.get(), np.column_stack((self.x_fitting,self.y_fitting)),header=self.header)
        print('fit_%s saved.'%self.combobox.get())
        
    def selectfile(self):
        filename=self.combobox.get()
        self.header=''
        self.x_fitting=np.zeros(5)
        self.y_fitting=np.zeros(5)
        if filename =='Example':
            x_example=np.linspace(1, 12, num=100)
            y_example=np.sin(x_example)*10+15
            self.xydata= np.column_stack((x_example,y_example))
            self.models.set('a*sin(x)+b')
        else:
            self.xydata= np.loadtxt(filename)
        self.plotxy()

    def plotxy(self):
        self.ax.clear()
        self.ax.scatter(self.xydata[:,0],self.xydata[:,1],s=50,facecolors='none', edgecolors='black',label='Original data')
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend([handles[0]],[labels[0]],loc='best',edgecolor='white',fontsize=12)
        self.ax.set_xlabel("x", fontsize=14)
        self.ax.set_ylabel("y", fontsize=14)
        if self.checklog.get()==1:
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        self.canvas.draw()

    def datafit(self):
        if len(self.xydata)==0:
            return
        
        symboldic={'a':a,'b':b,'c':c,'d':d,'e':e,'f':f,'g':g,'h':h}
        
        coefficients=self.models.get()
        for i in '+-*/()^':
            coefficients=coefficients.replace(i, ' ')

        my_tuple = (x,)
        pars=''
        for i in sorted(set(coefficients.split(' '))):
            if i !='' and i in 'abcdefgh':
                my_tuple += (symboldic[i],)
                pars+=i
                
        fun=lambdify(my_tuple,self.models.get(), 'numpy')
        popt, pcov = curve_fit(fun, self.xydata[:,0], self.xydata[:,1], bounds=(0,10000),method='trf')

        #----------R_squared--------------
        y_fitting=fun(self.xydata[:,0],*popt)
        residuals=(self.xydata[:,1]-y_fitting)**2
        ss_res=np.sum(residuals)
        ss_total=np.sum((self.xydata[:,1]-np.mean(self.xydata[:,1]))**2)
        R_squared=1-(ss_res/ss_total)

        self.header=''
        print('\n-------\n')
        print('y = %s '%self.models.get())
        self.header+='y = %s '%self.models.get()
        for i,j in zip(list(pars),popt):
            print('%s: %.4f '%(i,j))
            self.header+='%s: %.4f '%(i,j)
        print('R^2=%.4f'%(R_squared))
        self.header+='R^2=%.4f'%(R_squared)
        print('\n-------\n')

        # ---------plot fitting curve------------
        n=200
        self.x_fitting=np.linspace(np.min(self.xydata[:,0]), np.max(self.xydata[:,0]), num=n)
        self.y_fitting=fun(self.x_fitting,*popt)
                
        self.ax.clear()
        self.ax.scatter(self.xydata[:,0],self.xydata[:,1],s=50,facecolors='none', edgecolors='black',label='Original data')
        self.ax.plot(self.x_fitting, self.y_fitting,'r',label='Data Fitting')
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend([handles[1],handles[0]],[labels[1],labels[0]],loc='best',edgecolor='white',fontsize=12)
        self.ax.set_xlabel("x", fontsize=14)
        self.ax.set_ylabel("y", fontsize=14)
        if self.checklog.get()==1:
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        self.canvas.draw()

def main():
    root = tkinter.Tk()
    root.title("Data Fitting GUI")
    view = View(root)
    tkinter.mainloop()

if __name__ == "__main__":
    main()
