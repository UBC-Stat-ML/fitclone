
# coding: utf-8

#     

# In[4]:

import os
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import *
#import matplotlib.ticker as mticker
import pandas as pn
import xarray as xr
import pickle


# ## Running R scripts from within Python for plotting and evaluation
# ```Python
# subprocess.call (["/usr/bin/Rscript", "--vanilla", "/pathto/MyrScript.r"])
# ```
# ```R
# options <- commandArgs(trailingOnly = TRUE)
# ```
# 
# Look at Pathos multiprocessing for Python
# https://pypi.python.org/pypi/multiprocess
# 
# 
# Use `cc = globals()[CLASS_NAME_STRING]` to instantiate an object from class name string.
# /gsc/software/linux-x86_64-centos6/python-3.5.2/bin/pip3 install --user scipy matplotlib ipython jupyter pandas
# 
# ## More code snippets
# 
# ### Remove last 10 jobs
#  qu | tail -10 | cut -d' ' -f1 | xargs qdel

# In[5]:

#  TimeSeriesDataUtilities 


# In[12]:

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        global time_length
        ts = time()
        result = f(*args, **kw)
        te = time()
        #print('func:{} args:[{}, {}] took: {} sec'.format(f.__name__, args, kw, te-ts))
        print('func:{} took: {} sec'.format(f.__name__, te-ts))
        time_length = te-ts
        return result
    return wrap


# In[ ]:

def list_from_string(some_str):
    res = some_str.split()
    try:
        return([int(i) for i in res])
    except ValueError:
        return([float(i) for i in res])
    return()


# In[1]:

def time_string_from_seconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return("%d:%02d:%02d." % (h, m, s))


# In[4]:

def quick_save(something, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(something, handle, protocol=pickle.HIGHEST_PROTOCOL)

def quick_load(file_path):
    with open(file_path, 'rb') as handle:
        return(pickle.load(handle))


# In[1]:

class TimeSeriesDataUtility:
    '''Read time series data in tall format, frmo '''
    def __init__(self, file_path):
        print('hello')
    
    def parse_meta_data(input_path):
        """
        Grabs the commented by # lines at the begining of the document
        and parases them as YAML
        """
        lines = []
        with open(input_path, "r") as in_lines:
            for line in in_lines:
                if line.startswith("#"):
                    lines.append(line)
                else:
                    break            

        lines = [x.replace('#', '') for x in lines] 
        return(yaml.load(''.join(lines)))
    
    def save_multidim_array(arr, dims, attr_name, output_path, use_index=True, use_header=True):
        x_ar = xr.DataArray(arr, dims=dims, name=attr_name)
        df = x_ar.to_dataframe()
        TimeSeriesDataUtility.save_time_series(df=df, output_path=output_path, use_index=use_index, use_header=use_header)
    
    def save_NTK(arr, output_path, use_index=True, use_header=True):
        TimeSeriesDataUtility.save_multidim_array(arr, dims=['n', 'time', 'K'] , attr_name='X', output_path=output_path, use_index=use_index, use_header=use_header)
    
    def save_time_series(df, output_path, use_index=True, use_header=True):
        df.to_csv('{}.gz'.format(output_path), sep='\t', mode='a', compression='gzip', index=use_index, header=use_header)
        
    def read_time_series(input_path):
        if not os.path.isfile(input_path):
            input_path += '.gz'
        dat = pn.read_csv(input_path, sep='\t', index_col=None, comment='#')
        return(dat)
    
    def count_to_percentage(dat):
        # At every time, sum of X should be 1
        time_points = dat.loc[:, 'time'].unique()
        for t in time_points:
            dat.loc[dat.time == t, 'X'] = dat.loc[:, 'X'][dat.time == t]/np.sum(dat.loc[:, 'X'][dat.time == t])
        return(dat)
    
    # from tallFormat to N*T*K ndarray
    def tall_to_TK(dat):
        #print(dat)
        time_points = np.array(dat.loc[:, 'time'].unique(), np.float128)
        T = len(time_points)
        K = len(dat.loc[:, 'K'].unique())
        print('Warning! NOT NORMALISING VALUES TO COUNTS...')
        #dat = TimeSeriesDataUtility.count_to_percentage(dat)
        tmp = np.empty([T, K])
        for index, t in  np.ndenumerate(time_points):
            #tmp[index[0], ] = dat[['X']][(dat.time == t)].values[:, 0]
            tmp[index[0], ] = dat[['X']][(dat.time == t)].values[:, 0]
            #tmp[index[0], ] = dat.loc[:, 'X'][dat.time == t].values[:, 0]
        
        return({'times':time_points, 'value':tmp})
    
    def list_TK_to_tall(dat_list, times=None, base_index=0):
        dfs = []
        for i in range(len(dat_list)):
            df = TimeSeriesDataUtility.TK_to_tall(dat_list[i], times)
            df['np'] = i + base_index
            dfs.append(df)        
        return(pn.concat(dfs))
            
        
    def TK_to_tall(dat, times=None):
        df = pn.DataFrame(data = dat, columns=range(0, dat.shape[1]), index=range(0, dat.shape[0]))
        if times is None:
            df['time'] = list(df.index)
        else:
            df['time'] = times[list(df.index)]
        # X K time
        return(pn.melt(df, id_vars=['time'], value_name='X', var_name='K').sort_values(by=['time', 'K']))
    
    
    def plot_TK(dat, title="", legend=None, xlabels=None):
        tall_dat = TimeSeriesDataUtility.TK_to_tall(dat, times=xlabels)
        TimeSeriesDataUtility.plot_tall(dat=tall_dat,title=title,legend=legend)
    
    def plot_tall(dat, title='', legend=None):
        columns = dat.loc[:, 'K'].unique()
        pdat = pn.DataFrame(data=TimeSeriesDataUtility.tall_to_TK(dat)['value'], columns=columns)
        pdat = pdat.set_index(np.round(dat.time.unique(), 5))
        fig, ax = subplots()
        if legend is not None:
            ax.legend(legend);
        pdat.plot(title=title, ax=ax)

    def plot_tall_old(dat, title='', legend=None):
        columns = dat.loc[:, 'K'].unique()
        pdat = pn.DataFrame(data=TimeSeriesDataUtility.tall_to_TK(dat)['value'], columns=columns)
        #%matplotlib inline
        fig, ax = subplots()
        xtickslocs = np.round(TimeSeriesDataUtility.tall_to_TK(dat)['times'], 5)
        #ax.set_xticks(np.linspace(start=0, stop=141, num=20))
        #ax.set_xticklabels(df.C, rotation=90)
        #plt.locator_params(axis='x', nticks=10)
#         df = pd.DataFrame({'A':26, 'B':20}, index=['N'])
        pdat.plot(title=title, ax=ax)
#         df.plot(kind='bar', ax=ax)
        if legend is not None:
            ax.legend(legend);
        nTicks = 20
        ticks = np.append(xtickslocs[0:-1:int(len(xtickslocs)/nTicks)], xtickslocs[-1])
        nTicks = len(ticks)
        ax.set_xticks(np.linspace(start=0, stop=len(xtickslocs), num=nTicks))    
        ax.set_xticklabels(ticks, rotation=90)    
    
    def get_sample_file():
        return("~/Google Drive/BCCRC/simulated_wf_passage/K14_2017-05-28-14-54-31.996607.tsv")


# In[2]:

class MCMC_loader():
    """
    mode is either 'predict' or 'infer'
    """
    def __init__(self, exp_path, is_predict=None):
        self._infer_x_path = os.path.join(exp_path, 'infer_x.tsv.gz')
        self._infer_theta_path = os.path.join(exp_path, 'infer_theta.tsv.gz')
        self._predict_path = os.path.join(exp_path, 'predict.tsv.gz')
        
        if os.path.isfile(self._infer_x_path) is False and os.path.isfile(self._predict_path) is False:
            return(None)
        
        if is_predict is None:
            is_predict = self._guess_mode()
        x_path = self._predict_path if is_predict == True else self._infer_x_path
        x = TimeSeriesDataUtility.read_time_series(x_path)
        self.last_iter = x.np.iloc[-1]
        # T by K
        T_values = x.time.unique()
        K_values = x.K.unique()
        self._x_mat = np.empty([len(T_values), len(K_values)])
        x = x[['time', 'K', 'X', 'np']]
        self._x_mat[:,:] = x[x.np == self.last_iter].pivot(index='time', columns='K', values='X').values

        if is_predict is False:
            theta = TimeSeriesDataUtility.read_time_series(self._infer_theta_path)
            print(theta)
            print(theta.shape)
            self._theta = theta.iloc[-1]
            # Sanity check
            if (self.last_iter+1) != (theta.shape[0]):
                raise ValueError('The theta(iter={}) and {}(iter={}) files are out of sync.'.format(theta.shape[0], 'predict' if is_predict else 'infer', self.last_iter+1))
    
    def get_last_infer_x(exp_path):
        x_path = os.path.join(exp_path, 'infer_x.tsv.gz')
        x = TimeSeriesDataUtility.read_time_series(x_path)
        last_iter = x.np.iloc[-1]
        # T by K
        T_values = x.time.unique()
        K_values = x.K.unique()
        x_mat = np.empty([len(T_values), len(K_values)])
        x = x[['time', 'K', 'X', 'np']]
        x_mat[:,:] = x[x.np==last_iter].pivot(index='time', columns='K', values='X').values
        return(x_mat)
    
    
    def get_last_infer_theta(exp_path):
        infer_theta_path = os.path.join(exp_path, 'infer_theta.tsv.gz')
        theta = TimeSeriesDataUtility.read_time_series(infer_theta_path)
        return(theta.iloc[-1])
    
    def get_all_theta(exp_path):
        return(TimeSeriesDataUtility.read_time_series(os.path.join(exp_path, 'infer_theta.tsv.gz')).values)
    
    def get_all_x(exp_path, is_predict=False):
        x_path = os.path.join(exp_path, 'predict.tsv.gz') if is_predict else os.path.join(exp_path, 'infer_x.tsv.gz')
        x = TimeSeriesDataUtility.read_time_series(x_path)
        last_iter = x.np.iloc[-1]
        # T by K
        T_values = x.time.unique()
        K_values = x.K.unique()
        x_mat = np.empty([last_iter+1, len(T_values), len(K_values)])
        x = x[['time', 'K', 'X', 'np']]
        for i in range(last_iter+1):
            x_mat[i, :,:] = x[x.np == i].pivot(index='time', columns='K', values='X').values
        return(x_mat)
    
    def _guess_mode(self):
        return(os.path.isfile(self._predict_path))
    
    def get_last_iter(self):
        return(self.last_iter)
    def get_x(self):
        return(self._x_mat)
    def get_theta(self):
        return(self._theta)
    def test():
        exp_path='/Users/sohrab/Google Drive/BCCRC/exp_OM489_201706-26-16226.085942_k10_1000particles'
        d = MCMC_loader(exp_path, False)
        x = d.get_x()
        s=d.get_theta()
        dd = d.get_last_iter()()


# 

# # On generating code for job submission
# Ues `#!/usr/bin/env python` at the beginning of the script.
# To import a script in another, one way is to use `execfile("/home/el/foo2/mylib.py")`.
# See other methods https://stackoverflow.com/questions/2349991/python-how-to-import-other-python-files.
# 
# See http://protips.maxmasnick.com/ipython-notebooks-automatically-export-py-and-html on how to export .py everytime you save.
# 
# ## Use command line
# `jupyter nbconvert --to script pgas-dir.ipynb`
# 
# ### How to remove empty cells?
# Look here https://github.com/jupyter/notebook/blob/master/docs/source/extending/savehooks.rst
# Read this and references therein: http://jupyter-notebook.readthedocs.io/en/latest/extending/frontend_extensions.html
# 
# ### How to remove comments?
# Look here https://github.com/jupyter/notebook/blob/master/docs/source/extending/savehooks.rst
# 
# A promising extension
# https://mindtrove.info/4-ways-to-extend-jupyter-notebook/#nb-extensions
# 
# 
# This will add another button:
# https://stackoverflow.com/questions/42991535/how-to-automatically-delete-all-blank-lines-in-a-jupyter-notebook
# 
# How to add an extension
# 
# 
# Manuplate a file direcltly:
# https://gist.github.com/damianavila/5305869
# 
# 
# Add another language dictionary
# https://chromium.googlesource.com/chromium/deps/hunspell_dictionaries/+/master
