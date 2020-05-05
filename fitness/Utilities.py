import os
import pandas as pn
import xarray as xr
import pickle


from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        global time_length
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:{} took: {} sec'.format(f.__name__, te-ts))
        time_length = te-ts
        return result
    return wrap



def list_from_string(some_str):
    res = some_str.split()
    try:
        return([int(i) for i in res])
    except ValueError:
        return([float(i) for i in res])
    return()

def time_string_from_seconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return("%d:%02d:%02d." % (h, m, s))

def quick_save(something, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(something, handle, protocol=pickle.HIGHEST_PROTOCOL)

def quick_load(file_path):
    with open(file_path, 'rb') as handle:
        return(pickle.load(handle))

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
        time_points = np.array(dat.loc[:, 'time'].unique(), np.float128)
        T = len(time_points)
        K = len(dat.loc[:, 'K'].unique())
        print('Warning! NOT NORMALISING VALUES TO COUNTS...')
        tmp = np.empty([T, K])
        for index, t in  np.ndenumerate(time_points):
            tmp[index[0], ] = dat[['X']][(dat.time == t)].values[:, 0]
        
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
        pdat.plot(title=title, ax=ax)
        if legend is not None:
            ax.legend(legend);
        nTicks = 20
        ticks = np.append(xtickslocs[0:-1:int(len(xtickslocs)/nTicks)], xtickslocs[-1])
        nTicks = len(ticks)
        ax.set_xticks(np.linspace(start=0, stop=len(xtickslocs), num=nTicks))    
        ax.set_xticklabels(ticks, rotation=90)    
    
    def get_sample_file():
        return("~/Google Drive/BCCRC/simulated_wf_passage/K14_2017-05-28-14-54-31.996607.tsv")

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

