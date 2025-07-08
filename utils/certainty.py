import numpy as np
import seaborn as sns
def uncertainty(values):
        if isinstance(values,list):
                values=np.asarray(values)
        return np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))


