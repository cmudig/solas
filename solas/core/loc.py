import pandas as pd
from solas.history.history import History
from solas.core.series import SolasSeries
from solas.core.frame import SolasDataFrame
from solas.utils.utils import convert_indices_to_columns, convert_names_to_columns
from typing import Tuple
class SolasLocIndexer(pd.core.indexing._LocIndexer):

    _metadata = ["_parent_df"]

    def __init__(self, *args, **kwargs):
        super(SolasLocIndexer, self).__init__(*args, **kwargs)
        self._parent_df = None
    
    def __getitem__(self, key):
        '''
        This function is called for example df.loc[row, col] or df.loc[row]
        where both the row and column index could be 
        a row index/column name, a list, and a slice object.
        Possible examples are listed as follows:
            df.loc["cobra", "shield"], df.loc[["cobra"], ["shield"]], 
            df.loc[["cobra"], :], df.loc[["cobra"],], 
            df.loc[:, ["shield"]], df.loc[:,:]

        Parameters:
            key: either a tuple or a str/int
            
        Returns:
            Depending on whether a single column, a single value, or a dataframe is referenced, 
            the returned value would be correspondingly a series, a singe value, and a dataframe
        '''
        if self._parent_df is not None:
            self._parent_df.history.freeze()
            ret_value = super(SolasLocIndexer, self).__getitem__(key)
            self._parent_df.history.unfreeze()

            columns = convert_names_to_columns(self._parent_df.columns, key) if type(key) is tuple else []
            if (isinstance(ret_value, SolasSeries) or isinstance(ret_value, SolasDataFrame)) and (id(ret_value) != id(self._parent_df)):
                # If it returned a series or dataframe object, 
                # and it is not the same as the parent dataframe (which is possible when df.loc[:,:] is called)
                # then copy all metadata log action to the child dataframe
                ret_value = self._solas_copymd(ret_value)
                ret_value._parent_df = self._parent_df
                ret_value.history.append_event("loc", columns, rank_type="child", child_df=None)
            self._parent_df.history.append_event("loc", columns, rank_type="parent", child_df=ret_value)
        else:
            ret_value = super(SolasLocIndexer, self).__getitem__(key)
        return ret_value
    
    def __setitem__(self, key, value):
        '''
        This function is called for example df.loc[row, col] = value or df.loc[row] = value 
        where both the row and column index could be 
        a row index/coluumn name, a list, and a slice object.
        Possible examples are listed as follows:
            df.loc["cobra", "shield"], df.loc[["cobra"], ["shield"]], 
            df.loc[["cobra"], :], df.loc[["cobra"],], 
            df.loc[:, ["shield"]], df.loc[:,:]

        Parameters:
            key: either a tuple or a str/int
            
        Returns:
            None
        '''
        if self._parent_df is not None:
            self._parent_df.history.freeze()
            super(SolasLocIndexer, self).__setitem__(key, value)
            self._parent_df.history.unfreeze()
            columns = convert_names_to_columns(self._parent_df.columns, key) if type(key) is tuple else []
            if columns is not None: 
                # if the key[1] is multi-index instead of list, str, slice, we choose to not log such action for now.
                self._parent_df.history.append_event("loc", columns, rank_type="parent", child_df=None)
        else:
            super(SolasLocIndexer, self).__setitem__(key, value)
    
    def _solas_copymd(self, ret_value):
        for attr in self._metadata:
            ret_value.__dict__[attr] = getattr(self._parent_df, attr, None)
        if hasattr(self._parent_df, "_history"):
            ret_value.history = getattr(self._parent_df, "_history").copy()
        return ret_value

class iSolasLocIndexer(pd.core.indexing._iLocIndexer):

    _metadata = ["_parent_df"]

    def __init__(self, *args, **kwargs):
        super(iSolasLocIndexer, self).__init__(*args, **kwargs)
        self._parent_df = None
   
    def __getitem__(self, key):
        '''
        This function is called for example df.iloc[row, col] or df.iloc[row]
        where both the row and column index could be 
        an integer, a list of integers, and a slice object of integers.
        Possible examples are listed as follows:
            df.iloc[1, 2], df.iloc[[1,2], [1,2]], 
            df.iloc[[1,2], :], df.iloc[1,], 
            df.iloc[:, [1]], df.iloc[:,:]

        Parameters:
            key: either a tuple or a str/int
            
        Returns:
            Depending on whether a single column, a single value, or a dataframe is referenced, 
            the returned value would be correspondingly a series, a singe value, and a dataframe
        '''
        if self._parent_df is not None:
            self._parent_df.history.freeze()
            ret_value = super(iSolasLocIndexer, self).__getitem__(key)
            self._parent_df.history.unfreeze()
            
            columns = convert_indices_to_columns(self._parent_df.columns, key) if type(key) is tuple else []
            if (isinstance(ret_value, SolasSeries) or isinstance(ret_value, SolasDataFrame)) and (id(ret_value) != id(self._parent_df)):
                ret_value = self._solas_copymd(ret_value)
                ret_value._parent_df = self._parent_df
                ret_value.history.append_event("iloc", columns, rank_type="child", child_df=None)
            self._parent_df.history.append_event("iloc", columns, rank_type="parent", child_df=ret_value)
        else:
            ret_value = super(iSolasLocIndexer, self).__getitem__(key)
        return ret_value

    
    def __setitem__(self, key, value):
        '''
        This function is called for example df.iloc[row, col] = value or df.iloc[row] = value
        where both the row and column index could be 
        an integer, a list of integers, and a slice object of integers.
        Possible examples are listed as follows:
            df.iloc[1, 2], df.iloc[[1,2], [1,2]], 
            df.iloc[[1,2], :], df.iloc[1,], 
            df.iloc[:, [1]], df.iloc[:,:]

        Parameters:
            key: either a tuple or a str/int
            
        Returns:
            None
        '''
        if self._parent_df is not None:
            self._parent_df.history.freeze()
            super(iSolasLocIndexer, self).__setitem__(key, value)
            self._parent_df.history.unfreeze()

            columns = convert_indices_to_columns(self._parent_df.columns, key) if type(key) is tuple else []
            if columns is not None: 
                # if the key[1] is multi-index instead of list, int, slice, we choose to not log such action for now.
                self._parent_df.history.append_event("iloc", columns, rank_type="parent", child_df=None)
        else:
            super(iSolasLocIndexer, self).__setitem__(key, value)

    def _solas_copymd(self, ret_value):
        for attr in self._metadata:
            ret_value.__dict__[attr] = getattr(self._parent_df, attr, None)
        if hasattr(self._parent_df, "_history"):
            ret_value.history = getattr(self._parent_df, "_history").copy()
        return ret_value
