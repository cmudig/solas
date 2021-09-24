#  Copyright 2019-2020 The Lux Authors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pandas as pd
import lux
import warnings
import traceback
import numpy as np
from lux.history.history import History
from lux.utils.message import Message
from lux.implicit.utils import rename_from_history


from pandas._typing import (
    FrameOrSeries,
    ArrayLike,
)
from typing import Optional, Tuple, Union, Hashable, Dict, Union, List, Callable
from lux.vis.VisList import VisList

from IPython.core.debugger import set_trace


class LuxSeries(pd.Series):
    """
    A subclass of pd.Series that supports all 1-D Series operations
    """

    _metadata = [
        "_intent",
        "_inferred_intent",
        "_data_type",
        "unique_values",
        "cardinality",
        "_rec_info",
        "_min_max",
        "plotting_style",
        "_current_vis",
        "_widget",
        "_recommendation",
        "_prev",
        "_history",
        "_saved_export",
        "name",
        "_sampled",
        "_toggle_pandas_display",
        "_message",
        "_pandas_only",
        "pre_aggregated",
        "_type_override",
    ]

    def __init__(self, *args, **kw):
        super(LuxSeries, self).__init__(*args, **kw)

        # defaults
        self._intent = []
        self._inferred_intent = []
        self._current_vis = []
        self._recommendation = []
        self._toggle_pandas_display = True
        self._pandas_only = False
        self._type_override = {}
        self._history = History(self)
        self._message = Message()

        # others
        self._data_type = {}
        self.unique_values = None
        self.cardinality = None
        self._rec_info = None
        self._min_max = None
        self.plotting_style = None
        self._widget = None
        self._prev = None
        self._saved_export = None
        self._sampled = None
        self.pre_aggregated = None
        self._parent_df = None  # if series comes from a df this will be populated with ref to df

    @property
    def _constructor(self):
        return LuxSeries

    @property
    def _constructor_expanddim(self):
        from lux.core.frame import LuxDataFrame

        def f(*args, **kwargs):
            df = LuxDataFrame(*args, **kwargs)
            for attr in self._metadata:
                # if attr in self._default_metadata:
                #     default = self._default_metadata[attr]
                # else:
                #     default = None
                df.__dict__[attr] = getattr(self, attr, None)
            return df

        f._get_axis_number = LuxDataFrame._get_axis_number
        return f

    @property
    def history(self):
        return self._history

    @property
    def data_type(self):
        return self._data_type

    @history.setter
    def history(self, history: History):
        self._history = history

    def to_pandas(self) -> pd.Series:
        """
        Convert Lux Series to Pandas Series

        Returns
        -------
        pd.Series
        """
        import lux.core

        return lux.core.originalSeries(self, copy=False)

    def set_data_type(self, types: dict):
        """
        Set the data type for this series
        overriding the automatically-detected type inferred by Lux
        which will happen in the _ipython_display when the series is converted to a dataframe
        thanks to the finalize method, its _type_override will be copied to the dataframe then.

        Parameters
        ----------
        types: dict
            Dictionary that maps the name of this series to a specified Lux Type.
            If the name is None, the convention is to use "Unnamed"
            Possible options: "nominal", "quantitative", "id", and "temporal".

        Example
        ----------
        df = pd.read_csv("https://raw.githubusercontent.com/lux-org/lux-datasets/master/data/absenteeism.csv")
        df.set_data_type({"ID":"id",
                          "Reason for absence":"nominal"})
        """
        if self._type_override == None:
            self._type_override = types
        else:
            self._type_override = {**self._type_override, **types}

        if not self.data_type:
            self._data_type = {}

        for attr in types:
            if types[attr] not in ["nominal", "quantitative", "id", "temporal"]:
                raise ValueError(
                    f'Invalid data type option specified for {attr}. Please use one of the following supported types: ["nominal", "quantitative", "id", "temporal"]'
                )
            self._data_type[attr] = types[attr]

    def _ipython_display_(self):
        from IPython.display import display
        from IPython.display import clear_output
        import ipywidgets as widgets
        from lux.core.frame import LuxDataFrame

        series_repr = super(LuxSeries, self).__repr__()

        # Default column name 0 causes errors
        if self.name is None:
            self.name = "Unnamed"

        child_df = None
        ## explanation for the `not self.pre_aggregated` condition
        # for series, if its parent is a dataframe, then it is very likely to come from a column reference of it
        # In such cases, we intend to show the visualization of the parent dataframe,
        # and select charts that are related to the attribute of this series.
        # The exception is that for df.std() etc function, the returned value satisfies other conditions
        # Even if we could show the parent dataframe, the attribute is not available.
        # and what the user want is actually the column group graph.

        ## explanation for the `mre_op_name != "iloc" and mre_op_name != "loc"` condition
        # in this case, the function call will be like newdf = df.loc[100:200,"Origin"] so that it will satisfy other conditions.
        # it involves the filter, and including the charts for the parent dataframe (df in this case) will be confusing
        # are charts about the dataframe before or after the visualization?
        # therefore, we choose not to including the charts for the parent dataframe in this case
        most_recent_event, hist_index = self.history.get_mre([self.name])
        most_recent_op = most_recent_event.op_name if most_recent_event is not None else None
        if (
            self._parent_df is not None
            and isinstance(self._parent_df, LuxDataFrame)
            and not self.pre_aggregated
            and most_recent_op != "iloc"
            and most_recent_op != "loc"
        ):
            ldf = self._parent_df
            ldf._parent_df = None  # se do we need information about the grandparent?
            child_df = LuxDataFrame(self)
            child_df._parent_df = self._parent_df
            # this line is necessary otherwise because of the `_finalize_` logic, this self will be recognized as the parent_df
            # which will cause a problem when we draw the implicit plotter like filter charts where the parent dataframe will be used
            # (remember we use the child information to draw it)
        else:
            ldf = LuxDataFrame(self)
            ldf._parent_df = (
                self._parent_df
            )  # tbd if this is good or bad, dont think I ever need the series itself
        self._ldf = ldf

        try:
            # Ignore recommendations when Series a results of:
            # 1) Values of the series are of dtype objects (df.dtypes)
            is_dtype_series = (
                all(isinstance(val, np.dtype) for val in self.values) and len(self.values) != 0
            )
            # 2) Mixed type, often a result of a "row" acting as a series (df.iterrows, df.iloc[0])
            # Tolerant for NaNs + 1 type
            mixed_dtype = len(set([type(val) for val in self.values])) > 2
            if ldf._pandas_only or is_dtype_series or (mixed_dtype and not ldf.pre_aggregated):
                # if the series is pre-aggregated, then we allow the visualization even if the series consist of mixed types.
                # for example, df["Origin"].describe().
                # this also works in other cases since we have generally provided visualizations for the aggregated series
                # in either column_group or implicit tab.
                print(series_repr)
                ldf._pandas_only = False
            else:
                if not self.index.nlevels >= 2:
                    ldf.maintain_metadata()

                if lux.config.default_display == "lux":
                    self._toggle_pandas_display = False
                else:
                    self._toggle_pandas_display = True

                # df_to_display.maintain_recs() # compute the recommendations (TODO: This can be rendered in another thread in the background to populate self._widget)
                ldf.maintain_recs(is_series="Series", child=child_df)

                # Observers(callback_function, listen_to_this_variable)
                ldf._widget.observe(ldf.remove_deleted_recs, names="deletedIndices")
                ldf._widget.observe(ldf.set_intent_on_click, names="selectedIntentIndex")

                self._widget = ldf._widget
                self._recommendation = ldf._recommendation

                # box = widgets.Box(layout=widgets.Layout(display='inline'))
                button = widgets.Button(
                    description="Toggle Pandas/Lux",
                    layout=widgets.Layout(width="140px", top="5px"),
                )
                ldf.output = widgets.Output()
                # box.children = [button,output]
                # output.children = [button]
                # display(box)
                display(button, ldf.output)

                def on_button_clicked(b):
                    with ldf.output:
                        if b:
                            self._toggle_pandas_display = not self._toggle_pandas_display
                        clear_output()
                        if self._toggle_pandas_display:
                            print(series_repr)
                        else:
                            # b.layout.display = "none"
                            display(ldf._widget)
                            # b.layout.display = "inline-block"

                button.on_click(on_button_clicked)
                on_button_clicked(None)

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            warnings.warn(
                "\nUnexpected error in rendering Lux widget and recommendations. "
                "Falling back to Pandas display.\n"
                "Please report the following issue on Github: https://github.com/lux-org/lux/issues \n",
                stacklevel=2,
            )
            warnings.warn(traceback.format_exc())
            display(self.to_pandas())

    @property
    def recommendation(self):

        if self._recommendation is not None and self._recommendation == {}:
            if self.name is None:
                self.name = " "
            ldf = LuxDataFrame(self)

            ldf.maintain_metadata()
            ldf.maintain_recs()
            self._recommendation = ldf._recommendation
        return self._recommendation

    @property
    def exported(self) -> Union[Dict[str, VisList], VisList]:
        """
        Get selected visualizations as exported Vis List

        Notes
        -----
        Convert the _selectedVisIdxs dictionary into a programmable VisList
        Example _selectedVisIdxs :

            {'Correlation': [0, 2], 'Occurrence': [1]}

        indicating the 0th and 2nd vis from the `Correlation` tab is selected, and the 1st vis from the `Occurrence` tab is selected.

        Returns
        -------
        Union[Dict[str,VisList], VisList]
                When there are no exported vis, return empty list -> []
                When all the exported vis is from the same tab, return a VisList of selected visualizations. -> VisList(v1, v2...)
                When the exported vis is from the different tabs, return a dictionary with the action name as key and selected visualizations in the VisList. -> {"Enhance": VisList(v1, v2...), "Filter": VisList(v5, v7...), ..}
        """
        return self._ldf.exported

    #####################
    ## Override Pandas ##
    #####################
    def groupby(self, *args, **kwargs):
        history_flag = False
        if "history" not in kwargs or ("history" in kwargs and kwargs["history"]):
            history_flag = True
        if "history" in kwargs:
            del kwargs["history"]
        if self.history is not None:
            self.history.freeze()
        groupby_obj = super(LuxSeries, self).groupby(*args, **kwargs)
        if self.history is not None:
            self.history.unfreeze()
        for attr in self._metadata:
            groupby_obj.__dict__[attr] = getattr(self, attr, None)
        if history_flag:
            groupby_obj._history = groupby_obj._history.copy()
            groupby_obj._history.append_event("groupby", [], *args, **kwargs)
        groupby_obj.pre_aggregated = True
        return groupby_obj

    def __finalize__(
        self: FrameOrSeries, other, method: Optional[str] = None, **kwargs
    ) -> FrameOrSeries:
        """
        See same method in frame.py
        """
        _this = super(LuxSeries, self).__finalize__(other, method, **kwargs)
        if _this._history is not None:
            _this._history = _this._history.copy()
        return _this

    """
    NOTE: df.Origin 's history is a little off right now.
    Since a column is put into a cache on the df it only copies over the history
    from df when the column is first referenced. If this history has to do with "Origin"
    this it will all be logged since the same object is left in the cache but a series history
    might be inconsistent with it's parent df so 

    `df.Origin.history` will not necessarily contain all of `df.history`
    The fix is to catch this the first time a column is pulled into the cache and either clear the history or 
    something else
    """

    def _log_events(self, op_name, ret_value):
        # add to history
        name = "Unnamed" if self.name is None else self.name
        self._history.append_event(op_name, [name])  # df.col
        if ret_value.history.check_event(-1, op_name="col_ref", cols=[name]):
            ret_value.history.edit_event(-1, op_name, [name], rank_type="child")
        else:
            ret_value.history.append_event(op_name, [name], rank_type="child")
        ## otherwise, there are two logs, one for col_ref, the othere for value_counts
        ## because it directly copies the history of the parent dataframe
        self.add_to_parent_history(op_name, [name])  # df

    def value_counts(self, *args, **kwargs):
        ret_value = super(LuxSeries, self).value_counts(*args, **kwargs)

        ret_value._parent_df = self
        # no need to use LuxDataFrame({name: self}) since the _parent_df won't be used in plotting implicit_tab
        ret_value._history = self._history.copy()  # self._parent_df._history.copy()
        # is there a need to copy from the history of the grandparent?
        ret_value.pre_aggregated = True

        # add to history
        self._log_events("value_counts", ret_value)
        return ret_value

    def describe(self, *args, **kwargs):
        with self.history.pause():
            if self._parent_df is not None:
                self._parent_df.history.freeze()
            ret_value = super(LuxSeries, self).describe(*args, **kwargs)
            if self._parent_df is not None:
                self._parent_df.history.unfreeze()

        from lux.core.frame import LuxDataFrame

        name = "Unnamed" if self.name is None else self.name
        ret_value._parent_df = LuxDataFrame({name: self})
        # this is different from the part in value_counts, simply to faciliate the visualization.
        # sinc in the boxplot, only this serires is needed.
        # it is ok to set ret_value._parent_df = self._parent_df but it will include other columns as well
        # and this approach could not handle the case when the self._parent_df is not avaiable.
        ret_value._history = (
            self._history.copy()
        )  # seems no need to inherit the history of the grandparent.
        ret_value.pre_aggregated = True

        # add to history
        self._log_events("describe", ret_value)
        return ret_value

    def isna(self, *args, **kwargs):
        with self._history.pause():
            ret_value = super(LuxSeries, self).isna(*args, **kwargs)

        from lux.core.frame import LuxDataFrame

        ret_value._parent_df = self
        # no need to use LuxDataFrame({name: self}) since the _parent_df won't be used in plotting implicit_tab
        # this is different from the part in describe, simply to faciliate the visualization.
        ret_value._history = (
            self._history.copy()
        )  # seems no need to inherit the history of the grandparent.

        name = "Unnamed" if self.name is None else self.name
        # manually set the data type to avoid mistakes like identifying the "Year" as temporal
        # see set_data_type for a more detailed explanation why this works for the series
        # ret_value.set_data_type({name: "nominal"})
        # add to history
        self._log_events("isna", ret_value)
        return ret_value

    def isnull(self, *args, **kwargs):
        with self._history.pause():
            ret_value = super(LuxSeries, self).isnull(*args, **kwargs)

        from lux.core.frame import LuxDataFrame

        ret_value._parent_df = self
        # no need to use LuxDataFrame({name: self}) since the _parent_df won't be used in plotting implicit_tab
        # this is different from the part in describe, simply to faciliate the visualization.
        ret_value._history = (
            self._history.copy()
        )  # seems no need to inherit the history of the grandparent.

        name = "Unnamed" if self.name is None else self.name
        # manually set the data type to avoid mistakes like identifying the "Year" as temporal
        # see set_data_type for a more detailed explanation why this works for the series
        # ret_value.set_data_type({name: "nominal"})
        # add to history
        self._log_events("isna", ret_value)
        return ret_value

    def notnull(self, *args, **kwargs):
        with self._history.pause():
            ret_value = super(LuxSeries, self).notnull(*args, **kwargs)

        from lux.core.frame import LuxDataFrame

        ret_value._parent_df = self
        # no need to use LuxDataFrame({name: self}) since the _parent_df won't be used in plotting implicit_tab
        # this is different from the part in describe, simply to faciliate the visualization.
        ret_value._history = (
            self._history.copy()
        )  # seems no need to inherit the history of the grandparent.

        name = "Unnamed" if self.name is None else self.name
        # manually set the data type to avoid mistakes like identifying the "Year" as temporal
        # see set_data_type for a more detailed explanation why this works for the series
        # ret_value.set_data_type({name: "nominal"})

        # add to history
        self._log_events("notnull", ret_value)
        return ret_value

    def notna(self, *args, **kwargs):
        with self._history.pause():
            ret_value = super(LuxSeries, self).notna(*args, **kwargs)

        from lux.core.frame import LuxDataFrame

        ret_value._parent_df = self
        # no need to use LuxDataFrame({name: self}) since the _parent_df won't be used in plotting implicit_tab
        # this is different from the part in describe, simply to faciliate the visualization.
        ret_value._history = (
            self._history.copy()
        )  # seems no need to inherit the history of the grandparent.

        name = "Unnamed" if self.name is None else self.name
        # manually set the data type to avoid mistakes like identifying the "Year" as temporal
        # see set_data_type for a more detailed explanation why this works for the series
        # ret_value.set_data_type({name: "nominal"})

        # add to history
        self._log_events("notnull", ret_value)
        return ret_value

    def unique(self, *args, **kwargs):
        """
        Overridden method for pd.Series.unique with cached results.
        Return unique values of Series object.
        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.
        Returns
        -------
        ndarray or ExtensionArray
            The unique values returned as a NumPy array.
        See Also
        --------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html
        """
        if self.unique_values and self.name in self.unique_values.keys():
            ret_value = np.array(self.unique_values[self.name])
        else:
            ret_value = super(LuxSeries, self).unique(*args, **kwargs)
        name = "Unnamed" if self.name is None else self.name
        self._history.append_event("unique", [name])
        self.add_to_parent_history("unique", [name])

        return ret_value

    #################
    # History Utils #
    #################

    def add_to_parent_history(self, op, cols):
        """
        Utility function for updating parent history

        N.B.: for df.col.value_counts() this is actually adding to the parent of df.col,
        not df.col.value_counts() so works how we want but is a subtle distinction.
        """
        if self._parent_df is not None:
            if self._parent_df.history.check_event(-1, op_name="col_ref", cols=cols):
                self._parent_df.history.edit_event(-1, op, cols, rank_type="parent")
            else:
                self._parent_df._history.append_event(op, cols, rank_type="parent")

    ##################
    # Type overrides #
    ##################

    def _infer_type(self, type):
        """
        update data type here and on parent
        type is in [ordinal, nominal, interval, ratio] and is converted to lux types
        Only update if a MORE selective type where nominal < ordinal < interval < ratio

        See: https://en.wikipedia.org/wiki/Level_of_measurement
        """
        name = "Unnamed" if self.name is None else self.name

        # print(f"Inferring type of {type} on {self.name}")

        # turn into broad lux categories, avoid if the series is an object because is likely a string or is temporal
        if (
            (type == "interval" or type == "ratio" or type == "quantitative")
            and self.dtype != "object"
            and not pd.api.types.is_datetime64_ns_dtype(self.dtype)
        ):
            type = "quantitative"

            self.set_data_type({name: type})

            if self._parent_df is not None and self.name is not None:
                self._parent_df.set_data_type({self.name: type})

    # -------------------------------------------------------------
    # Comparisons

    def __eq__(self, other):
        # ==, nominal
        self._infer_type("nominal")
        return super(LuxSeries, self).__eq__(other)

    def __ne__(self, other):
        # !=, nominal
        self._infer_type("nominal")
        return super(LuxSeries, self).__ne__(other)

    def __lt__(self, other):
        # <, ordinal
        self._infer_type("ordinal")
        return super(LuxSeries, self).__lt__(other)

    def __le__(self, other):
        # <=, ordinal
        self._infer_type("ordinal")
        return super(LuxSeries, self).__le__(other)

    def __gt__(self, other):
        # >, ordinal
        self._infer_type("ordinal")
        return super(LuxSeries, self).__gt__(other)

    def __ge__(self, other):
        # >=, ordinal
        self._infer_type("ordinal")
        return super(LuxSeries, self).__ge__(other)

    # -------------------------------------------------------------
    # Logical Methods

    # def __and__(self, other):
    #     return self._logical_method(other, operator.and_)

    # def __rand__(self, other):
    #     return self._logical_method(other, roperator.rand_)

    # def __or__(self, other):
    #     return self._logical_method(other, operator.or_)

    # def __ror__(self, other):
    #     return self._logical_method(other, roperator.ror_)

    # def __xor__(self, other):
    #     return self._logical_method(other, operator.xor)

    # def __rxor__(self, other):
    #     return self._logical_method(other, roperator.rxor)

    # -------------------------------------------------------------
    # Arithmetic Methods
    def __add__(self, other):
        # interval, quantitative
        self._infer_type("interval")
        return super(LuxSeries, self).__add__(other)

    def __radd__(self, other):
        # interval, quantitative
        self._infer_type("interval")
        return super(LuxSeries, self).__radd__(other)

    def __sub__(self, other):
        # interval, quantitative
        self._infer_type("interval")
        return super(LuxSeries, self).__sub__(other)

    def __rsub__(self, other):
        # interval, quantitative
        self._infer_type("interval")
        return super(LuxSeries, self).__rsub__(other)

    def __mul__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__mul__(other)

    def __rmul__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__rmul__(other)

    def __truediv__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__truediv__(other)

    def __rtruediv__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__rtruediv__(other)

    def __floordiv__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__floordiv__(other)

    def __rfloordiv__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__rfloordiv__(other)

    def __mod__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__mod__(other)

    def __rmod__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__rmod__(other)

    def __divmod__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__divmod__(other)

    def __rdivmod__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__rdivmod__(other)

    def __pow__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__pow__(other)

    def __rpow__(self, other):
        # ratio, quantitative
        self._infer_type("ratio")
        return super(LuxSeries, self).__rpow__(other)