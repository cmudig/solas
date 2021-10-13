import pandas as pd
from solas.vis.VisList import VisList
from solas.vis.Vis import Vis
from solas.vis.CustomVis import CustomVis
from solas.history.event import Event
from solas.core.frame import SolasDataFrame
from solas.core.series import SolasSeries
from solas.core.groupby import SolasGroupBy
import solas.utils.defaults as solas_default

from solas.implicit import cg_plotter

import solas
import altair as alt
import random

from sklearn.preprocessing import LabelEncoder
from pyemd import emd_samples


from IPython.core.debugger import set_trace

### common utils and settings
tf_scale = alt.Scale(domain=[True, False], range=[solas_default.MAIN_COLOR, solas_default.BG_COLOR])

PLOT_CARD_THRESH = 12

##################
# MAIN function #
##################
def generate_vis_from_signal(signal: Event, ldf: SolasDataFrame, ranked_cols=[]):
    """
    Parameters
    ----------
        signal: solas.history.Event
            History event

        ldf: solas.core.frame
            the ldf to be plotted

    Returns
    -------
        chart_list: VisList
            VistList of returned vis
        used_cols: list
            which columns were used in the returned vis(i)
    """
    ldf.history.freeze()
    vis_list = VisList([])
    used_cols = []
    if signal.op_name == "value_counts" or signal.op_name == "unique":
        vis_list, used_cols = process_value_counts(signal, ldf)

    elif signal.op_name == "describe":
        vis_list, used_cols = process_describe(signal, ldf)
    elif signal.op_name == "gb_describe":
        vis_list, used_cols = process_gb_describe(signal, ldf)
    elif (
        signal.op_name == "filter"
        or signal.op_name == "query"
        or signal.op_name == "slice"
        or signal.op_name == "gb_filter"
        or signal.op_name == "loc"
        or signal.op_name == "iloc"
    ):

        vis_list, used_cols = process_filter(signal, ldf, ranked_cols)

    elif signal.op_name == "dropna":
        vis_list, used_cols = process_filter(signal, ldf, ranked_cols)

    elif signal.op_name == "isna" or signal.op_name == "notnull":
        vis_list, used_cols = process_null_plot(signal, ldf)

    elif signal.cols and not ldf.pre_aggregated:  # generic recs
        vis_list, used_cols = process_generic(signal, ldf)

    ldf.history.unfreeze()

    return vis_list, used_cols


########################
# VALUE_COUNT plotting #
########################
def process_value_counts(signal, ldf):
    """
    Generate 1d distribution plot either from raw data (if parent and unaggregated)
    or from child

    Returns
    -------
        vis_list: VisList
            with the vis
        array: []
            which columns were used
    """
    try:
        rank_type = signal.kwargs.get("rank_type", None)
        # in the unique case, there is no parent but we still want the corresponding visualization.
        c_name = signal.cols[0]
        if ((rank_type == "parent") or rank_type is None) and not ldf.pre_aggregated:
            # due to the most recent event is set to be non-parent, 
            # this part of the conditional statement actually is never used.
            if ldf.data_type[c_name] == "quantitative":
                clse = solas.Clause(attribute=c_name, mark_type="histogram")
            else:
                clse = solas.Clause(attribute=c_name, mark_type="bar")

            vis_list = VisList([clse], ldf)

        else:  # "child" AND ldf is pre_aggregated
            # make vis consistent with normal histogram from history
            v = Vis(
                [
                    solas.Clause(
                        attribute=c_name,
                        data_type="nominal",
                        data_model="dimension",
                        aggregation="",
                        channel="x",
                    ),
                    solas.Clause(
                        attribute="Number of Records",
                        data_type="quantitative",
                        data_model="measure",
                        aggregation=None,
                        channel="y",
                    ),
                ]
            )
            flat = ldf.reset_index()
            flat = flat.rename(columns={"index": c_name, c_name: "Number of Records"})
            vis_list = VisList([v], flat)

        return vis_list, [c_name]

    except (IndexError, KeyError):
        return VisList([], ldf), []


#####################
# DESCRIBE plotting #
#####################
def process_gb_describe(signal, ldf):
    plot_df = None
    groupby_cols = None
    if ldf._parent_df is not None and ldf._parent_df._parent_df is not None:
        if isinstance(ldf._parent_df,  SolasGroupBy) and isinstance(ldf._parent_df._parent_df, SolasDataFrame):
            # the first case corresponds to the following three cases:
            # 1) df.groupby("Origin").describe()
            # 2) df.groupby("Origin")[["Cylinders", "Weight"]].describe()
            # 3) df.groupby("Origin")["Cylinders"].describe()
            # in the first two cases, the resulting dataframe is multi-index, 
            # while the third one needs to be specially treated


            # the peculiar thing is that, in the third case, df.groupby("Origin")["Cylinders"].describe()
            # the column reference event will be logged to df instead of df.groupby("Origin")
            # and because we have not overriden the __getitem__ funciton of the SolasGroupby,
            # the parent of df.groupby("Origin")["Cylinders"] is the same as the df.groupby("Origin"), that is, df
            # This is why the third case will fall into this condition branch
            plot_df = ldf._parent_df._parent_df
            if isinstance(ldf.columns, pd.MultiIndex):
                groupby_cols = list(ldf.columns.get_level_values(0).unique())
            elif ldf._parent_df._parent_df.history.check_event(-1, op_name="col_ref"):
                groupby_cols = ldf._parent_df._parent_df.history[-1].cols
        elif (isinstance(ldf._parent_df, SolasDataFrame)
            and isinstance(ldf._parent_df._parent_df, SolasGroupBy)
            and ldf._parent_df.history.check_event(-1, op_name="col_ref")
        ):
            # the second corresponds to the following two cases:
            # 1) df.groupby("Origin").describe()[["Cylinders", "Weight"]]
            # 2) df.groupby("Origin").describe()["Cylinders"]
            plot_df = ldf._parent_df._parent_df._parent_df
            groupby_cols = ldf._parent_df.history[-1].cols
        
    if plot_df is not None and groupby_cols is not None:
        plot_df.history.freeze()
        groupby_attr = ldf.index.name
        
        plot_df.maintain_metadata()
        filter_vals = plot_df.unique_values[groupby_attr]
        collection = []
        data_types = dict(plot_df.dtypes)
        for col in groupby_cols:
            for attr_val in filter_vals:
                if data_types[col] == object or plot_df._data_type[col] == "nominal":
                    v = Vis([solas.Clause(col, mark_type="bar"), solas.Clause(attribute=groupby_attr, value=attr_val)], plot_df)
                elif plot_df._data_type[col] == "temporal":
                    v = Vis([solas.Clause(col, mark_type="line"), solas.Clause(attribute=groupby_attr, value=attr_val)], plot_df)
                else:
                    # it is then numeric so it is safe to draw boxplot.
                    v = Vis([solas.Clause(col, mark_type="boxplot"), solas.Clause(attribute=groupby_attr, value=attr_val)], plot_df)
                collection.append(v)

        vl = VisList(collection)
        return vl, []
    else:
        return VisList([]), []

def process_describe(signal, ldf):
    """
    Plots boxplots of either parent df if this is the describe df or of this df

    Parameters
    ----------
        signal: solas.history.Event
            History event that is a FILTER

        ldf: solas.core.frame
            the ldf to be plotted

    Returns
    -------
        chart_list: VisList
        array: []
            Empty array of used cols so not excluded in other vis
    """
    plot_df = None
    if (ldf._parent_df is not None and (
            (len(ldf) == 8 and all(ldf.index == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"])) # the numeric case
        or  (len(ldf) == 4 and all(ldf.index == ["count", "unique", "top", "freq"])) # the object case
        or  (len(ldf) == 11 and all(ldf.index == ["count", "unique", "top", "freq", "mean", "std", "min", "25%", "50%", "75%", "max"])) # the mixed case
        ) 
    ):
        if isinstance(ldf._parent_df, SolasSeries):
            name = "Unnamed" if ldf._parent_df.name is None else ldf._parent_df.name
            plot_df = SolasDataFrame({name: ldf._parent_df})
        else:
            plot_df = ldf._parent_df
    else:
        plot_df = ldf
    
    collection = []
    data_types = dict(ldf.dtypes)
    for col in signal.cols:
        if data_types[col] == object:
            # then it is string; 
            # note here we choose to comply with the describe convention instead of solas.
            # by drawing quantitative but nominal variables as boxplot.
            v = Vis([solas.Clause(col, mark_type="bar")], plot_df)
        elif ldf._data_type[col] == "temporal":
            v = Vis([solas.Clause(col, mark_type="line")], plot_df)
        else:
            # it is then numeric so it is safe to draw boxplot.
            v = Vis([solas.Clause(col, mark_type="boxplot")], plot_df)
        collection.append(v)

    vl = VisList(collection)

    return vl, []


###################
# FILTER plotting #
###################
def process_filter(signal, ldf, ranked_cols, num_vis_cap=5):
    """
    Decides if plotting parent that WAS filtered, or the resulting df
    FROM a filter and plots accordingly.

    Parameters
    ----------
        signal: solas.history.Event
            History event that is a FILTER

        ldf: solas.core.frame
            the ldf to be plotted

    Returns
    -------
        chart_list: VisList or empty array
        plot_cols: array
            which cols were used


    """
    rank_type = signal.kwargs.get("rank_type", None)
    child_df = signal.kwargs.get("child_df", None)
    parent_mask = signal.kwargs.get("filt_key", None)
    filter_axis = signal.kwargs.get("filter_axis", None)

    # assign parent and child
    p_df, c_df = None, None
    if rank_type == "parent" and child_df is not None:
        p_df = ldf
        c_df = child_df
    elif rank_type == "child" and ldf._parent_df is not None:
        if isinstance(ldf._parent_df, solas.core.groupby.SolasGroupBy):
            p_df = ldf._parent_df._parent_df
        else:
            p_df = ldf._parent_df
        c_df = ldf

    if filter_axis == 0 or filter_axis == "columns":
        # we add this conditional statement to avoid drawing filter visualization for dropna(axis="columns")
        return VisList([], ldf), []
    else:
        # in the following, we assume that the filter is applied by rows instead of by columns
        # in other words, only rows are dropped, so the columns of the parernt and the child should be the same
        # get mask
        if parent_mask is not None:
            # this information is available only in the `_getitem_bool_array` case
            mask = parent_mask
        else:
            mask, same_cols = compute_filter_diff(p_df, c_df)

        # get cols with large dist change
        vis_cols = get_col_recs(p_df, c_df)

        # populate vis
        all_used_cols = set()
        chart_list = []
        if rank_type == "child":
            chart_list.append(plot_filter_count(p_df, mask))

        if p_df is not None:
            for c in vis_cols[:num_vis_cap]:
                _v = plot_filter(p_df, [c], mask)
                chart_list.append(_v)
                all_used_cols.add(c)

        vl = VisList(chart_list)

        return vl, list(all_used_cols)


def get_col_recs(parent_df, child_df):
    """
    Look at each column and calculate distance metric by column
    """
    dists = []
    parent_df.history.freeze()
    child_df.history.freeze()

    # TODO store this on the df so dont have to recalc so much
    # TODO calc distance for 2d as well
    valid_columns = set(parent_df.columns) & set(child_df.columns)
    for c in valid_columns:
        p_data = parent_df[c].dropna().values
        c_data = child_df[c].dropna().values

        if parent_df.data_type[c] != "quantitative" and parent_df.cardinality[c] > PLOT_CARD_THRESH:
            dist = -1
        else:
            dist = calc_dist_distance(p_data, c_data, parent_df.data_type[c])

        if dist != -1:
            dists.append((c, dist))

    dists.sort(key=lambda x: x[1], reverse=True)
    col_order = [i[0] for i in dists]

    parent_df.history.unfreeze()
    child_df.history.unfreeze()

    return col_order


def calc_dist_distance(p_data, c_data, dtype):
    """
    Calculate wasserstein / earth movers distance between two samples.
    c_data must be subset of p_data
    """
    try:
        if dtype == "nominal":
            le = LabelEncoder()
            le.fit(p_data)
            p_data = le.transform(p_data)
            c_data = le.transform(c_data)
        return emd_samples(p_data, c_data)
    except Exception:
        return -1


def compute_filter_diff(old_df, filt_df):
    """
    Assumes filt_df is a subset of old_df. Creates indicator the size of the larger df

    True = in both, False = only in larger
    """
    old_df = old_df.copy()
    filt_df = filt_df.copy()
    # filtered should always be smaller
    if len(filt_df) > len(old_df):
        _t = filt_df
        filt_df = old_df
        old_df = _t
    # this only works when the parent and child dataframe share a set of columns
    # while in the loc/iloc cases, it is possible that filters are applied to rows and columns together.
    # therefore we would choose to merge according to their indices.
    # Nevertheless, the current method is the safest and will always be our first choice if possible
    if len(old_df.columns) == len(filt_df.columns):
        _d = old_df.merge(filt_df, indicator=True, how="left")
        indicator = _d._merge == "both"  # .astype(int)
        # which cols change? this isnt very informative since
        # many columns change other than the filter.
        same_cols = list(old_df.columns[old_df.nunique() == filt_df.nunique()])
    else:
        _d = old_df.merge(filt_df, indicator=True, how="left", left_index=True, right_index=True)
        indicator = _d._merge == "both"  # .astype(int)
        same_cols = [column for column in old_df.columns if column not in filt_df.columns]
    return indicator, same_cols


def plot_filter_count(ldf, mask, c_col_name="In filter?", c_title="Filtered Data Count"):
    """
    For filtered dfs, plot the count of columns compared to count of unfiltered df

    Parameters
    ----------
        ldf: solas.core.SolasDataFrame
            parent ldf that was filtered

        mask: boolean list or series
            True if in filter, False if not

    Returns
    -------
        chart: CustomVis chart
    """

    ldf = ldf.copy()
    ldf[c_col_name] = mask

    chart = (
        alt.Chart(ldf)
        .mark_bar(size=75)
        .encode(
            y=alt.Y("count()", title=c_title),
            color=alt.Color(c_col_name, scale=tf_scale, legend=None),
            order=alt.Order(
                c_col_name, sort="descending"
            ),  # make sure stack goes True then False for filter
        )
    )

    # DONT use interactive for this chart, it breaks bc ordinal scale I think
    intent = []  # NOTE: this isnt great intent for this chart
    cv = CustomVis(intent, chart, ldf, width=90, override_c_config={"interactive": False})

    return cv


def plot_filter(ldf, cols, mask, card_thresh=PLOT_CARD_THRESH, filt_frac_thresh=0.1):
    """
    Plot 1d or 2d plot of filted df

    Parameters
    ----------
        ldf: solas.core.SolasDataFrame
            parent ldf that was filtered

        cols: list
            which col(s) should be in the plot

        mask: boolean list or series
            True if in filter, False if not

    Returns
    -------
        chart: CustomVis chart
    """
    ldf = ldf.copy()
    ldf["filt_mask"] = mask

    # make sure filtered data is at least 10% of dataset, if not only use filter
    n = len(ldf)
    filt_df_true = ldf[mask]
    filt_n = len(filt_df_true)

    if (filt_n / n) < filt_frac_thresh:
        ldf = filt_df_true

    chart = None
    intent = []

    if len(cols) == 1 and cols[0] in ldf.data_type:
        x_var = cols[0]
        x_title = f"{x_var}"
        x_d_type = ldf.data_type[x_var]
        _bin = x_d_type == "quantitative"

        filt_text = None
        if x_d_type == "nominal":
            if ldf.cardinality[x_var] > card_thresh:
                vc = filt_df_true[x_var].value_counts()
                _most_common = vc.iloc[:card_thresh].index
                ldf = ldf[ldf[x_var].isin(_most_common)]
                x_title += f" (top {card_thresh})"
                filt_text = f"+ {len(vc) - card_thresh} more..."
            alt_x_enc = alt.X(x_var, type=x_d_type, bin=_bin, title=x_title, sort="-y")
        else:
            alt_x_enc = alt.X(x_var, type=x_d_type, bin=_bin, title=x_title)

        chart = (
            alt.Chart(ldf)
            .mark_bar()
            .encode(
                x=alt_x_enc,
                y=f"count({x_var}):Q",
                color=alt.Color("filt_mask", scale=tf_scale, title="Is Filtered?", legend=None),
            )
        )

        intent = [solas.Clause(x_var, data_type=x_d_type)]

        if filt_text:
            filt_label = alt.Chart(ldf).mark_text(
                x=155,
                y=142,
                align="right",
                color="#ff8e04",
                fontSize=11,
                text=filt_text,
            )

            chart = chart + filt_label

    elif len(cols) >= 2 and (cols[0] in ldf.data_type) and (cols[1] in ldf.data_type):

        # set x as quant if possible
        if ldf.data_type[cols[0]] == "quantitative":
            x_var = cols[0]
            y_var = cols[1]
        else:
            x_var = cols[1]
            y_var = cols[0]

        x_d_type = ldf.data_type[x_var]
        y_d_type = ldf.data_type[y_var]
        x_bin = x_d_type == "quantitative"
        y_bin = y_d_type == "quantitative"

        x_title = f"{x_var}"
        y_title = f"{y_var}"

        filt_text_x = None
        # filt_text_y = None
        # filter for cardinality if high cardinality nominal vars
        if x_d_type == "nominal" and ldf.cardinality[x_var] > card_thresh:
            vc = filt_df_true[x_var].value_counts()
            _most_common = vc.iloc[:card_thresh].index
            ldf = ldf[ldf[x_var].isin(_most_common)]
            x_title += f" (top {card_thresh})"
            filt_text_x = f"+ {len(vc) - card_thresh} more..."

        if y_d_type == "nominal" and ldf.cardinality[y_var] > card_thresh:
            vc = filt_df_true[y_var].value_counts()
            _most_common = vc.iloc[:card_thresh].index
            ldf = ldf[ldf[y_var].isin(_most_common)]
            y_title += f" (top {card_thresh}/{len(vc)}...)"
            # filt_text_y = f"+ {len(vc) - card_thresh} more..." # TODO what if x and y are high card

        bg = (
            alt.Chart(ldf)
            .mark_circle(color=solas_default.BG_COLOR)
            .encode(
                x=alt.X(x_var, type=x_d_type, bin=x_bin, title=x_title),
                y=alt.Y(y_var, type=y_d_type, bin=y_bin, title=y_title),
                size=alt.Size("count()", legend=None),
            )
        )

        filt_chart = (
            alt.Chart(ldf)
            .mark_circle(color=solas_default.MAIN_COLOR)
            .transform_filter((alt.datum.filt_mask == True))
            .encode(
                x=alt.X(x_var, type=x_d_type, bin=x_bin),
                y=alt.Y(y_var, type=y_d_type, bin=y_bin),
                size=alt.Size("count()", legend=None),
            )
        )

        chart = bg + filt_chart

        intent = [solas.Clause(x_var, data_type=x_d_type), solas.Clause(y_var, data_type=y_d_type)]

        if filt_text_x:
            filt_c = alt.Chart(ldf).mark_text(
                x=155,
                y=142,
                align="right",
                color="#ff8e04",
                fontSize=11,
                text=filt_text_x,
            )

            chart = chart + filt_c

        # if filt_text_y:
        #     filt_c = alt.Chart(ldf).mark_text(
        #         x=15,
        #         y=135,
        #         align="right",
        #         color="#ff8e04",
        #         fontSize=11,
        #         text= filt_text_y,
        #     )

        #     chart = chart + filt_c

    cv = CustomVis(intent, chart, ldf)

    return cv


######################
# Null df plotting   #
######################
def process_null_plot(signal, ldf):
    """
    Generate count histograms of df if boolean showing isna
    """
    rank_type = signal.kwargs.get("rank_type", None)

    chart_list = []

    if rank_type == "child" and all(ldf.dtypes == "bool"):

        for c in ldf.columns:
            chart = plot_na_count(ldf, c, f"{c} {signal.op_name}")
            chart_list.append(chart)

    vl = VisList(chart_list)

    return vl, []


def plot_na_count(ldf, c_col_name, c_title):
    """
    For count dfs, plot the count of columns compared to count of unfiltered df

    Parameters
    ----------
        ldf: solas.core.SolasDataFrame
            parent ldf that was filtered

        mask: boolean list or series
            True if in filter, False if not

    Returns
    -------
        chart: CustomVis chart
    """
    chart = (
        alt.Chart(ldf)
        .mark_bar(size=75)
        .encode(
            y=alt.Y("count()", title=c_title),
            color=alt.Color(c_col_name, scale=tf_scale, legend=None),
            order=alt.Order(
                c_col_name, sort="descending"
            ),  # make sure stack goes True then False for filter
        )
    )

    # DONT use interactive for this chart, it breaks bc ordinal scale I think
    intent = [c_col_name]
    cv = CustomVis(intent, chart, ldf, width=90, override_c_config={"interactive": False})

    return cv


######################
# GENERIC plotting   #
######################
def process_generic(signal, ldf):
    vl = []

    for c in signal.cols:
        v = Vis([solas.Clause(attribute=c)])
        vl.append(v)

    used_cols = signal.cols
    vis_list = VisList(vl, ldf)

    return vis_list, used_cols
