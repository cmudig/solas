#  Copyright 2019-2020 The Solas Authors.
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

from solas.vis.VisList import VisList
from solas.vis.Vis import Vis

# from solas.core.frame import SolasDataFrame
from solas.implicit import implicit_plotter

import solas

from IPython.core.debugger import set_trace


def implicit_mre(ldf: solas.core.frame.SolasDataFrame, hist_index=None):
    """
    Generates vis based off only most recent implicit action.

    Parameters
    ----------
    ldf : solas.core.frame
            SolasDataFrame with underspecified intent.

    Returns
    -------
    recommendations : Dict[str,obj]
            object with a collection of visualizations that result from the Implicit action.
    """
    # these events are cleansed when fetched
    col_list = ldf.history.get_implicit_intent(ldf.columns)
    if hist_index is not None:
        most_recent_event = ldf.history.get_hist_item(hist_index, ldf.columns)
    else:
        most_recent_event, hist_index = ldf.history.get_mre(ldf.columns)

    solas_vis = VisList([], ldf)

    # get unique vis for recent col ref
    if most_recent_event:
        solas_vis, used_cols = implicit_plotter.generate_vis_from_signal(most_recent_event, ldf, col_list)

    recommendation = {
        "action": "Implicit",
        "description": "",
        "long_description": "",
        "collection": solas_vis,
    }

    return recommendation, hist_index
