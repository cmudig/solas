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

from solas.interestingness.interestingness import interestingness
import solas
from solas.executor.PandasExecutor import PandasExecutor
from solas.executor.SQLExecutor import SQLExecutor
import solas


def custom(ldf, **kwargs):
    """
    Generates user-defined vis based on the intent.

    Parameters
    ----------
    ldf : solas.core.frame
        SolasDataFrame with underspecified intent.

    Returns
    -------
    recommendations : Dict[str,obj]
        object with a collection of visualizations that result from the Distribution action.
    """
    recommendation = {
        "action": "Current Vis",
        "description": "Shows the list of visualizations generated based on user specified intent",
        "long_description": "Shows the list of visualizations generated based on user specified intent",
    }

    recommendation["collection"] = ldf.current_vis

    vlist = ldf.current_vis
    solas.config.executor.execute(vlist, ldf)
    for vis in vlist:
        vis.score = interestingness(vis, ldf)
    # ldf.clear_intent()
    col_order = ldf.history.get_implicit_intent(ldf.columns)
    vlist.sort(remove_invalid=True, intent_cols=col_order)
    return recommendation


def custom_actions(ldf, **kwargs):
    """
    Generates user-defined vis based on globally defined actions.

    Parameters
    ----------
    ldf : solas.core.frame
        SolasDataFrame with underspecified intent.

    filter_cols: list or None
        attributes that must be used as one of channels in each returned chart from the action tab

    Returns
    -------
    recommendations : Dict[str,obj]
        object with a collection of visualizations that were previously registered.
    """
    if len(solas.config.actions) > 0 and (len(ldf) > 0 or solas.config.executor.name != "PandasExecutor"):
        recommendations = []
        for action_name in solas.config.actions.keys():
            display_condition = solas.config.actions[action_name].display_condition
            if display_condition is None or (display_condition is not None and display_condition(ldf)):
                args = solas.config.actions[action_name].args
                if args:
                    recommendation = solas.config.actions[action_name].action(ldf, args, **kwargs)
                    # filter_cols then passed to each action tab
                else:
                    recommendation = solas.config.actions[action_name].action(ldf, **kwargs)
                recommendations.append(recommendation)
        return recommendations
    else:
        return []
