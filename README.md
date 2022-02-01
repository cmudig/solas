# Solas

Solas is a jupyter extension for visualization recommendation that uses your analysis history to provide better recommendations.

Solas tracks analysis commands issued through the Pandas API and automatically visualies the returned data. By tracking the history of analysis commands, Solas visualizes returned data in a sensible encoding.

For example, when `df.describe()` or `df.corr()` are called Solas visulizes these functions as a boxplot and heatmap, respectively.

![describe and corr visualizations](figures/describe_corr.gif)

Solas also models your interest in each column of the data and uses this to show you relevant visualizations. Below, a user has most recently interacted with `Class` so it is shown at the top of the interface. Since they have also recently interacted with `Worldwide_Gross`, `MPAA_Rating`, and `Viewership`, they are shown in the enhance tab.

![Class vs other recent columns visualizations](figures/solas_interest_model.png)

## Easy install
Solas is a python library and jupyter extension. The python library can be downloaded from pip and the jupyter extension loaded through the extension manager.
```
pip install solas
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install solaswidget
```

Then you can run `import solas` in jupyter environments. Note this wont work outside of ipython envs.

## Contributing

For more information about installing for Development purposes and contributing see `CONTRIBUTING.md`