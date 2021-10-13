## Set up locally for dev purposes

You can install solas by building from the source code in your fork (or clone) directly:
```bash
git clone https://github.com/willeppy/solas.git
cd solas/
pip install -r requirements.txt
pip install -r requirements-dev.txt
python setup.py install
```

There are two ways to make sure your local changes show up when running locally. 

To rebuild and re-install the local version of `solas` system wide (or conda profile wide) run the below. (tbh not totally sure what this script does bc you have to re pip install as well. See below...)
```bash
python setup.py install
```

To get changes to show up IN OTHER LOCAL folders, you have to re pip install like this:
```bash
pip install .
```

However since this takes a bit to run it can be cumbersome. To develop faster, run a jupyter notebook __in this repo__ (like `DEMO_NB.ipynb`). Changes will propogate when you kill and re-start jupyter. You do not have to re-run the install script for changes to show up.

__*NOTE*__: if you run a notebook in another directory and import solas it will import the solas version from pip and NOT your local dev version. Run `solas.__path__` to see where your package is being imported from.

## solaswidget

If you are also making local edits to the [solaswidget](https://github.com/willeppy/solas-widget) repo, then follow the instructions in that repo to install. After making changes to `solaswidget`, re-install by running the below in the `solaswidget` repo

```bash
pip install . 
```

May have to re-build jupyter after widget is installed for the first time, to do so use
```bash
jupyter lab build
```
