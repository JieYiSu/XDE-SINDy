# Engineering experiment

`underwater_welding_simulation.py` is a reduced-order, calibratable digital twin of a six-axis underwater welding robot with process variables, thermal response, melt-pool targets, path tracking, porosity, workspace, and collision penalties.  It compares XDE-SINDy with budget-matched engineering sanity baselines implemented locally in the script.

Run the two files directly:

```powershell
python experiments/underwater_welding_simulation.py
python experiments/underwater_welding_report.py
```

The configuration is embedded in the scripts and in `configs/underwater_welding_digital_twin.json`; there are no required command-line options.
