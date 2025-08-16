# Bitcoin-Price-Forecasting



# Can get the environment ready using these commands after fetching

python -m venv myenv

source myenv/bin/activate   # Mac/Linux
--or myenv\Scripts\activate  # Windows

pip install -r requirements.txt

---

A small difference in pandas-ta is required to resolve a small clashing.
Go into `myenv/lib/python3.12/site-packages/pandas_ta/momentum/squeeze_pro.py`
Replace `from numpy import NaN as npNaN` with `import numpy as np npNaN = np.nan`
