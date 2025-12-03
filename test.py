import pandas as pd
from metric import score

# load a solution slice (must contain forward_returns and risk_free_rate)
sol = pd.read_csv("/mnt/data/train.csv", index_col="date_id").loc[8810:8990, ["forward_returns", "risk_free_rate"]]

# prepare a submission DataFrame with a prediction column
sub = pd.DataFrame({"prediction": 0.5}, index=sol.index)  # constant 0.5 position

print("metric:", score(sol.reset_index(drop=True), sub.reset_index(drop=True), None))
# NOTE: The metric implementation in your paste sets/uses DataFrame indexing a bit differently; ensure shapes align.
