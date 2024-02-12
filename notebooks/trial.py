# %%
import sys
sys.path.append('..')


# %%
from utils_reboot.datasets import Dataset

dataset = Dataset("cardio", path = "../data/real/")
dataset.drop_duplicates()

# %%
from model_reboot.EIF_reboot import ExtendedTree, ExtendedIsolationForest
import numpy as np

# %%
I=ExtendedIsolationForest(True,n_estimators=200)

# %%
I.fit(dataset.X)
I.fit(dataset.X)

# %%
# from pyinstrument import Profiler
# profiler = Profiler()
# profiler.start()
# I.fit(dataset.X)
# profiler.stop()
# profiler.print()

# # %%
# dataset.y

# # %%
# m = np.c_[I.predict(dataset.X),dataset.y]
# m[np.argsort(m[:,0,])][::-1]

# # %% [markdown]
# # # Test time

# # %%
# import tqdm

# # %%
# dataset = Dataset("wine", path = "../data/real/")
# dataset.drop_duplicates()
# I=ExtendedIsolationForest(True,n_estimators=400)
# for _ in tqdm.tqdm(range(10)):
#     I.fit(dataset.X)
#     I.predict(dataset.X)


# %%



