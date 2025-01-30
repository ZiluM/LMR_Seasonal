import xarray as xr
import numpy as np
from dataclasses import dataclass
import os
import pandas as pd
import cftime
import datetime as dt
from dateutil.relativedelta import relativedelta
# import util
from .util import time_range


# Define a dictionary mapping climate variable names to their respective labels.
label_dict = {"pr": "Amon", "psl": "Amon", "tas": "Amon", "tos": "Omon", "zg": "Amon", "zos": "Omon"}

# Define a dictionary specifying time ranges for different experiments.
time_range_dict = {"historical": "185001-201412",
                   "past1000": "085001-185012"
                   }
# Define a dictionary mapping level data names to their corresponding variable names.
level_data = {"z500": "zg",'z200': "zg"}


# Define a dataclass for handling individual climate data variables.
@dataclass
class SingleData:
    name: str  # Name of the variable
    path: str  # Path to the data files
    model_name: str  # Model name
    exp_name: str = "historical"  # Experiment name
    exp_setting: str = "r1i1p1f1"  # Experiment setting
    grid: str = "gn"  # Grid type
    chunk: dict = None  # Chunking configuration (optional)
    extra: str = "" # Extra string for the file name
    # Compute a time array based on the specified experiment and time range.
    # @property
    # def time_array(self):
    #     if not hasattr(self, "time_range"):
    #         self.time_range = time_range_dict[self.exp_name]
    #     time_begin, time_end = self.time_range.split("-")
        # time_begin = cftime.DatetimeGregorian(time_str[:4], 1, 1)
        # time_begin, time_end = map(lambda time_str: np.datetime64(cftime.DatetimeGregorian(int(time_str[:4]), int(time_str[4:]),1).isoformat(),"M"), [time_begin, time_end])
        # time_range = np.arange(time_begin, time_end + np.timedelta64(1, 'M'), np.timedelta64(1, 'M'),)
        # time_range = [cftime.datetime.strptime(tm,'%Y-%m', calendar='noleap') for tm in time_range.astype('str')] 
        # time_begin = dt.datetime(int(time_begin[:4]),int(time_begin[4:6]),1,0)
        # time_end = dt.datetime(int(time_end[:4]),int(time_end[4:6]),1,0)
        # date_len = (time_end.year - time_begin.year + 1)*12  
        # time_range = []
        # for date in range(date_len):
        #     time_range.append( time_begin + relativedelta(months=date))
        # # print(time_range)
        # return time_range
        # return pd.date_range(time_begin, time_end, freq="MS")

    # .shift(14, freq="d")

    # Load the climate data for the specified variable.
    def load(self):
        if self.name in level_data:
            self.name, self.level = level_data[self.name], int(self.name[1:]) * 100
        else:
            self.level = None
        self.part_label = label_dict[self.name]
        self.time_range = time_range_dict[self.exp_name]
        self.full_path = os.path.join(
            self.path, f"{self.extra}{self.name}_{self.part_label}_" +
            f"{self.model_name}_{self.exp_name}_{self.exp_setting}_{self.grid}_{self.time_range}.nc")
        # return
        data = xr.open_dataset(self.full_path)[self.name]
        # print(data.time)
        # data = data.assign_coords(time=self.time_array)
        data['time'] = time_range(time_range_dict[self.exp_name])
        if self.level is not None:
            data = data.loc[{'plev': self.level}].drop("plev")
        return data


# Define a dataclass for loading multiple climate data variables.
@dataclass
class DataLoader:
    names: list  # List of variable names
    path: str
    model_name: str
    exp_name: str = "historical"
    exp_setting: str = "r1i1p1f1"
    grid: str = "gn"
    chunk: dict = None  # Chunking configuration (optional)
    extra: str = "" # Extra string for the file name

    # Load and concatenate multiple climate data variables into a DataArray.
    def load(self) -> xr.DataArray:
        data_ls = []
        # if list(self.chunk.keys())[0] == "channel" and len(list(self.chunk.keys())) == 1:
        #     chunk = None
        #     channel_chunk = True
        # else:
        #     chunk = self.chunk
        for name in self.names:
            single_data = SingleData(name=name,
                                     path=self.path,
                                     model_name=self.model_name,
                                     exp_name=self.exp_name,
                                     exp_setting=self.exp_setting,
                                     grid=self.grid,
                                     extra=self.extra,
                                    #  chunk=chunk
                                     )
            data_ls.append(single_data.load().expand_dims({"channel": 1}))
        # if channel_chunk is True:
        #     dataarray_ls = xr.concat(data_ls, dim="channel").chunk(self.chunk)
        #     # print(dataarray_ls)
        # else:
        dataarray_ls = xr.concat(data_ls, dim="channel")
        #.chunk(self.chunk) 
        del data_ls
        return dataarray_ls

        # print(res)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # sd = SingleData(
    #     name="tos",
    #     path="/glade/work/zilumeng/SSNLIM/data/processed",
    #     model_name="CESM2"
    # )
    # # print(sd.time_array)
    # print(sd.load())
    sd = DataLoader(names=["tos", "z500", 'zos', 'pr', 'psl', 'tas'],
                    path="/glade/work/zilumeng/SSNLIM/data/processed",
                    model_name="CESM2",
                    chunk={"time": 10})
    res = sd.load()
    # res.std("channel")
    resa = res.groupby("time.month") - res.groupby("time.month").mean()
    # resa.compute()
    resa[-1, 0].plot()
    print(resa[0, 0])
    plt.savefig("kk.png")
    print(resa)
