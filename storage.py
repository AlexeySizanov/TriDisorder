import numpy as np
import pandas as pd
from pathlib import Path
from uuid import uuid4
from .system2d import TDSystem
import os

class Storage:
    def __init__(self, name, replace=False):
        self.name = name
        self.path = Path('/Data/td') / name

        if not self.path.exists():
            self.path.mkdir()
            self.data = pd.DataFrame(columns=['energy', 'name', 'comment'])

        else:
            if replace:
                os.removedirs(self.path)
                self.path.mkdir()
                self.data = pd.DataFrame(columns=['energy', 'name', 'comment'])

            else:
                df_path = self.path / (name + '.csv')
                if not df_path.exists():
                    raise Exception(f'Storage.__init__ : folder "{name}" exists but file "{name + ".csv"}" does not.')
                self.data = pd.read_csv(df_path)

    def add(self, s: TDSystem, comment=None):
        name = str(uuid4())
        self.data.loc[self.data.shape[0]] = float(s.energy_density()), name, comment
        s.save(self.path / (name+'.npz'))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        path = self.path / (self.data.name[index] + '.npz')
        return TDSystem.load(filename=path)

    def save(self):
        path = self.path / (self.name + '.csv')
        self.data.to_csv(path)
