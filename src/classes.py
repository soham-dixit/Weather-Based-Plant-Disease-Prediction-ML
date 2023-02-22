from sdv.tabular import GaussianCopula, TVAE, CTGAN
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from src.base import BaseDataAugmentation


class gaussian_copula(BaseDataAugmentation):
    def __init__(self, df: pd.DataFrame, categorical: list, target: str) -> None:
        super().__init__(df, categorical, target)

    def fit(self) -> None:
        """
        Function to fit the Gaussain Copula model to each class
        """
        for classe in self.classes:
            self.models.append(GaussianCopula())
        beg = time.time()
        for i, classe in tqdm(enumerate(self.classes)):
            self.models[i].fit(self.df[self.df[self.target] == str(classe)])
        end = time.time()
        print("time:", end-beg)


class variational_autoencoder(BaseDataAugmentation):
    def __init__(self, df: pd.DataFrame, categorical: list, target: str) -> None:
        super().__init__(df, categorical, target)

    def fit(self) -> None:
        """
        Function to fit the TVAE model to each class
        """
        for classe in self.classes:
            self.models.append(TVAE())
        beg = time.time()
        for i, classe in tqdm(enumerate(self.classes)):
            self.models[i].fit(self.df[self.df[self.target] == str(classe)])
        end = time.time()
        print("time:", end-beg)


class ctgan_model(BaseDataAugmentation):
    def __init__(self, df: pd.DataFrame, categorical: list, target: str) -> None:
        super().__init__(df, categorical, target)

    def fit(self) -> None:
        """
        Function to fit the CTGAN model to each class
        """
        for classe in self.classes:
            self.models.append(CTGAN())
        beg = time.time()
        for i, classe in tqdm(enumerate(self.classes)):
            self.models[i].fit(self.df[self.df[self.target] == str(classe)])
        end = time.time()
        print("time:", end-beg)
