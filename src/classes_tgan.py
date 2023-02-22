import time
import pandas as pd
from src.base import BaseDataAugmentation
from tgan.model import TGANModel
import tensorflow as tf

class TGAN(BaseDataAugmentation):
    def __init__(self, df:pd.DataFrame, categorical:list, target:str) -> None:
        super().__init__(df, categorical, target)

    def fit(self) -> None:
        """
        Function to fit the TGAN model
        """
        self.df[self.categorical] = self.df[self.categorical].astype(str)
        continuous_columns = []
        self.models.append(TGANModel(continuous_columns, max_epoch=1))
        beg = time.time()
        tf.reset_default_graph()
        self.models[0].fit(self.df)
        end = time.time()
        print("time:", end-beg)

    def generate(self) -> pd.DataFrame:
        """
        Function to generate new data samples 
        Output: dataframe with generated data
        """
        generated_data = pd.DataFrame()
        initial_size = self.df.shape[0]
        new_data = self.models[0].sample(initial_size*10)
        generated_data = pd.concat([generated_data, new_data])
        return generated_data