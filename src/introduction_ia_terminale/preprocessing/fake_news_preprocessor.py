import pathlib
from typing import Tuple

import pandas as pd


class FakeNewsPreprocessor:
    def __init__(self, root_path: pathlib.PurePath) -> None:
        self._true_mapping = {
            'politicsNews': 'politics',
            'worldnews': 'news'
        }
        self._fake_mapping = {
            'News': 'news',
            'politics': 'politics',
            'Government News': 'politics',
            'left-news': 'politics',
            'US_News': 'news',
            'Middle-east': 'news'
        }
        self._root_path = root_path
        self._df_fake, self._df_true = self._load_dataframes()
        self._dataframe = self._run_preprocessing()

    @property
    def fake(self):
        return self._df_fake
    
    @property
    def true(self):
        return self._df_true

    @property
    def dataframe(self):
        return self._dataframe

    def _load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_fake = pd.read_csv(self._root_path.joinpath('Fake.csv'))
        df_true = pd.read_csv(self._root_path.joinpath('True.csv'))
        return df_fake, df_true
    
    def _run_preprocessing(self) -> pd.DataFrame:
        self._df_true['subject'] = self._df_true['subject'].apply(lambda sub: self._true_mapping[sub])
        self._df_fake['subject'] = self._df_fake['subject'].apply(lambda sub: self._fake_mapping[sub])

        self._df_true['is_fake'] = 0
        self._df_fake['is_fake'] = 1

        df = pd.concat([self._df_true, self._df_fake]).reset_index(drop=True)
        df['content'] = df['title'] + '. ' + df['text']
        df.drop(columns=['date', 'title', 'text'], inplace=True)

        return df
