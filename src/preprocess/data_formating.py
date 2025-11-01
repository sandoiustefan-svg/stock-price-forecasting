import pandas as pd
from typing import Tuple

class DataFormating:

    def __init__(self):
        pass

    def long_format(self, row: pd.Series, value_columns: list[str]):
        start = pd.Timestamp(year=int(row["Starting Year"]),
                         month = int(row["Starting Month"]),
                         day=1)
        
        n = int(row["N"])

        vals = row[value_columns].values[:n]

        dates = pd.date_range(start=start, periods=n, freq="MS")

        out = pd.DataFrame({
            'Series' : row['Series'],
            'date' : dates,
            'value' : vals
        })

        return out
    
    def split_the_data(self, data: pd.DataFrame, number_val: int, number_test: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = data.copy()

        parts = {'train': [], 'val': [], 'test': []}

        for id, sequence in data.groupby("Series", sort=False):
            sequence = sequence.sort_values("date")

            sequence_test = sequence.iloc[-number_test:]
            sequence_remaining = sequence.iloc[:-number_test]

            sequence_val = sequence_remaining[-number_val:]
            sequence_train = sequence_remaining[:-number_val]

            parts["train"].append(sequence_train)
            parts["val"].append(sequence_val)
            parts["test"].append(sequence_test)

        train = pd.concat(parts["train"], ignore_index=True)
        val = pd.concat(parts["val"], ignore_index=True)
        test = pd.concat(parts["test"], ignore_index=True)

        return train, val, test

        