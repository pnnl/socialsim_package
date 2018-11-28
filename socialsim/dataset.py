import pandas as pd

class Dataset(pd.DataFrame):
    def __init__(self, data, meta_info):
        super(Dataset, self).__init__(data)

        self.meta_info = meta_info
