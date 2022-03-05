import os
import pandas as pd
from dataLoader import DataLoader

class DataVisualizer:
    def __init__(self, dl: DataLoader):
        self.dl = dl
        self.order = [
            'NoisySum',
            'NoisyHistogram1',
            'NoisyHistogram2',
            'ReportNoisyMax1-Lap',
            'ReportNoisyMax2-Exp',
            'ReportNoisyMax3-Lap',
            'ReportNoisyMax4-Exp',
            'SVT1',
            'SVT2',
            'SVT3',
            'SVT4',
            'SVT5',
            'SVT6',
            'NumericalSVT',
            'LaplaceMechanism',
            'OneTimeRAPPOR',
            'RAPPOR',
            'TruncatedGeometric',
            'LaplaceParallel',
            'PrefixSum',
        ]

    def cmp(self, key):
        try:
            return self.order.index(key)
        except ValueError:
            return 100

    def to_excel(self, topk=-1, filename='result.xls'):
        # 每个alg为一个sheet，每个method占若干个column
        d = self.dl._get_data()

        f = pd.ExcelWriter(filename)
        for alg, wits in sorted(d.items(), key=lambda a:self.cmp(a[0])):
            wits = sorted(wits, reverse=True)
            if topk > 0:
                wits = wits[:topk]
            df = pd.DataFrame(data=[wit.get_full() for wit in wits], index=range(1,len(wits)+1))
            df = df[sorted(df.columns, key=lambda k:k=='method')]  # put method column to last
            df.to_excel(f, sheet_name=alg)

        f.close()


