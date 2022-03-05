import os
import sys
import numpy as np
from dpfinder.log_parser.parser import Parser
from witness import Witness
from dataLoader import DataLoader


class MyParser(Parser):
    def get_empirical_eps(self, decimals=3):
        l = self.d['eps-empirical']
        l = np.array(l, dtype=np.float64)
        nans = np.isnan(l)
        l[nans] = 0
        l = np.round(l, decimals)
        return l

    def get_confirmed_eps(self, decimals=3):
        l = self.d['eps-confirmed']
        l = np.array(l, dtype=np.float64)
        nans = np.isnan(l)
        l[nans] = 0
        l = np.round(l, decimals)
        return l

    def get_witnesses(self, decimals=3):
        eps = self.get_confirmed_eps(decimals)
        eps_h = self.get_empirical_eps(decimals)
        a = self.d['a']
        a = np.round(a, decimals).tolist()
        b = self.d['b']
        b = np.round(b, decimals).tolist()
        o = self.d['o']
        o = np.array(o, dtype=self.output_type())
        o = np.round(o, decimals).tolist()
        pa = self.d['pa']
        pa = np.round(pa, decimals).tolist()
        pb = self.d['pb']
        pb = np.round(pb, decimals).tolist()
        err = eps-eps_h
        ws = []
        for i in range(len(eps)):
            ws.append(Witness(a[i],b[i],o[i],eps[i],'dp-finder', p1=pa[i], p2=pb[i], eps_h=eps_h[i], err=err[i]))
        ws.sort(reverse=True)
        return ws

    def get_universal_alg_name(self):
        name = self.get_alg_name()
        names = ['aboveThreshold', 'alg1', 'alg2', 'alg3', 'alg4', 'alg5', 'expMech', 'reportNoisyMax', 'sum']
        uni_names = ['SVT6', 'SVT1', 'SVT2', 'SVT3', 'SVT4', 'SVT5', 'ReportNoisyMax2-Exp', 'ReportNoisyMax1-Lap', 'NoisySum']
        name_map = dict(zip(names, uni_names))
        return name_map[name]

    def output_type(self):
        INTS = ['SVT6', 'SVT1', 'SVT2', 'SVT4', 'SVT5', 'ReportNoisyMax2-Exp', 'ReportNoisyMax1-Lap']
        if self.get_universal_alg_name() in INTS:
            return np.int
        return np.float64

class DPFinderLoader(DataLoader):
    def __init__(self):
        super(DPFinderLoader, self).__init__()
        self.logsdir = os.path.join(os.path.dirname(__file__), 'dpfinder/runners/logs/tf_runner')

    def load_data(self):
        for dir, dirnames, filenames in os.walk(self.logsdir):
            for filename in filenames:
                if filename.endswith('data.log'):
                    p = MyParser(os.path.join(dir, filename))
                    self._push(p.get_universal_alg_name(), *p.get_witnesses())


if __name__ == '__main__':
    from dataVisualizer import DataVisualizer
    dl = DPFinderLoader()
    dl.load_data()
    vi = DataVisualizer(dl)
    vi.to_excel()
