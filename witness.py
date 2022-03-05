from abc import ABC
from collections import OrderedDict


class Witness(ABC):
    def __init__(self, a1, a2, s, eps, method, **kwargs):
        self.meth = method
        self.a1 = a1
        self.a2 = a2
        self.s = s
        self.eps = eps
        self.kwargs = OrderedDict(kwargs)

    def get_witness(self):
        d = OrderedDict({'a1': self.a1,
             'a2': self.a2,
             's': self.s,
             'eps': self.eps})
        return d

    def get_full(self):
        d = self.get_witness()
        d.update(self.kwargs)
        d['method'] = self.meth
        return d

    def get_keys(self):
        return list(self.get_full().keys())

    def __lt__(self, other):
        return self.eps < other.eps

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        d = self.get_full()
        l = []
        for k, v in d.items():
            l.append(f'{k}={v}')
        return ' '.join(l)

if __name__ == '__main__':
    from dataLoader import DataLoader
    from dataVisualizer import DataVisualizer
    import pandas as pd
    w=Witness(1,2,3,3,"dpfinder",pa=1,pb=2)
    w2 = Witness(3,4,5,5,'statdp',epsm=3)
    df = pd.DataFrame([w.get_full(), w2.get_full()])
    # df1.to_excel("test.xls")

    # dl = DataLoader()
    # dl._push('SVT', w, w2)
    # dv = DataVisualizer(dl)
    # dv.to_excel(filename='test.xls')