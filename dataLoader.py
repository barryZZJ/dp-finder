from abc import ABC, abstractmethod
from typing import List, Dict
from witness import Witness


class DataLoader(ABC):
    def __init__(self):
        self.data = {}  # type: Dict[List[Witness]]

    def load_data(self): pass

    def _push(self, alg, *witnesses):
        lWits = self.data.setdefault(alg, [])
        lWits.extend(witnesses)

    def _get_data(self):
        return self.data

    def merge(self, other):
        for alg, wits in other._get_data().items():
            self._push(alg, *wits)


if __name__ == '__main__':
    dl1 = DataLoader()
    dl1._push('SVT', Witness(1, 1, 1, 1, 'dpfiner'))
    dl1._push('SVT', Witness(2, 2, 2, 2, 'dpfiner'))
    dl2 = DataLoader()
    dl2._push('SVT2', Witness(2, 2, 2, 2, 'dpstat'))
    dl2._push('SVT', Witness(3, 3, 3, 3, 'dpstat'))
    dl2._push('SVT', Witness(4, 4, 4, 4, 'dpfinder'))
    print(dl1._get_data())
    print(dl2._get_data())
    dl1.merge(dl2)
    print(dl1._get_data())
