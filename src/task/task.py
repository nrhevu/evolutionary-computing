import os
from abc import abstractmethod

from src.representation import Representation


class Task:

    dimension: int
    capacity: int
    task_name: str
    representation: Representation

    def __init__(self, filename, representation):
        self.task_name = os.path.basename(filename).split(".")[0]
        self.representation = representation

    def get_stride(self, tx: list, tmp_tx: list, window):
        stride = int()
        if len(tmp_tx) - len(window) % (self.dimension - 1) == 0:
            stride = (len(tmp_tx) - len(window)) % (self.dimension - 1)
        else:
            stride = (len(tmp_tx) - len(window)) / (self.dimension - 1) + 1
            zero_padding = (self.dimension - 1) * stride + len(window) - len(tx)
            for i in range(zero_padding):
                tmp_tx.append(0.0)

    def __repr__(self) -> str:
        return "Task name : " + self.task_name

    @abstractmethod
    def compute_fitness():
        pass

    @abstractmethod
    def get_len_gene():
        pass

    @abstractmethod
    def show_result():
        pass
