"""Unit-average-power square-QAM wrapper around commpy's QAMModem."""

import numpy as np
from commpy.modulation import QAMModem


class QAM:
    def __init__(self, order):
        self.mod = QAMModem(order)
        self.bits_per_symbol = self.mod.num_bits_symbol
        # power-normalization so E[|s|^2] = 1
        self._scale = np.sqrt(np.mean(np.abs(self.mod.constellation) ** 2))

    def modulate(self, bits):
        return self.mod.modulate(np.asarray(bits, dtype=int)) / self._scale

    def demodulate_hard(self, symbols):
        return self.mod.demodulate(np.asarray(symbols) * self._scale, 'hard')

    def num_symbols(self, num_bits):
        return num_bits // self.bits_per_symbol
