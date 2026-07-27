"""Constraint-length-7 convolutional coding (rate 1/2, generators 171/133 oct)
with hard-decision Viterbi decoding, mirroring the inner code of the reference
FANET coherent modem (poly2trellis(7,[171 133])).

Scope note: this is the CC-only first step. The Reed-Solomon outer layer of the
full concatenated RS+CC modem can be layered on top later via, e.g., the
`galois` package; the interfaces here (encode/decode on bit arrays) are chosen
so that an outer code slots in without changing the waveform code.
"""

import numpy as np
from commpy.channelcoding import Trellis, conv_encode, viterbi_decode

# constraint length 7 -> memory 6; standard rate-1/2 generators (octal 171,133)
_MEM = np.array([6])
_G = np.array([[0o171, 0o133]])
_TRELLIS = Trellis(_MEM, _G)
_TB_DEPTH = 5 * (int(_MEM[0]) + 1)  # ~35, standard traceback rule of thumb


class ConvCode:
    """Rate-1/2 K=7 convolutional code, hard-decision Viterbi."""

    rate = 0.5

    def encode(self, bits):
        bits = np.asarray(bits, dtype=int)
        return conv_encode(bits, _TRELLIS)

    def decode(self, coded_bits):
        dec = viterbi_decode(np.asarray(coded_bits, dtype=float), _TRELLIS,
                             tb_depth=_TB_DEPTH, decoding_type='hard')
        return dec

    def info_len_for(self, num_coded_bits):
        """Number of information bits that produce ~num_coded_bits coded bits.

        conv_encode appends a zero tail of length `memory`; account for it so
        the encoded stream matches the requested frame payload closely.
        """
        return max(1, num_coded_bits // 2 - int(_MEM[0]))
