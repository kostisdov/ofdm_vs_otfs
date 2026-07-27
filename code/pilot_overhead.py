"""Schematic of time-domain / delay-axis pilot overhead (no simulation).

A pilot impulse needs a guard of the channel-impulse-response length L=l_max+1 for
its own delay spread, plus a further ~L to isolate it from data. So an OFDM block
with a separate time-domain pilot (top) and an OTFS block with an embedded
delay-Doppler pilot (bottom) both spend about 2L resource elements around the
pilot. This is the figure form of Table III: an OTFS block carries the same pilot
overhead as the OFDM block.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

L = 4                      # l_max + 1, illustrative
C_DATA = '#bcd4e6'; C_ZERO = '#f4f4f4'; C_PILOT = '#e2892f'; C_CP = '#c9b6d6'

fig, ax = plt.subplots(figsize=(9.4, 3.0))
ax.set_xlim(-9, 23); ax.set_ylim(0.4, 7.8); ax.axis('off')
H = 0.85


def cell(x, y, w, color, label='', fs=8.5):
    ax.add_patch(Rectangle((x, y), w, H, facecolor=color, edgecolor='0.35', lw=0.8))
    if label:
        ax.text(x + w / 2, y + H / 2, label, ha='center', va='center', fontsize=fs)


def zeros(x, y, n):
    for i in range(n):
        cell(x + i, y, 1, C_ZERO, '0', fs=7.5)
    return x + n


def cpblock(x, y):
    cell(x, y, 2.5, C_CP, 'CP', fs=8.5)
    return x + 2.5


def bracket(x0, x1, y, text):
    ax.plot([x0, x0, x1, x1], [y + 0.12, y, y, y + 0.12], color='0.15', lw=1.1)
    ax.text((x0 + x1) / 2, y - 0.22, text, ha='center', va='top', fontsize=9)


def rowlabel(y, text):
    ax.text(-8.7, y + H / 2, text, ha='left', va='center', fontsize=9)


# ---- Row 1: OFDM block, separate TD pilot  (~2L) ----
y = 5
rowlabel(y, 'OFDM block,\nseparate TD pilot')
x = cpblock(0, y)
cell(x, y, 6, C_DATA, 'data'); x += 6
gx = x; x = zeros(x, y, L)                     # guard for pilot spread
cell(x, y, 1, C_PILOT, 'P'); x += 1; x = zeros(x, y, L - 1)   # sounding window
bracket(gx, x, y - 0.15, r'pilot guard ($L$) + sounding ($L$) $\approx 2L$')
cell(x, y, 6, C_DATA, 'data')

# ---- Row 2: OTFS block, embedded DD pilot  (~2L, same) ----
y = 2
rowlabel(y, 'OTFS block,\nembedded DD pilot')
x = cpblock(0, y)
cell(x, y, 6, C_DATA, 'data'); x += 6
gx = x; x = zeros(x, y, L - 1)                 # delay guard (spread)
cell(x, y, 1, C_PILOT, 'P'); x += 1
x = zeros(x, y, L)                             # delay guard (isolation)
bracket(gx, x, y - 0.15, r'delay guard $\approx 2l_{\max} \approx 2L$  (same)')
cell(x, y, 6, C_DATA, 'data')

# legend
handles = [Rectangle((0, 0), 1, 1, fc=C_CP, ec='0.35'),
           Rectangle((0, 0), 1, 1, fc=C_DATA, ec='0.35'),
           Rectangle((0, 0), 1, 1, fc=C_ZERO, ec='0.35'),
           Rectangle((0, 0), 1, 1, fc=C_PILOT, ec='0.35')]
ax.legend(handles, ['cyclic prefix', 'data', 'null guard (0)', 'pilot impulse (P)'],
          loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False, fontsize=8.5)

fig.tight_layout()
fig.savefig('pilot_overhead.png', dpi=150, bbox_inches='tight')
print('Saved pilot_overhead.png')
