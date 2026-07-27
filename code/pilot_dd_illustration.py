"""Illustration (for review, not yet in the paper): how an OTFS embedded pilot in
the delay-Doppler grid maps to the time-domain block.

Left  : the M x N delay-Doppler transmit grid. A single pilot P sits at (l_p, k_p)
        inside a null-guard band of +/- l_max in delay that spans ALL N Doppler
        bins (a full Doppler guard is needed because fractional Doppler leaks the
        pilot across the whole Doppler axis). Data fills the remaining delay rows.
Right : the time-domain block after the IFFT along Doppler, arranged as
        N slots x M samples. The pilot becomes a tone at delay-sample l_p present
        in EVERY slot (a comb), the guard rows are zero in every slot, and the
        data rows carry data in every slot. So each slot reads [data | 0..0 P 0..0
        | data] along delay -- the '0..0 P 0..0' pattern repeats across all N
        slots, mixed with data, rather than appearing as a single time impulse.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

M, N = 9, 7           # delay bins, Doppler bins / slots
lmax = 2
lp, kp = 4, 3         # pilot (delay, Doppler), centered
strip = set(range(lp - lmax, lp + lmax + 1))   # guarded delay band

C_DATA = '#bcd4e6'; C_GUARD = '#f0f0f0'; C_PILOT = '#e2892f'


def draw(ax, kind, title, xlabel, ylabel):
    for l in range(M):
        for c in range(N):
            if kind == 'dd':
                if l == lp and c == kp:   col, lab = C_PILOT, 'P'
                elif l in strip:          col, lab = C_GUARD, '0'
                else:                     col, lab = C_DATA, 'd'
            else:  # time block
                if l == lp:               col, lab = C_PILOT, 'P'
                elif l in strip:          col, lab = C_GUARD, '0'
                else:                     col, lab = C_DATA, 'd'
            ax.add_patch(Rectangle((c, l), 1, 1, facecolor=col, edgecolor='0.55', lw=0.6))
            fs = 8.5 if lab in ('P',) else 7.5
            ax.text(c + 0.5, l + 0.5, lab, ha='center', va='center', fontsize=fs,
                    color='0.35' if lab == 'd' else 'black')
    ax.set_xlim(-0.05, N + 0.05); ax.set_ylim(-0.05, M + 0.05)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.7))

# ---- left: DD grid ----
draw(axL, 'dd', 'Delay-Doppler transmit grid', 'Doppler  $k$', 'delay  $l$')
# delay-guard bracket (left of grid)
axL.annotate('', xy=(-0.55, lp - lmax), xytext=(-0.55, lp + lmax + 1),
             arrowprops=dict(arrowstyle='<->', color='0.15', lw=1.2))
axL.text(-0.9, lp + 0.5, r'delay guard $2l_{\max}$', rotation=90,
         ha='center', va='center', fontsize=9)
# full-Doppler guard bracket (below grid)
axL.annotate('', xy=(0, -0.4), xytext=(N, -0.4),
             arrowprops=dict(arrowstyle='<->', color='0.15', lw=1.2))
axL.text(N / 2, -0.62, 'null guard spans all $N$ Doppler bins (fractional Doppler)',
         ha='center', va='top', fontsize=8.5)

# ---- right: time block ----
draw(axR, 'time', 'Time-domain block:  $N$ slots $\\times$ $M$ samples',
     'slot  $n$', 'sample  $l$')
# highlight the pilot row
axR.add_patch(Rectangle((0, lp), N, 1, fill=False, edgecolor=C_PILOT, lw=2.2))
axR.text(N + 0.25, lp + 0.5, 'pilot = delay-$l_p$ tap\nin every slot (comb)',
         ha='left', va='center', fontsize=8.5, color=C_PILOT)

# arrow between panels, vertically centered on the grids
fig.subplots_adjust(left=0.06, right=0.84, top=0.90, bottom=0.11, wspace=0.55)
pL = axL.get_position(); pR = axR.get_position()
xmid = 0.5 * (pL.x1 + pR.x0); ymid = 0.5 * (pL.y0 + pL.y1)
fig.text(xmid, ymid + 0.075, 'IFFT along\nDoppler', ha='center', va='center', fontsize=9)
arr = FancyArrowPatch((xmid - 0.035, ymid), (xmid + 0.035, ymid), transform=fig.transFigure,
                      arrowstyle='-|>', mutation_scale=16, color='0.2', lw=1.4)
fig.add_artist(arr)

fig.subplots_adjust(left=0.06, right=0.84, top=0.90, bottom=0.11, wspace=0.55)
fig.savefig('pilot_dd_illustration.png', dpi=150, bbox_inches='tight')
print('Saved pilot_dd_illustration.png')
