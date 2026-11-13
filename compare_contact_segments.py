"""Compare P1/P2/P3 on identical contact-only BMX track segments.

Run from the folder containing index.html and bmx_outputs/: 
    pip install playwright numpy matplotlib
    python -m playwright install chromium
    python compare_contact_segments.py

Uses the dashboard's OWN JavaScript simulation, so its physics and video bridge
match the website. Outputs PNG + CSVs in contact_segment_results/.
"""
from __future__ import annotations
import csv
import functools
import http.server
import math
import threading
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

# CONFIGURATION
ROOT = Path(__file__).resolve().parent  # put this file beside index.html
HTML = 'index.html'                    # change if your website HTML has another name
CYCLISTS = ('sleeman_p1', 'sleeman_p2', 'sleeman_p3')
N_SEGMENTS = 4
GRID_STEP_M = 0.25                   # candidate boundaries in track coordinates
MIN_SEGMENT_M = 3.0                  # minimum usable contact stretch
MAX_SEGMENT_M = 20.0                 # length of each selected comparison window
CONTACT_BUFFER_M = 0.35             # exclude area near any non-contact event
OUTPUT = ROOT / 'contact_segment_results'

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args): pass

def read_dashboard_data():
    if not (ROOT / HTML).is_file():
        raise FileNotFoundError(f'Missing {HTML} in {ROOT}')
    for cyclist in CYCLISTS:
        file = ROOT / 'bmx_outputs' / cyclist / 'bmx_dashboard_data.json'
        if not file.is_file(): raise FileNotFoundError(f'Missing {file}')
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    result = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f'http://127.0.0.1:{server.server_port}/{HTML}', wait_until='domcontentloaded')
            for cyclist in CYCLISTS:
                # Calling init directly avoids UI differences between dashboard variants.
                page.evaluate('''async (name) => {
                    document.getElementById('cyclistName').value = name;
                    await init();
                }''', cyclist)
                state = page.evaluate('''() => ({
                    current: current.map(r=>({x:r.x,t:r.t,speed:r.speed,bridge:!!r.video_bridge})),
                    observed: videoVelocityEstimate.map(r=>({x:r.x,v:r.speed})),
                    origin: data.roller_start_x,
                    intervals: data.measured_intervals,
                    geometry: {x: data.geometry.x, y: data.geometry.y},
                    status: document.getElementById('status').textContent
                })''')
                if len(state['current']) < 2 or len(state['observed']) < 2:
                    raise RuntimeError(f'No trajectory or observed curve for {cyclist}: {state["status"]}')
                result[cyclist] = state
                print(f'{cyclist}: {len(state["current"])} simulated samples, '
                      f'{len(state["observed"])} observed-curve samples')
            browser.close()
    finally:
        server.shutdown()
    return result

def prepare(state):
    sim = state['current']
    state['sim_x'] = np.array([r['x'] for r in sim], dtype=float)
    state['sim_t'] = np.array([r['t'] for r in sim], dtype=float)
    state['obs_x'] = np.array([r['x'] for r in state['observed']], dtype=float)
    state['obs_v'] = np.array([r['v'] for r in state['observed']], dtype=float)
    # The observed velocity is dx/dt, so segment elapsed time = integral dx/v(x).
    # Numerical integration on the dashboard's dense observed curve.
    if np.any(np.diff(state['obs_x']) <= 0):
        raise ValueError('Observed position samples are not strictly increasing')
    if np.any(state['obs_v'] <= 0):
        raise ValueError('Observed velocity must be positive for time integration')
    ox, ov = state['obs_x'], state['obs_v']
    state['obs_cum_t'] = np.r_[0, np.cumsum(np.diff(ox)*0.5*(1/ov[:-1]+1/ov[1:]))]
    # Bridge segments: exclude any Euler step with either endpoint bridged.
    state['bridge_ranges'] = [(sim[i]['x'],sim[i+1]['x'])
        for i in range(len(sim)-1) if sim[i]['bridge'] or sim[i+1]['bridge']]

def interval_mean(state, x0, x1, observed):
    if observed:
        x, t = state['obs_x'], state['obs_cum_t']
    else:
        x, t = state['sim_x'], state['sim_t']
    if x0 < x[0]-1e-7 or x1 > x[-1]+1e-7: return math.nan
    duration = float(np.interp(x1,x,t)-np.interp(x0,x,t))
    return (x1-x0)/duration if duration > 0 else math.nan

def clean_contact(state, x0, x1):
    for a,b in state['bridge_ranges']:
        if a < x1+CONTACT_BUFFER_M and b > x0-CONTACT_BUFFER_M:
            return False
    return True

def common_runs(states):
    lower = max(max(s['sim_x'][0],s['obs_x'][0]) for s in states.values())
    upper = min(min(s['sim_x'][-1],s['obs_x'][-1]) for s in states.values())
    # Same track coordinates for all three cyclists.
    xs = np.arange(math.ceil(lower/GRID_STEP_M)*GRID_STEP_M,
                   upper+1e-8, GRID_STEP_M)
    good = np.array([all(clean_contact(s,x,x) for s in states.values()) for x in xs])
    runs=[]; start=None
    for i,ok in enumerate(good):
        if ok and start is None: start=i
        if start is not None and (not ok or i==len(good)-1):
            end=i if ok else i-1
            if xs[end]-xs[start] >= MIN_SEGMENT_M:
                runs.append((float(xs[start]),float(xs[end])))
            start=None
    return runs

def choose_segments(runs):
    # Divide available contact-only distance into four equal-distance portions.
    # Choose one window in each portion, favoring longer clean runs, so four
    # neighboring windows don't misleadingly count as four independent regions.
    candidates=[]
    for a,b in runs:
        length=min(MAX_SEGMENT_M,b-a)
        if length < MIN_SEGMENT_M: continue
        # candidate windows slide within clean runs
        starts=np.arange(a,b-length+1e-7,GRID_STEP_M)
        if not len(starts): starts=np.array([a])
        for st in starts:
            candidates.append((float(st),float(st+length)))
    if not candidates: raise ValueError('No shared contact-only windows meet the minimum length')
    # Greedy: choose spatially separated windows, preferring broad track coverage.
    selected=[]
    for target in np.linspace(min(a for a,b in candidates), max(b for a,b in candidates), N_SEGMENTS+2)[1:-1]:
        options=[z for z in candidates if all(z[1]<=q[0] or z[0]>=q[1] for q in selected)]
        if not options: break
        selected.append(min(options,key=lambda z:(abs((z[0]+z[1])/2-target),-(z[1]-z[0]))))
    if len(selected)<N_SEGMENTS:
        for z in sorted(candidates,key=lambda z:-(z[1]-z[0])):
            if all(z[1]<=q[0] or z[0]>=q[1] for q in selected):
                selected.append(z)
                if len(selected)==N_SEGMENTS: break
    selected.sort()
    if len(selected)<N_SEGMENTS:
        raise ValueError(f'Only {len(selected)} nonoverlapping common-contact segments available; '
                         f'need {N_SEGMENTS}. Try reducing MIN_SEGMENT_M or CONTACT_BUFFER_M.')
    return selected

def plot_segment_track_map(states, runs, segments):
    """Show the shared terrain, chosen windows, and union of video-bridged zones."""
    first = states[CYCLISTS[0]]['geometry']
    terrain_x = np.asarray(first['x'], dtype=float)
    terrain_y = np.asarray(first['y'], dtype=float)
    if len(terrain_x) != len(terrain_y) or len(terrain_x) < 2:
        raise ValueError('Missing or invalid terrain geometry in dashboard data')

    # Merge all bridged integration intervals across P1/P2/P3. Red indicates
    # a non-contact/video-bridge event for ANY cyclist, not necessarily all.
    raw = sorted((max(float(a), float(terrain_x[0])), min(float(b), float(terrain_x[-1])))
                 for state in states.values() for a, b in state['bridge_ranges'] if b > a)
    merged = []
    for a, b in raw:
        if not merged or a > merged[-1][1] + 1e-8:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    fig, ax = plt.subplots(figsize=(13, 4.7), layout='constrained')
    ax.plot(terrain_x, terrain_y, color='#64748b', lw=2.3, label='Track terrain', zorder=2)
    for i, (a, b) in enumerate(merged):
        ax.axvspan(a, b, color='#ef4444', alpha=.22, zorder=0,
                   label='Non-contact: at least one cyclist' if i == 0 else None)
    # The buffer is a selection exclusion margin; show it separately so the
    # plot does not imply these extra meters were themselves non-contact.
    for i, (a, b) in enumerate(merged):
        ax.axvspan(a-CONTACT_BUFFER_M, b+CONTACT_BUFFER_M,
                   color='#ef4444', alpha=.07, zorder=0,
                   label=f'Contact exclusion buffer (±{CONTACT_BUFFER_M:g} m)' if i == 0 else None)

    colors = ['#2563eb', '#d97706', '#059669', '#9333ea', '#0891b2', '#be185d']
    for i, (a, b) in enumerate(segments):
        color = colors[i % len(colors)]
        ax.axvspan(a, b, color=color, alpha=.24, zorder=1,
                   label=f'Segment {i+1}: {a:.1f}–{b:.1f} m')
        mid = (a+b)/2
        yy = float(np.interp(mid, terrain_x, terrain_y))
        ax.annotate(str(i+1), (mid, yy), xytext=(0, 15), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=.22', facecolor='white', edgecolor=color, alpha=.95))
    ax.set(xlabel='Track horizontal position x (m)', ylabel='Terrain height (m)',
           title='Selected common-contact segments on the BMX track')
    ax.set_xlim(float(terrain_x[0]), float(terrain_x[-1]))
    ax.grid(alpha=.17)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, fontsize=9)
    fig.savefig(OUTPUT/'contact_segment_track_map.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    states=read_dashboard_data()
    for s in states.values(): prepare(s)
    runs=common_runs(states)
    print('Common contact-only runs (track x, m):', runs)
    segments=choose_segments(runs)
    OUTPUT.mkdir(exist_ok=True)
    plot_segment_track_map(states, runs, segments)
    records=[]; ranking=[]
    for k,(x0,x1) in enumerate(segments,1):
        calc={};obs={}
        for cyclist,s in states.items():
            vc=interval_mean(s,x0,x1,False)
            vo=interval_mean(s,x0,x1,True)
            if not np.isfinite(vc) or not np.isfinite(vo):
                raise ValueError(f'Invalid velocity for {cyclist} in segment {k}')
            calc[cyclist]=vc;obs[cyclist]=vo
            records.append(dict(segment=k,start_x_m=x0,end_x_m=x1,
                cyclist=cyclist,observed_mps=vo,calculated_mps=vc,error_mps=vc-vo))
        observed_order=sorted(CYCLISTS,key=lambda c:-obs[c]);calculated_order=sorted(CYCLISTS,key=lambda c:-calc[c])
        pairs=[(a,b) for i,a in enumerate(CYCLISTS) for b in CYCLISTS[i+1:]]
        correct=sum((obs[a]-obs[b])*(calc[a]-calc[b])>0 for a,b in pairs)
        ranking.append(dict(segment=k,start_x_m=x0,end_x_m=x1,
            observed_order=' > '.join(c.upper().replace('SLEEMAN_','') for c in observed_order),
            calculated_order=' > '.join(c.upper().replace('SLEEMAN_','') for c in calculated_order),
            full_order_match=observed_order==calculated_order,
            pairwise_correct=correct,pairwise_total=len(pairs)))
    for filename,rows in [('contact_segment_velocities.csv',records),('contact_segment_rankings.csv',ranking)]:
        with (OUTPUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    # Four panels, each with three cyclists and two bars per cyclist.
    fig,axes=plt.subplots(N_SEGMENTS,1,figsize=(10,3.0*N_SEGMENTS),sharey=True,layout='constrained')
    axes=np.atleast_1d(axes)
    labels=['P1','P2','P3']; x=np.arange(3); width=.34
    for i,(ax,rank) in enumerate(zip(axes,ranking),1):
        subset=[next(r for r in records if r['segment']==i and r['cyclist']==c) for c in CYCLISTS]
        observed=[r['observed_mps'] for r in subset]; calculated=[r['calculated_mps'] for r in subset]
        ax.bar(x-width/2,observed,width,label='Observed',color='#6fe2b0')
        ax.bar(x+width/2,calculated,width,label='Calculated',color='#59b5ff')
        ax.set_xticks(x,labels);ax.set_ylabel('Mean speed (m/s)')
        ax.set_title(f'Segment {i}: x = {rank["start_x_m"]:.2f}–{rank["end_x_m"]:.2f} m  |  '
                     f'Order match: {"YES" if rank["full_order_match"] else "NO"}  '
                     f'({rank["pairwise_correct"]}/3 pairs)',loc='left',fontsize=11)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc='upper right')
    fig.suptitle('BMX velocity validation — common contact-only track segments',fontsize=15)
    fig.savefig(OUTPUT/'contact_segment_bars.png',dpi=220)
    plt.close(fig)
    print('\nSegment ranking results:')
    for r in ranking:
        print(f'  {r["segment"]}: {r["start_x_m"]:.2f}–{r["end_x_m"]:.2f} m | '
              f'observed {r["observed_order"]} | calculated {r["calculated_order"]} | '
              f'{r["pairwise_correct"]}/3 pairwise correct')
    print(f'Full order matches: {sum(r["full_order_match"] for r in ranking)}/{len(ranking)}')
    print(f'Pairwise correct: {sum(r["pairwise_correct"] for r in ranking)}/{3*len(ranking)}')
    print('Saved track map: ', OUTPUT/'contact_segment_track_map.png')
    print('Saved outputs to', OUTPUT)

if __name__=='__main__': main()
