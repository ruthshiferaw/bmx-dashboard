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

def style_axis(ax):
    ax.set_facecolor('#101827')
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(colors='#b9c8db', labelsize=10, length=0, pad=8)
    ax.xaxis.label.set_color('#cbd5e1')
    ax.yaxis.label.set_color('#cbd5e1')
    ax.grid(axis='y', color='#94a3b8', alpha=.13, linewidth=.8)
    ax.set_axisbelow(True)


def plot_segment_track_map(states, runs, segments):
    """Dark-mode map: terrain, common-contact windows and any-cyclist bridge zones."""
    first = states[CYCLISTS[0]]['geometry']
    terrain_x = np.asarray(first['x'], dtype=float)
    terrain_y = np.asarray(first['y'], dtype=float)
    if len(terrain_x) != len(terrain_y) or len(terrain_x) < 2:
        raise ValueError('Missing or invalid terrain geometry in dashboard data')
    raw = sorted((max(float(a), float(terrain_x[0])), min(float(b), float(terrain_x[-1])))
                 for state in states.values() for a, b in state['bridge_ranges'] if b > a)
    merged = []
    for a, b in raw:
        if not merged or a > merged[-1][1] + 1e-8: merged.append([a, b])
        else: merged[-1][1] = max(merged[-1][1], b)

    colors = ['#60a5fa', '#fbbf24', '#34d399', '#c084fc', '#22d3ee', '#fb7185']
    fig, ax = plt.subplots(figsize=(15, 5.6), facecolor='#080d17')
    fig.subplots_adjust(left=.075, right=.975, top=.82, bottom=.23)
    style_axis(ax)
    ax.plot(terrain_x, terrain_y, color='#b2c2d8', lw=2.6, zorder=5)
    for a,b in merged:
        ax.axvspan(a-CONTACT_BUFFER_M,b+CONTACT_BUFFER_M,color='#fb7185',alpha=.055,zorder=0)
        ax.axvspan(a,b,color='#fb7185',alpha=.23,zorder=1)
    for i,(a,b) in enumerate(segments):
        color=colors[i%len(colors)]
        ax.axvspan(a,b,color=color,alpha=.19,zorder=2)
        ax.axvline(a,color=color,alpha=.6,lw=.8,zorder=3)
        ax.axvline(b,color=color,alpha=.6,lw=.8,zorder=3)
        mid=(a+b)/2
        yy=float(np.interp(mid,terrain_x,terrain_y))
        ax.annotate(f'{i+1}',(mid,yy),xytext=(0,20),textcoords='offset points',
                    ha='center',va='bottom',fontsize=12,fontweight='bold',color='#080d17',
                    bbox=dict(boxstyle='round,pad=.35',facecolor=color,edgecolor='none'))
    ax.set_xlim(float(terrain_x[0]),float(terrain_x[-1]))
    ax.set_xlabel('TRACK POSITION  ·  m',labelpad=14,fontsize=10,fontweight='bold')
    ax.set_ylabel('TERRAIN HEIGHT  ·  m',labelpad=12,fontsize=10,fontweight='bold')
    fig.text(.075,.935,'COMMON-CONTACT SEGMENTS',color='#f1f5f9',fontsize=19,fontweight='bold')
    fig.text(.075,.884,'Shared track windows where all three cyclists remain in contact',color='#94a3b8',fontsize=11)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend=[Line2D([0],[0],color='#b2c2d8',lw=2.5,label='Terrain'),
            Patch(facecolor='#fb7185',alpha=.4,label='Non-contact: any cyclist'),
            Patch(facecolor='#fb7185',alpha=.12,label=f'Exclusion buffer ±{CONTACT_BUFFER_M:g} m')]
    legend += [Patch(facecolor=colors[i],alpha=.75,label=f'{i+1}  ·  {a:.1f}–{b:.1f} m')
               for i,(a,b) in enumerate(segments)]
    lg=fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,
                  frameon=False,fontsize=9,columnspacing=2,handlelength=1.7)
    for txt in lg.get_texts():txt.set_color('#cbd5e1')
    fig.savefig(OUTPUT/'contact_segment_track_map.png',dpi=220,facecolor=fig.get_facecolor())
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
    # Four coordinated dark panels. Each compares observed vs calculated
    # segment-average speeds for the same three cyclists; no order-match labels.
    fig,axes=plt.subplots(N_SEGMENTS,1,figsize=(12,3.05*N_SEGMENTS),
                          sharey=True,facecolor='#080d17')
    fig.subplots_adjust(left=.10,right=.96,top=.88,bottom=.075,hspace=.54)
    axes=np.atleast_1d(axes)
    labels=['P1','P2','P3']; x=np.arange(3); width=.30
    obs_color='#34d399';calc_color='#60a5fa'
    ymax=max(max(r['observed_mps'],r['calculated_mps']) for r in records)*1.18
    for i,ax in enumerate(axes,1):
        style_axis(ax)
        subset=[next(r for r in records if r['segment']==i and r['cyclist']==c) for c in CYCLISTS]
        observed=[r['observed_mps'] for r in subset]
        calculated=[r['calculated_mps'] for r in subset]
        leftbars=ax.bar(x-width/2,observed,width,color=obs_color,zorder=3)
        rightbars=ax.bar(x+width/2,calculated,width,color=calc_color,zorder=3)
        ax.set_ylim(0,ymax);ax.set_xlim(-.55,2.55)
        ax.set_xticks(x,labels,color='#e2e8f0',fontweight='bold',fontsize=11)
        ax.set_ylabel('SPEED  ·  m/s',fontsize=9,fontweight='bold',labelpad=13)
        x0,x1=segments[i-1]
        ax.text(0,1.12,f'SEGMENT {i}',transform=ax.transAxes,color='#f1f5f9',
                fontsize=13,fontweight='bold',va='bottom')
        ax.text(1,1.12,f'{x0:.1f}–{x1:.1f} m  ·  {x1-x0:.1f} m long',
                transform=ax.transAxes,color='#94a3b8',fontsize=10,ha='right',va='bottom')
        for bars in (leftbars,rightbars):
            for bar in bars:
                height=bar.get_height()
                ax.annotate(f'{height:.2f}',(bar.get_x()+bar.get_width()/2,height),
                            xytext=(0,5),textcoords='offset points',ha='center',va='bottom',
                            color='#e2e8f0',fontsize=9,fontweight='medium')
    from matplotlib.patches import Patch
    fig.text(.10,.965,'BMX  /  VELOCITY VALIDATION',color='#f1f5f9',fontsize=19,fontweight='bold')
    fig.text(.10,.933,'Observed vs calculated average speed · shared contact-only segments',
             color='#94a3b8',fontsize=11)
    leg=fig.legend(handles=[Patch(facecolor=obs_color,label='Observed'),
                            Patch(facecolor=calc_color,label='Calculated')],
                   loc='upper right',bbox_to_anchor=(.96,.972),frameon=False,ncol=2,fontsize=10)
    for txt in leg.get_texts():txt.set_color('#e2e8f0')
    fig.savefig(OUTPUT/'contact_segment_bars.png',dpi=220,facecolor=fig.get_facecolor())
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
