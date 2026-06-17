"""
report_generator.py
───────────────────
Generates a self-contained HTML session report from:
  • the CSV log  (dms_logs/session_*.csv)
  • session stats dict
  • incident list from IncidentRecorder

Usage:
    from report_generator import generate_report
    path = generate_report(csv_path, stats, incidents)
    # opens automatically in default browser
"""

import csv
import os
import json
import webbrowser
from datetime import datetime


# ── helpers ──────────────────────────────────────────────────────────────────
def _read_csv(path: str) -> tuple[list, list]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
    if not rows:
        return [], []
    headers = list(rows[0].keys())
    return headers, rows


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── main function ─────────────────────────────────────────────────────────────
def generate_report(csv_path: str, stats: dict, incidents: list) -> str:
    """
    Build report HTML, save to same folder as CSV, return path.
    stats keys expected:
        duration, blinks, yawns, long_closures, l1, l2, l3, neutral_pitch, ear_thr
    """
    headers, rows = _read_csv(csv_path)

    # ── extract time-series for charts ──────────────────────────────────────
    times, scores, ear_vals, perclos_vals = [], [], [], []
    alarm_events = []

    for i, r in enumerate(rows):
        times.append(r.get("time", "")[-12:-4])    # HH:MM:SS.mmm → last 12 chars
        scores.append(_safe_float(r.get("score", 0)))
        ear_vals.append(_safe_float(r.get("ear", 0)))
        perclos_vals.append(_safe_float(r.get("perclos", 0)) * 100)
        alv = r.get("alarm", "OK")
        if alv not in ("OK", ""):
            alarm_events.append({"x": i, "label": alv, "time": times[-1]})

    # downsample to max 400 points for chart performance
    step = max(1, len(times) // 400)
    times_s    = times[::step]
    scores_s   = scores[::step]
    ear_s      = ear_vals[::step]
    perclos_s  = perclos_vals[::step]

    # ── alarm counts bar data ────────────────────────────────────────────────
    l1 = stats.get("l1", 0)
    l2 = stats.get("l2", 0)
    l3 = stats.get("l3", 0)

    # ── incident table rows ──────────────────────────────────────────────────
    inc_rows = ""
    for inc in incidents:
        badge = {"L1":"#f0a500","L2":"#e06020","L3":"#cc2020"}.get(inc["level"],"#666")
        inc_rows += f"""
        <tr>
          <td>{inc['time']}</td>
          <td><span class="badge" style="background:{badge}">{inc['level']}</span></td>
          <td>{inc['reason']}</td>
          <td>{inc['duration']}</td>
          <td><a href="{os.path.abspath(inc['file'])}" target="_blank">▶ Open clip</a></td>
        </tr>"""
    if not inc_rows:
        inc_rows = "<tr><td colspan='5' style='text-align:center;color:#666'>No incidents recorded</td></tr>"

    # ── overall safety rating ────────────────────────────────────────────────
    total_alarms = l1 + l2 + l3
    if   l3 > 0 or l2 >= 5: rating, rating_col = "UNSAFE",  "#cc2020"
    elif l2 > 0 or l1 >= 3: rating, rating_col = "CAUTION", "#e06020"
    elif l1 > 0:             rating, rating_col = "FAIR",    "#f0a500"
    else:                    rating, rating_col = "SAFE",    "#22aa44"

    # ── build HTML ────────────────────────────────────────────────────────────
    report_time = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DMS Session Report — {report_time}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d0f14; --panel: #161a22; --border: #252b38;
    --text: #dce0ea; --dim: #7a8090; --accent: #4f9ef8;
    --ok: #22aa44; --warn: #e06020; --danger: #cc2020;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
          font-size: 14px; padding: 24px; }}
  h1 {{ font-size: 22px; color: var(--accent); margin-bottom: 4px; }}
  .subtitle {{ color: var(--dim); font-size: 12px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
           gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border);
           border-radius: 10px; padding: 16px; text-align: center; }}
  .card .val {{ font-size: 32px; font-weight: 700; line-height: 1.1; }}
  .card .lbl {{ font-size: 11px; color: var(--dim); margin-top: 4px; text-transform:uppercase;letter-spacing:.5px }}
  .rating-card {{ background: var(--panel); border: 2px solid {rating_col};
                  border-radius: 10px; padding: 16px; text-align: center; }}
  .rating-val {{ font-size: 36px; font-weight: 800; color: {rating_col}; }}
  section {{ background: var(--panel); border: 1px solid var(--border);
             border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  section h2 {{ font-size: 14px; color: var(--accent); margin-bottom: 16px;
                text-transform: uppercase; letter-spacing: 1px; }}
  .chart-wrap {{ position: relative; height: 220px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 10px; background:#1e232e;
        color: var(--dim); font-weight: 600; border-bottom: 1px solid var(--border); }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #1a1f2a; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px;
            font-weight:700; font-size:11px; color:#fff; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .footer {{ text-align:center; color:var(--dim); font-size:11px; margin-top:24px; }}
</style>
</head>
<body>

<h1>🚗 Driver Monitoring System — Session Report</h1>
<div class="subtitle">Generated: {report_time} &nbsp;|&nbsp; Log: {os.path.basename(csv_path)}</div>

<!-- KPI cards -->
<div class="grid">
  <div class="rating-card">
    <div class="rating-val">{rating}</div>
    <div class="lbl" style="color:var(--dim)">Safety Rating</div>
  </div>
  <div class="card">
    <div class="val" style="color:var(--accent)">{stats.get('duration','--')}</div>
    <div class="lbl">Session Duration</div>
  </div>
  <div class="card">
    <div class="val">{stats.get('blinks',0)}</div>
    <div class="lbl">Total Blinks</div>
  </div>
  <div class="card">
    <div class="val" style="color:#f0a500">{stats.get('yawns',0)}</div>
    <div class="lbl">Total Yawns</div>
  </div>
  <div class="card">
    <div class="val" style="color:#e06020">{stats.get('long_closures',0)}</div>
    <div class="lbl">Long Eye Closures</div>
  </div>
  <div class="card">
    <div class="val" style="color:#f0a500">{l1}</div>
    <div class="lbl">L1 Yawn Alerts</div>
  </div>
  <div class="card">
    <div class="val" style="color:#e06020">{l2}</div>
    <div class="lbl">L2 Warnings</div>
  </div>
  <div class="card">
    <div class="val" style="color:#cc2020">{l3}</div>
    <div class="lbl">L3 Critical</div>
  </div>
</div>

<!-- Drowsiness Score chart -->
<section>
  <h2>Drowsiness Score Over Time</h2>
  <div class="chart-wrap"><canvas id="scoreChart"></canvas></div>
</section>

<!-- EAR + PERCLOS chart -->
<section>
  <h2>EAR &amp; PERCLOS Over Time</h2>
  <div class="chart-wrap"><canvas id="earChart"></canvas></div>
</section>

<!-- Alert breakdown -->
<section>
  <h2>Alert Breakdown</h2>
  <div style="max-width:340px;height:200px;margin:auto"><canvas id="alertChart"></canvas></div>
</section>

<!-- Incident table -->
<section>
  <h2>Incident Recordings ({len(incidents)} clips)</h2>
  <table>
    <thead><tr><th>Time</th><th>Level</th><th>Reason</th><th>Duration</th><th>Clip</th></tr></thead>
    <tbody>{inc_rows}</tbody>
  </table>
</section>

<!-- Calibration info -->
<section>
  <h2>Driver Calibration</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Calibrated EAR Threshold</td><td>{stats.get('ear_thr', 'N/A')}</td></tr>
    <tr><td>Neutral Head Pitch</td><td>{stats.get('neutral_pitch', 'N/A')}°</td></tr>
    <tr><td>Total data points logged</td><td>{len(rows)}</td></tr>
  </table>
</section>

<div class="footer">Driver Monitoring System v3.0 &nbsp;·&nbsp; Built with Python · OpenCV · MediaPipe</div>

<script>
const labels = {json.dumps(times_s)};
const scores = {json.dumps(scores_s)};
const earVals = {json.dumps(ear_s)};
const perclosVals = {json.dumps(perclos_s)};

// gradient helper
function makeGrad(ctx, top, bot) {{
  const g = ctx.createLinearGradient(0,0,0,220);
  g.addColorStop(0, top); g.addColorStop(1, bot); return g;
}}

// ── Score chart ──────────────────────────────────────────────────────────
const sc = document.getElementById('scoreChart').getContext('2d');
new Chart(sc, {{
  type: 'line',
  data: {{
    labels,
    datasets: [{{
      label: 'Drowsiness Score',
      data: scores,
      borderColor: '#4f9ef8',
      backgroundColor: makeGrad(sc,'rgba(79,158,248,0.25)','rgba(79,158,248,0.02)'),
      fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color:'#dce0ea' }} }} }},
    scales: {{
      x: {{ ticks: {{ color:'#7a8090', maxTicksLimit:8 }}, grid: {{ color:'#252b38' }} }},
      y: {{ min:0, max:100, ticks: {{ color:'#7a8090' }}, grid: {{ color:'#252b38' }} }}
    }}
  }}
}});

// ── EAR + PERCLOS chart ──────────────────────────────────────────────────
const ec = document.getElementById('earChart').getContext('2d');
new Chart(ec, {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'EAR', data: earVals, borderColor:'#22aa44',
        backgroundColor:'rgba(34,170,68,0.12)', fill:true,
        tension:0.3, pointRadius:0, borderWidth:2, yAxisID:'y'
      }},
      {{
        label: 'PERCLOS %', data: perclosVals, borderColor:'#e06020',
        backgroundColor:'rgba(224,96,32,0.12)', fill:true,
        tension:0.3, pointRadius:0, borderWidth:2, yAxisID:'y2'
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color:'#dce0ea' }} }} }},
    scales: {{
      x:  {{ ticks: {{ color:'#7a8090', maxTicksLimit:8 }}, grid: {{ color:'#252b38' }} }},
      y:  {{ min:0, max:0.5, position:'left',  ticks: {{ color:'#22aa44' }}, grid: {{ color:'#252b38' }}, title:{{ display:true,text:'EAR',color:'#22aa44' }} }},
      y2: {{ min:0, max:100, position:'right', ticks: {{ color:'#e06020' }}, grid: {{ drawOnChartArea:false }}, title:{{ display:true,text:'PERCLOS %',color:'#e06020' }} }}
    }}
  }}
}});

// ── Alert donut chart ────────────────────────────────────────────────────
const ac = document.getElementById('alertChart').getContext('2d');
new Chart(ac, {{
  type: 'doughnut',
  data: {{
    labels: ['L1 Yawn ({l1})', 'L2 Warning ({l2})', 'L3 Critical ({l3})'],
    datasets: [{{
      data: [{l1}, {l2}, {l3}],
      backgroundColor: ['#f0a500','#e06020','#cc2020'],
      borderColor: '#161a22', borderWidth: 3,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position:'right', labels: {{ color:'#dce0ea', padding:16 }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    out_path = csv_path.replace(".csv", "_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[REPORT] Saved → {out_path}")
    try:
        webbrowser.open(f"file:///{os.path.abspath(out_path)}")
        print("[REPORT] Opened in browser.")
    except Exception:
        pass

    return out_path