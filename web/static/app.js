// ── State ──
let allRuns = [];
let currentModalRunId = null;
let charts = {};
let activeTab = localStorage.getItem('dash_tab') || 'dashboard';
let metricsCache = {};
let selectedExp = localStorage.getItem('dash_exp') || null;
let selectedRunId = localStorage.getItem('dash_run') || null;
let backlogItems = [];
let metricsXAxis = 'epoch'; // 'epoch' | 'time'
let _expDragging = false;
const _expResizeHandle = document.createElement('div');
_expResizeHandle.className = 'exp-resize';

// ── Theme ──
const CHART_TICK = '#4a5060';
const CHART_GRID = '#e1e4e822';
const CHART_LEGEND = '#4a5060';
const COLORS = ['#0969da','#1a7f37','#9a6700','#cf222e','#8250df','#0891b2','#bc4c00','#0550ae'];

// ── Helpers ──
function ago(ts) {
  if (!ts) return '?';
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 60) return Math.floor(s) + 's';
  if (s < 3600) return Math.floor(s/60) + 'm';
  if (s < 86400) return Math.floor(s/3600) + 'h';
  return Math.floor(s/86400) + 'd';
}

function fmtVal(v) {
  if (typeof v !== 'number') return String(v);
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  if (Math.abs(v) >= 0.001) return v.toFixed(4);
  return v.toExponential(2);
}

function statusHtml(status, alive) {
  if (status === 'running' && !alive)
    return '<span class="status status-stale">stale</span>';
  const cls = {done:'status-done', running:'status-running', failed:'status-failed'}[status] || '';
  return `<span class="status ${cls}">${status}</span>`;
}

function statusDot(status, alive) {
  if (status === 'running' && !alive) return '<span class="exp-run-dot stale"></span>';
  const cls = {done:'done', running:'running', failed:'failed'}[status] || 'stale';
  return `<span class="exp-run-dot ${cls}"></span>`;
}

function calcETA(run) {
  const m = run.latest;
  if (!m || !run.info.config?.epochs) return null;
  const epoch = m.epoch;
  const total = run.info.config.epochs;
  if (epoch == null || epoch <= 0 || epoch >= total) return null;
  const started = run.info.started_at ? new Date(run.info.started_at).getTime() : null;
  if (!started) return null;
  const elapsed = (Date.now() - started) / 1000;
  const perEpoch = elapsed / (epoch + 1);
  const remaining = perEpoch * (total - epoch - 1);
  if (remaining < 60) return Math.floor(remaining) + 's';
  if (remaining < 3600) return Math.floor(remaining / 60) + 'm';
  return (remaining / 3600).toFixed(1) + 'h';
}

// ── Chart helpers ──
function chartScales(xTitle) {
  const scales = {
    x: { ticks: { color: CHART_TICK, font: { size: 11 } }, grid: { color: CHART_GRID } },
    y: { ticks: { color: CHART_TICK, font: { size: 11 } }, grid: { color: CHART_GRID } }
  };
  if (xTitle) scales.x.title = { display: true, text: xTitle, color: CHART_TICK, font: { size: 12, weight: '500' } };
  return scales;
}

function chartScalesXY(xTitle, yTitle) {
  const s = chartScales(xTitle);
  if (yTitle) s.y.title = { display: true, text: yTitle, color: CHART_TICK, font: { size: 12, weight: '500' } };
  return s;
}

function chartLegend() {
  return { labels: { color: CHART_LEGEND, font: { size: 11 }, usePointStyle: true, pointStyle: 'circle', padding: 16 } };
}

function baseChartOpts() {
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: 'nearest', intersect: true },
    plugins: {
      legend: chartLegend(),
      tooltip: { backgroundColor: '#1f2328ee', titleFont: { size: 12 }, bodyFont: { size: 11 }, padding: 10, cornerRadius: 6 },
      zoom: {
        pan: { enabled: true, mode: 'x' },
        zoom: { drag: { enabled: true, backgroundColor: 'rgba(9,105,218,0.08)', borderColor: 'var(--blue)', borderWidth: 1 }, mode: 'x' },
      },
    },
  };
}

// Metric grouping: keys sharing a unit go on the same chart
const METRIC_UNITS = {
  loss: { pattern: /loss/i, unit: 'Loss' },
  nmse_db: { pattern: /nmse.*db|db.*nmse/i, unit: 'NMSE (dB)' },
  db: { pattern: /db/i, unit: 'dB' },
  acc: { pattern: /acc/i, unit: 'Accuracy' },
  lr: { pattern: /^lr$|learning.rate/i, unit: 'Learning Rate' },
  snr: { pattern: /snr/i, unit: 'SNR (dB)' },
  pct: { pattern: /pct|percent/i, unit: '%' },
};

function _metricUnit(key) {
  for (const [, v] of Object.entries(METRIC_UNITS)) {
    if (v.pattern.test(key)) return v.unit;
  }
  return key; // fallback: use key name as unit
}

function groupMetricKeys(keys) {
  const chartGroups = [];
  const keyList = [...keys];
  const used = new Set();
  for (const k of keyList) {
    if (used.has(k)) continue;
    const unit = _metricUnit(k);
    const group = [k]; used.add(k);
    for (const k2 of keyList) {
      if (used.has(k2)) continue;
      if (_metricUnit(k2) === unit) { group.push(k2); used.add(k2); }
    }
    chartGroups.push(group);
  }
  return chartGroups;
}

function _groupYTitle(group, runMetricUnits) {
  // 1. Check explicit metric_units from tracker (e.g. {"val_nmse_db": "NMSE (dB)"})
  if (runMetricUnits) {
    for (const k of group) {
      if (runMetricUnits[k]) return runMetricUnits[k];
    }
  }
  // 2. Auto-detect from key name patterns
  const units = [...new Set(group.map(k => _metricUnit(k)))];
  if (units.length === 1 && units[0] !== group[0]) return units[0];
  if (group.length === 1) return group[0];
  return units[0];
}

// ── Persist UI state ──
const _st = key => localStorage.getItem('dash_' + key);
const _ss = (key, val) => localStorage.setItem('dash_' + key, val);

// ── Sidebar toggle ──
function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  sb.classList.toggle('collapsed');
  const collapsed = sb.classList.contains('collapsed');
  document.getElementById('sidebar-toggle').textContent = collapsed ? '›' : '‹';
  _ss('sidebar', collapsed ? '1' : '0');
}

// Restore sidebar state
if (_st('sidebar') === '1') {
  document.getElementById('sidebar').classList.add('collapsed');
  document.getElementById('sidebar-toggle').textContent = '›';
}

// ── Right sidebar (Tasks) toggle ──
function toggleRightSidebar() {
  const sb = document.getElementById('right-sidebar');
  sb.classList.toggle('collapsed');
  const collapsed = sb.classList.contains('collapsed');
  document.getElementById('rsb-toggle').textContent = collapsed ? '‹' : '›';
  _ss('rsb', collapsed ? '1' : '0');
}

// Restore right sidebar state
if (_st('rsb') === '1') {
  document.getElementById('right-sidebar').classList.add('collapsed');
  document.getElementById('rsb-toggle').textContent = '‹';
}

function renderRightSidebar() {
  const inProgress = backlogItems.filter(b => b.status === 'in_progress');
  const queued = backlogItems.filter(b => b.status === 'queued');
  const done = backlogItems.filter(b => b.status === 'done')
    .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || '')).slice(0, 6);
  const failed = backlogItems.filter(b => b.status === 'failed')
    .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || '')).slice(0, 4);

  const itemHtml = (b) => {
    const type = b.type || 'experiment';
    const clr = TYPE_COLORS[type] || 'var(--text-tertiary)';
    const icon = TYPE_ICONS[type] || '◆';
    const session = b.session ? `<span class="rsb-session">${b.session}</span>` : '';
    return `<div class="rsb-item ${b.status}">
      <div class="rsb-item-name">${b.name}</div>
      ${b.description ? `<div class="rsb-item-desc">${b.description}</div>` : ''}
      <div class="rsb-item-meta">
        <span class="rsb-type" style="color:${clr}">${icon} ${type}</span>
        ${session}
        <span>${ago(b.created_at)}</span>
      </div>
      ${b.summary ? `<div class="rsb-item-desc" style="color:var(--green);margin-top:3px">${b.summary}</div>` : ''}
    </div>`;
  };

  const sectionHtml = (label, items, emptyMsg) => {
    if (!items.length && !emptyMsg) return '';
    return `<div class="rsb-section-label">${label}<span style="opacity:0.5;margin-left:4px">${items.length}</span></div>
      ${items.length ? items.map(itemHtml).join('') : `<div style="color:var(--text-tertiary);font-size:11px;padding:4px 4px 8px">${emptyMsg}</div>`}`;
  };

  let html = '';
  html += sectionHtml('In Progress', inProgress, 'None');
  if (queued.length) html += sectionHtml('Queued', queued);
  if (done.length) html += sectionHtml('Done', done);
  if (failed.length) html += sectionHtml('Failed', failed);

  document.getElementById('rsb-active').innerHTML = html;
  // Hide the old "recent" section (now merged above)
  document.getElementById('rsb-recent-label').style.display = 'none';
  document.getElementById('rsb-recent').innerHTML = '';

  // Mini sidebar (collapsed) — show dots for active tasks
  const miniEl = document.getElementById('rsb-mini');
  let miniHtml = '';
  inProgress.forEach(b => { miniHtml += `<div class="rsb-mini-dot" style="background:var(--blue)" title="${b.name}"></div>`; });
  queued.forEach(b => { miniHtml += `<div class="rsb-mini-dot" style="background:var(--yellow)" title="${b.name}"></div>`; });
  if (!miniHtml) miniHtml = '<div style="font-size:8px;color:var(--text-tertiary)">0</div>';
  miniEl.innerHTML = miniHtml;
}

// ── Tab system ──
function switchToTab(tabName) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  activeTab = tabName;
  _ss('tab', tabName);
  const sec = document.getElementById('sec-' + activeTab);
  if (sec) sec.classList.add('active');
  if (activeTab === 'experiments') renderExpList();
}

document.getElementById('tabs').addEventListener('click', e => {
  const tab = e.target.closest('.tab');
  if (!tab) return;
  switchToTab(tab.dataset.tab);
});

// Restore tab state
{ const saved = _st('tab'); if (saved) switchToTab(saved); }

// ════════════════════════════════════════════════
// DASHBOARD TAB — Compact unified status list
// ════════════════════════════════════════════════

async function renderDashboard() {
  const el = document.getElementById('dash-list');

  // 1. Running experiments (with live progress)
  const running = allRuns.filter(r => r.info.status === 'running' && r.alive);
  // 2. In-progress tasks from backlog
  const inProgress = backlogItems.filter(b => b.status === 'in_progress');
  // 3. Recent completed/failed (runs + tasks)
  const recent = allRuns.filter(r => r.info.status !== 'running')
    .sort((a, b) => b.mtime - a.mtime).slice(0, 10);
  const stale = allRuns.filter(r => r.info.status === 'running' && !r.alive);

  let rows = '';

  // Running experiments — compact with inline progress
  for (const r of running) {
    const m = r.latest || {};
    const epoch = m.epoch, total = r.info.config?.epochs;
    let prog = '';
    if (epoch != null && total) {
      const pct = Math.min(100, ((epoch+1)/total)*100);
      prog = `<div class="dash-prog"><div class="dash-prog-fill" style="width:${pct}%"></div></div><span class="dash-pct">${pct.toFixed(0)}%</span>`;
    }
    const metrics = Object.entries(m)
      .filter(([k]) => !['t','step','epoch'].includes(k) && typeof m[k] === 'number')
      .slice(0, 3).map(([k,v]) => `<span class="dash-metric">${k} <b>${fmtVal(v)}</b></span>`).join('');
    const eta = calcETA(r);
    const etaStr = eta ? `<span class="dash-metric" style="color:var(--cyan)">ETA <b>${eta}</b></span>` : '';
    const gitTag = r.info.git_commit ? `<span class="dash-metric" style="opacity:0.6"><code>${r.info.git_commit}</code></span>` : '';
    rows += `<div class="dash-row running" onclick="openRunDetail('${r.info.id}')">
      <span class="dash-status-dot running"></span>
      <span class="dash-name">${r.info.name}</span>
      <span class="dash-detail">${metrics}${etaStr}${gitTag}</span>
      <span class="dash-progress">${prog}</span>
      <span class="dash-time">${ago(r.info.started_at)}</span>
    </div>`;
  }

  // In-progress tasks (non-experiment)
  for (const b of inProgress) {
    const type = b.type || 'experiment';
    const clr = TYPE_COLORS[type] || 'var(--text-tertiary)';
    const sBadge = b.session ? `<span class="rsb-session">${b.session}</span>` : '';
    rows += `<div class="dash-row in-progress">
      <span class="dash-status-dot in-progress"></span>
      <span class="dash-name">${b.name}</span>
      <span class="dash-detail"><span style="color:${clr};font-size:11px">${TYPE_ICONS[type]||'◆'} ${type}</span>${sBadge}${b.description ? ` &middot; ${b.description}` : ''}</span>
      <span class="dash-progress"></span>
      <span class="dash-time">${ago(b.created_at)}</span>
    </div>`;
  }

  // Stale
  for (const r of stale) {
    rows += `<div class="dash-row stale" onclick="openRunDetail('${r.info.id}')">
      <span class="dash-status-dot stale"></span>
      <span class="dash-name">${r.info.name}</span>
      <span class="dash-detail"><span style="color:var(--text-tertiary)">stale — PID dead</span></span>
      <span class="dash-progress"></span>
      <span class="dash-time">${ago(r.mtime)}</span>
    </div>`;
  }
  // Stale cleanup button
  if (stale.length > 0) {
    rows += `<div style="padding:6px 16px;text-align:right">
      <button class="btn" style="font-size:11px;padding:3px 10px" onclick="cleanupStale()">Mark ${stale.length} stale as failed</button></div>`;
  }

  // Separator if there are active items
  if (running.length + inProgress.length + stale.length > 0 && recent.length > 0) {
    rows += '<div class="dash-sep"></div>';
  }

  // Recent completed/failed
  for (const r of recent) {
    const s = r.info.status;
    const result = r.info.result || {};
    const resultStr = Object.entries(result).filter(([,v]) => typeof v === 'number').slice(0, 3)
      .map(([k,v]) => `<span class="dash-metric">${k} <b>${fmtVal(v)}</b></span>`).join('');
    const errStr = r.info.error ? `<span style="color:var(--red)">${r.info.error.slice(0, 40)}</span>` : '';
    const gitTag2 = r.info.git_commit ? `<span class="dash-metric" style="opacity:0.6"><code>${r.info.git_commit}</code></span>` : '';
    rows += `<div class="dash-row ${s}" onclick="openRunDetail('${r.info.id}')">
      <span class="dash-status-dot ${s}"></span>
      <span class="dash-name">${r.info.name}</span>
      <span class="dash-detail">${resultStr || errStr}${gitTag2}</span>
      <span class="dash-progress"></span>
      <span class="dash-time">${ago(r.mtime)}</span>
    </div>`;
  }

  if (!rows) {
    rows = '<div class="empty">No activity yet. Run an experiment or add tasks via <code>/backlog</code>.</div>';
  }

  el.innerHTML = `<div class="dash-table">${rows}</div>`;
}

// ════════════════════════════════════
// EXPERIMENTS TAB — List + Detail
// ════════════════════════════════════

function _buildExpSidebarHtml() {
  const groups = {};
  allRuns.forEach(r => { const exp = r.info.experiment || 'unknown'; (groups[exp] = groups[exp] || []).push(r); });
  const exps = Object.keys(groups).sort();
  return exps.map(exp => {
    const rs = groups[exp].sort((a,b) => b.mtime - a.mtime);
    const isOpen = selectedExp === exp;
    const runs = isOpen ? rs.map(r => {
      const name = r.info.name.split('/').pop();
      const s = r.info.status;
      const dot = s === 'running' && r.alive ? 'running' : s === 'done' ? 'done' : s === 'failed' ? 'failed' : 'stale';
      const isActive = selectedRunId === r.info.id;
      return `<div class="exp-run${isActive ? ' active' : ''}" onclick="event.stopPropagation(); selectRun('${r.info.id}')">
        <span class="exp-run-dot ${dot}"></span>${name}</div>`;
    }).join('') : '';
    return `<div class="exp-group${isOpen ? ' open' : ''}">
      <div class="exp-item${isOpen && !selectedRunId ? ' active' : ''}" onclick="selectExp('${exp}')">
        <span class="exp-arrow">${isOpen ? '▾' : '▸'}</span>${exp}<span class="exp-item-count">${rs.length}</span>
      </div>${runs}</div>`;
  }).join('') || '<div class="empty" style="padding:20px">No experiments</div>';
}

// Full render (sidebar + detail) — called on tab switch / initial load
function renderExpList() {
  const el = document.getElementById('exp-list');
  el.innerHTML = _buildExpSidebarHtml();
  el.appendChild(_expResizeHandle);
  if (selectedRunId) selectRun(selectedRunId);
  else if (selectedExp) selectExp(selectedExp);
}

// Sidebar only — called on poll (no detail flicker)
function renderExpListOnly() {
  const el = document.getElementById('exp-list');
  el.innerHTML = _buildExpSidebarHtml();
  el.appendChild(_expResizeHandle);
}

function selectRun(runId) {
  const run = allRuns.find(r => r.info.id === runId);
  if (!run) return;
  selectedRunId = runId;
  _ss('run', runId);
  // Ensure parent exp is open
  const exp = run.info.experiment;
  if (selectedExp !== exp) { selectedExp = exp; _ss('exp', exp); }
  // Highlight in sidebar
  document.querySelectorAll('.exp-run').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.exp-item').forEach(el => el.classList.remove('active'));
  const activeEl = document.querySelector(`.exp-run[onclick*="${runId}"]`);
  if (activeEl) activeEl.classList.add('active');
  // Render run detail in exp-detail
  showRunDetailInline(run);
}

let inlineRunTab = 'info';

function showRunDetailInline(run) {
  const detail = document.getElementById('exp-detail');
  const i = run.info;

  detail.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0">
      <div>
        <span class="run-meta" style="cursor:pointer" onclick="selectExp('${i.experiment}')">${i.experiment} /</span>
        <span style="font-size:15px; font-weight:700">${i.name.split('/').pop()}</span>
      </div>
      <div style="display:flex; gap:8px; align-items:center">
        ${statusHtml(i.status, run.alive)}
        <button class="btn btn-danger" style="padding:4px 10px" onclick="deleteRunInline('${i.id}')">Delete</button>
      </div>
    </div>
    <div class="inline-tabs" id="inline-tabs">
      <div class="inline-tab${inlineRunTab==='info'?' active':''}" onclick="switchInlineTab('info')">Info</div>
      <div class="inline-tab${inlineRunTab==='logs'?' active':''}" onclick="switchInlineTab('logs')">Logs</div>
      <div class="inline-tab${inlineRunTab==='metrics'?' active':''}" onclick="switchInlineTab('metrics')">Metrics</div>
    </div>
    <div id="inline-body"></div>`;

  showRunContent('inline-body', run, inlineRunTab);
}

function switchInlineTab(tab) {
  inlineRunTab = tab;
  document.querySelectorAll('#inline-tabs .inline-tab').forEach(t => t.classList.toggle('active', t.textContent.toLowerCase() === tab));
  const run = allRuns.find(r => r.info.id === selectedRunId);
  if (run) showRunContent('inline-body', run, tab);
}

async function deleteRunInline(runId) {
  const run = allRuns.find(r => r.info.id === runId);
  if (!confirm(`Delete "${run?.info.name || runId}"?`)) return;
  const data = await (await fetch('/api/runs/' + runId, { method: 'DELETE' })).json();
  if (data.ok) { selectedRunId = null; _ss('run', ''); poll(); }
  else alert('Cannot delete active run.');
}

async function selectExp(exp) {
  selectedExp = exp;
  selectedRunId = null;
  _ss('run', '');
  _ss('exp', exp);
  document.querySelectorAll('.exp-item').forEach(el => el.classList.toggle('active', el.textContent.trim().startsWith(exp)));

  const detail = document.getElementById('exp-detail');
  const runs = allRuns.filter(r => r.info.experiment === exp).sort((a,b) => b.mtime - a.mtime);
  const done = runs.filter(r => r.info.status === 'done');

  // Fetch experiment-level metadata
  let expMeta = {};
  try { expMeta = await (await fetch('/api/experiment-meta/' + exp)).json(); } catch {};

  // ── Detect varying config keys + result keys ──
  const allConfigs = runs.map(r => r.info.config || {});
  const configKeys = _varyingKeys(allConfigs);
  const resultKeysSet = new Set();
  done.forEach(r => { if (r.info.result) Object.entries(r.info.result).forEach(([k,v]) => { if (typeof v === 'number') resultKeysSet.add(k); }); });
  const resultKeys = [...resultKeysSet].slice(0, 4);
  const mainResultKey = resultKeys.find(k => k.includes('db') || k.includes('nmse') || k.includes('acc')) || resultKeys[0];

  // ── Sweep coverage (if there are varying configs) ──
  let sweepHtml = '';
  if (configKeys.length > 0 && configKeys.length <= 3) {
    sweepHtml = _renderSweepGrid(runs, configKeys, mainResultKey);
  }

  // ── Compact runs table ──
  const cfgCols = configKeys.slice(0, 5);
  const resCols = resultKeys.slice(0, 3);
  let tableHtml = '<table class="exp-table"><tr><th></th><th>Run</th>';
  cfgCols.forEach(k => { tableHtml += `<th class="num">${k}</th>`; });
  tableHtml += '<th class="num">Epoch</th>';
  resCols.forEach(k => { tableHtml += `<th class="num">${k}</th>`; });
  tableHtml += '<th class="num">ETA</th><th class="num">Time</th></tr>';
  runs.forEach(r => {
    const cfg = r.info.config || {};
    const res = r.info.result || {};
    const m = r.latest || {};
    const isBest = mainResultKey && done.length > 1 && r === done.reduce((best, c) => {
      const bv = best.info.result?.[mainResultKey], cv = c.info.result?.[mainResultKey];
      if (bv == null) return c; if (cv == null) return best;
      return mainResultKey.includes('acc') ? (cv > bv ? c : best) : (cv < bv ? c : best);
    }, done[0]);
    const rowCls = r.info.status === 'running' && r.alive ? ' class="row-running"' : isBest ? ' class="row-best"' : '';
    const epoch = m.epoch != null ? m.epoch : (r.info.status === 'done' ? cfg.epochs || '\u2014' : '\u2014');
    const totalEpochs = cfg.epochs;
    const epochStr = totalEpochs ? `${epoch}/${totalEpochs}` : epoch;
    const eta = r.info.status === 'running' && r.alive ? (calcETA(r) || '\u2014') : '\u2014';
    tableHtml += `<tr${rowCls}>`;
    tableHtml += `<td style="width:12px;padding-right:0">${statusDot(r.info.status, r.alive)}</td>`;
    tableHtml += `<td><span class="run-link" onclick="openRunDetail('${r.info.id}')">${r.info.name.split('/').pop()}</span></td>`;
    cfgCols.forEach(k => { tableHtml += `<td class="num">${cfg[k] ?? '\u2014'}</td>`; });
    tableHtml += `<td class="num">${epochStr}</td>`;
    resCols.forEach(k => {
      const v = res[k];
      tableHtml += `<td class="num">${v != null ? `<b>${fmtVal(v)}</b>` : '\u2014'}</td>`;
    });
    tableHtml += `<td class="num run-meta">${eta}</td>`;
    const time = r.info.ended_at ? ago(r.info.ended_at) : ago(r.mtime);
    tableHtml += `<td class="num run-meta">${time}</td></tr>`;
  });
  tableHtml += '</table>';

  // Experiment design card
  let designHtml = '';
  const hasMeta = expMeta.purpose || expMeta.hypothesis || expMeta.eval_criteria || expMeta.variables;
  if (hasMeta || expMeta.git_commit) {
    const lines = [];
    if (expMeta.git_commit) lines.push(`<span class="run-meta">git <code>${expMeta.git_commit}</code></span>`);
    if (expMeta.updated_at) lines.push(`<span class="run-meta">${expMeta.updated_at}</span>`);
    const header = lines.length ? `<div style="display:flex;gap:12px;margin-bottom:8px">${lines.join('')}</div>` : '';

    const grid = [];
    if (expMeta.purpose) grid.push(`<div class="detail-key">Purpose</div><div>${expMeta.purpose}</div>`);
    if (expMeta.hypothesis) grid.push(`<div class="detail-key">Hypothesis</div><div>${expMeta.hypothesis}</div>`);
    if (expMeta.eval_criteria) grid.push(`<div class="detail-key">Eval</div><div>${expMeta.eval_criteria}</div>`);
    if (expMeta.variables) {
      Object.entries(expMeta.variables).forEach(([k,v]) => {
        grid.push(`<div class="detail-key">${k}</div><div>${Array.isArray(v) ? v.join(', ') : v}</div>`);
      });
    }
    designHtml = `<div class="card" style="margin-bottom:12px">${header}
      ${grid.length ? `<div class="detail-grid">${grid.join('')}</div>` : ''}</div>`;
  }

  let html = `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px">
    <h2 style="font-size:16px; font-weight:700">${exp} <span class="run-meta" style="font-weight:400">${runs.length} runs</span></h2>
    <button class="btn btn-danger" onclick="deleteExperiment('${exp}')">Delete All</button></div>${designHtml}`;

  if (sweepHtml) html += sweepHtml;
  html += `<div class="card" style="padding:0; overflow-x:auto">${tableHtml}</div>`;

  if (done.length >= 2) {
    html += `<div class="card"><div class="card-header"><span class="card-title">Comparison</span></div>
      <div class="chart-grid"><div class="chart-wrap" style="height:280px"><canvas id="exp-cmp"></canvas></div></div></div>`;
  }
  if (done.length >= 1) {
    html += `<div class="card"><div class="card-header"><span class="card-title">Training Curves</span>
      <button class="btn" style="font-size:10px;padding:2px 8px" onclick="resetAllZoom()">Reset Zoom</button></div>
      <div class="chart-grid" id="exp-curves"></div></div>`;
  }
  detail.innerHTML = html;

  // ── Comparison bar chart ──
  if (done.length >= 2 && mainResultKey) {
    const ctx = document.getElementById('exp-cmp');
    if (ctx) {
      if (charts['exp-cmp']) charts['exp-cmp'].destroy();
      const vals = done.map(r => r.info.result?.[mainResultKey]).filter(v => v != null);
      const vMin = Math.min(...vals), vMax = Math.max(...vals);
      const pad = Math.max(Math.abs(vMax - vMin) * 0.15, Math.abs(vMax) * 0.02 || 0.1);
      const scales = chartScales();
      scales.y.title = { display: true, text: _metricUnit(mainResultKey), color: CHART_TICK };
      scales.y.min = vMin - pad;
      scales.y.max = vMax + pad;
      charts['exp-cmp'] = new Chart(ctx, { type: 'bar',
        data: { labels: done.map(r => r.info.name.split('/').pop()),
          datasets: [{ label: mainResultKey, data: done.map(r => r.info.result?.[mainResultKey] ?? 0), backgroundColor: COLORS.slice(0, done.length), borderRadius: 4 }] },
        options: { ...baseChartOpts(), scales } });
    }
  }

  // ── Training curves (parallel fetch) ──
  if (done.length >= 1) {
    const curvesEl = document.getElementById('exp-curves');
    if (!curvesEl) return;
    const allMetrics = {};
    const fetches = done.slice(0, 8).map(async r => {
      try { const resp = await fetch('/api/metrics/' + r.info.id); return [r.info.id, await resp.json()]; }
      catch { return [r.info.id, []]; }
    });
    (await Promise.all(fetches)).forEach(([id, data]) => { allMetrics[id] = data; });

    const metricKeys = new Set();
    Object.values(allMetrics).forEach(ms => ms.forEach(m => Object.keys(m).forEach(k => {
      if (!['t','step','epoch'].includes(k) && typeof m[k] === 'number') metricKeys.add(k);
    })));
    const groups = groupMetricKeys(metricKeys);
    curvesEl.innerHTML = groups.map((_, gi) => `<div class="chart-wrap"><canvas id="expc-${gi}"></canvas></div>`).join('');
    groups.forEach((group, gi) => {
      const ctx = document.getElementById(`expc-${gi}`);
      if (!ctx) return;
      const ckey = `expc-${gi}`;
      if (charts[ckey]) charts[ckey].destroy();
      const datasets = [];
      for (const k of group) {
        done.slice(0, 8).forEach((r, ri) => {
          const ms = allMetrics[r.info.id] || [];
          const data = ms.map(m => m[k] ?? null);
          if (!data.some(v => v !== null)) return;
          datasets.push({ label: `${r.info.name.split('/').pop()} · ${k}`,
            data, borderColor: COLORS[ri % COLORS.length], backgroundColor: 'transparent',
            tension: 0.3, pointRadius: 0, borderWidth: 2,
            borderDash: group.indexOf(k) > 0 ? [4, 2] : [] });
        });
      }
      if (!datasets.length) return;
      const maxLen = Math.max(...datasets.map(d => d.data.length));
      const yTitle = _groupYTitle(group);
      charts[ckey] = new Chart(ctx, { type: 'line',
        data: { labels: Array.from({length: maxLen}, (_, i) => i), datasets },
        options: { ...baseChartOpts(),
          plugins: { ...baseChartOpts().plugins, title: { display: true, text: group.join(', '), color: CHART_TICK, font: { size: 13, weight: '600' } } },
          scales: chartScalesXY('Epoch', yTitle) } });
    });
  }
}

// ── Helper: find config keys that vary across runs ──
function _varyingKeys(configs) {
  if (configs.length <= 1) return Object.keys(configs[0] || {});
  const allKeys = new Set();
  configs.forEach(c => Object.keys(c).forEach(k => allKeys.add(k)));
  return [...allKeys].filter(k => {
    const vals = new Set(configs.map(c => JSON.stringify(c[k] ?? null)));
    return vals.size > 1;
  });
}

// ── Sweep grid: show which config combos have been tried ──
function _renderSweepGrid(runs, configKeys, resultKey) {
  if (configKeys.length === 1) {
    // 1D sweep: horizontal cells
    const key = configKeys[0];
    const vals = [...new Set(runs.map(r => r.info.config?.[key]).filter(v => v != null))].sort((a,b) => a-b || String(a).localeCompare(String(b)));
    let cells = vals.map(v => {
      const matching = runs.filter(r => r.info.config?.[key] === v);
      const best = matching.find(r => r.info.status === 'done');
      const running = matching.find(r => r.info.status === 'running');
      const res = best?.info.result?.[resultKey];
      const cls = running ? 'sweep-running' : best ? 'sweep-done' : 'sweep-empty';
      return `<div class="sweep-cell ${cls}" title="${key}=${v}${res != null ? ' → '+resultKey+'='+fmtVal(res) : ''}">
        <div class="sweep-val">${v}</div>
        ${res != null ? `<div class="sweep-res">${fmtVal(res)}</div>` : running ? '<div class="sweep-res">...</div>' : ''}
      </div>`;
    }).join('');
    return `<div class="card" style="padding:14px"><div class="run-meta" style="margin-bottom:8px">Sweep: <b>${key}</b>${resultKey ? ' → '+resultKey : ''}</div>
      <div class="sweep-row">${cells}</div></div>`;
  }
  if (configKeys.length === 2) {
    // 2D sweep: grid
    const [kRow, kCol] = configKeys;
    const rowVals = [...new Set(runs.map(r => r.info.config?.[kRow]).filter(v => v != null))].sort((a,b) => a-b || String(a).localeCompare(String(b)));
    const colVals = [...new Set(runs.map(r => r.info.config?.[kCol]).filter(v => v != null))].sort((a,b) => a-b || String(a).localeCompare(String(b)));
    let header = `<tr><th class="sweep-corner">${kRow} \\ ${kCol}</th>${colVals.map(c => `<th class="sweep-hdr">${c}</th>`).join('')}</tr>`;
    let body = rowVals.map(rv => {
      let cells = colVals.map(cv => {
        const matching = runs.filter(r => r.info.config?.[kRow] === rv && r.info.config?.[kCol] === cv);
        const best = matching.find(r => r.info.status === 'done');
        const running = matching.find(r => r.info.status === 'running');
        const res = best?.info.result?.[resultKey];
        const cls = running ? 'sweep-running' : best ? 'sweep-done' : 'sweep-empty';
        return `<td class="sweep-td ${cls}">${res != null ? fmtVal(res) : running ? '...' : ''}</td>`;
      }).join('');
      return `<tr><th class="sweep-hdr">${rv}</th>${cells}</tr>`;
    }).join('');
    return `<div class="card" style="padding:14px"><div class="run-meta" style="margin-bottom:8px">Sweep: <b>${kRow}</b> × <b>${kCol}</b>${resultKey ? ' → '+resultKey : ''}</div>
      <table class="sweep-grid">${header}${body}</table></div>`;
  }
  // 3+ keys: just show count summary
  const total = configKeys.reduce((acc, k) => acc * new Set(runs.map(r => r.info.config?.[k])).size, 1);
  return `<div class="card" style="padding:14px"><div class="run-meta">Sweep: ${configKeys.map(k => `<b>${k}</b>(${new Set(runs.map(r=>r.info.config?.[k])).size})`).join(' × ')} = ${total} combos, ${runs.length} done</div></div>`;
}

// ════════════════════
// QUEUE TAB
// ════════════════════

const TYPE_ICONS = {experiment:'⬡',research:'◈',implement:'◆',fix:'●',review:'◎',config:'◇'};
const TYPE_COLORS = {experiment:'var(--blue)',research:'var(--purple)',implement:'var(--cyan)',fix:'var(--red)',review:'var(--green)',config:'var(--text-tertiary)'};

async function renderQueue() {
  try { const resp = await fetch('/api/backlog'); backlogItems = await resp.json(); } catch (e) { backlogItems = []; }
  const active = backlogItems.filter(b => b.status === 'queued' || b.status === 'in_progress');
  const rest = backlogItems.filter(b => b.status === 'done' || b.status === 'failed');
  const badge = document.getElementById('queue-badge');
  badge.textContent = active.length; badge.style.display = active.length > 0 ? 'inline' : 'none';
  document.getElementById('queue-list').innerHTML = active.length === 0
    ? '<div class="empty">No active tasks.<br>Use <code>/backlog add</code> or click + Add.</div>'
    : active.map(b => queueItemHtml(b)).join('');
  document.getElementById('queue-done').innerHTML = rest.length === 0
    ? '<div class="empty">No completed items</div>'
    : rest.map(b => queueItemHtml(b, true)).join('');
}

function queueItemHtml(b, showStatus) {
  const type = b.type || 'experiment';
  const icon = TYPE_ICONS[type] || '◆';
  const iconColor = TYPE_COLORS[type] || 'var(--text-tertiary)';
  const configStr = b.config && Object.keys(b.config).length > 0
    ? Object.entries(b.config).slice(0, 4).map(([k,v]) => `${k}=${v}`).join(', ') : '';
  const statusCls = b.status === 'done' ? 'done' : b.status === 'in_progress' ? 'running' : 'failed';
  const statusBadge = showStatus || b.status === 'in_progress' ? `<span class="status status-${statusCls}">${b.status}</span> ` : '';
  const runLink = b.run_id ? `<span class="run-link" onclick="openRunDetail('${b.run_id}')">view run</span>` : '';
  const summary = b.summary ? `<div class="queue-desc" style="color:var(--green);font-size:12px">${b.summary}</div>` : '';
  const sessionBadge = b.session ? `<span class="rsb-session">${b.session}</span>` : '';
  const metaParts = [
    `<span style="color:${iconColor};font-size:11px">${icon} ${type}</span>`,
    sessionBadge,
    b.script ? `<span>${b.script}</span>` : '',
    configStr ? `<span>${configStr}</span>` : '',
    `<span>${ago(b.created_at)}</span>`,
    runLink
  ].filter(Boolean).join('');
  return `<div class="queue-item">
    <div class="queue-priority ${b.priority || 'normal'}"></div>
    <div class="queue-body">
      <div class="queue-name">${statusBadge}${b.name}</div>
      ${b.description ? `<div class="queue-desc">${b.description}</div>` : ''}
      ${summary}
      <div class="queue-meta">${metaParts}</div>
    </div>
    <div class="queue-actions"><button class="btn btn-danger" style="padding:4px 10px" onclick="deleteBacklogItem('${b.id}')">&times;</button></div>
  </div>`;
}

function toggleExpFields() {
  const isExp = document.getElementById('bl-type').value === 'experiment';
  document.getElementById('exp-fields').style.display = isExp ? 'block' : 'none';
}

function openAddBacklog() {
  document.getElementById('backlog-modal').classList.add('open');
  ['bl-name','bl-desc','bl-script','bl-config'].forEach(id => document.getElementById(id).value = '');
  document.getElementById('bl-type').value = 'experiment';
  document.getElementById('bl-priority').value = 'normal';
  toggleExpFields();
  setTimeout(() => document.getElementById('bl-name').focus(), 100);
}
function closeBacklogModal() { document.getElementById('backlog-modal').classList.remove('open'); }

async function submitBacklog() {
  const name = document.getElementById('bl-name').value.trim();
  if (!name) { alert('Name required'); return; }
  const type = document.getElementById('bl-type').value;
  let config = {};
  const cs = document.getElementById('bl-config').value.trim();
  if (cs) { try { config = JSON.parse(cs); } catch (e) { alert('Invalid JSON'); return; } }
  try {
    const resp = await fetch('/api/backlog', { method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ name, type, description: document.getElementById('bl-desc').value.trim(),
        script: document.getElementById('bl-script').value.trim(), config, priority: document.getElementById('bl-priority').value }) });
    if ((await resp.json()).ok) { closeBacklogModal(); renderQueue(); }
  } catch (e) { alert('Failed: ' + e.message); }
}
async function deleteBacklogItem(id) { await fetch('/api/backlog/' + id, { method: 'DELETE' }); renderQueue(); }
async function clearDoneBacklog() {
  const toClear = backlogItems.filter(b => b.status === 'done' || b.status === 'failed');
  if (!toClear.length || !confirm(`Remove ${toClear.length} item(s)?`)) return;
  for (const b of toClear) await fetch('/api/backlog/' + b.id, { method: 'DELETE' });
  renderQueue();
}
async function cleanupStale() {
  const data = await (await fetch('/api/cleanup-stale')).json();
  if (data.cleaned > 0) poll();
}

async function deleteExperiment(expName) {
  if (!confirm(`Delete ALL runs in "${expName}"?`)) return;
  const data = await (await fetch('/api/experiments/' + encodeURIComponent(expName), { method: 'DELETE' })).json();
  alert(`Deleted ${data.deleted} run(s).`); poll();
}

// ════════════════════
// MODAL
// ════════════════════

// ── Run detail (inline panel in experiments tab, modal elsewhere) ──

function openRunDetail(runId) {
  const run = allRuns.find(r => r.info.id === runId);
  if (!run) return;

  if (activeTab === 'experiments') {
    selectRun(runId);
    return;
  }
  currentModalRunId = runId;
  openRunModal(run);
}

// ── Modal (dashboard/backlog tabs) ──
function openRunModal(run) {
  document.getElementById('modal-title').textContent = run.info.name;
  document.getElementById('run-modal').classList.add('open');
  document.querySelectorAll('.modal-tab').forEach((t,i) => t.classList.toggle('active', i===0));
  showRunContent('modal-body', run, 'info');
  const btn = document.getElementById('modal-delete-btn');
  btn.disabled = run.info.status === 'running' && run.alive;
  btn.style.opacity = btn.disabled ? '0.4' : '1';
  document.getElementById('modal-status').textContent = `${run.info.id} \u00b7 PID ${run.info.pid || '?'}`;
}
function closeModal() { document.getElementById('run-modal').classList.remove('open'); currentModalRunId = null; }
document.getElementById('run-modal').addEventListener('click', e => { if (e.target === e.currentTarget) closeModal(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeModal(); }
});

function switchModalTab(tabEl, tab) {
  document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
  tabEl.classList.add('active');
  const run = allRuns.find(r => r.info.id === currentModalRunId);
  if (run) showRunContent('modal-body', run, tab);
}

// ── Shared render functions (target = element id) ──

function showRunContent(targetId, run, tab) {
  if (tab === 'info') showRunInfo(targetId, run);
  else if (tab === 'logs') showRunLogs(targetId, run.info.id);
  else if (tab === 'metrics') showRunMetrics(targetId, run.info.id);
}

function showRunInfo(targetId, run) {
  const i = run.info;
  const configHtml = i.config && Object.keys(i.config).length > 0
    ? Object.entries(i.config).map(([k,v]) => `<div class="detail-key">${k}</div><div>${typeof v === 'object' ? JSON.stringify(v) : v}</div>`).join('')
    : '<div class="detail-key">\u2014</div><div>No config</div>';
  const resultHtml = i.result && Object.keys(i.result).length > 0
    ? Object.entries(i.result).filter(([,v]) => typeof v === 'number')
        .map(([k,v]) => `<div class="detail-key">${k}</div><div><strong>${fmtVal(v)}</strong></div>`).join('') : '';
  const errorHtml = i.error ? `<div class="error-box">${i.error}</div>` : '';
  const eta = calcETA(run);
  const etaLine = eta ? `<div class="detail-key">ETA</div><div style="color:var(--cyan);font-weight:600">${eta}</div>` : '';
  document.getElementById(targetId).innerHTML = `<div class="detail-multi">
    <div class="detail-panel"><div class="detail-grid">
      <div class="detail-key">Status</div><div>${statusHtml(i.status, run.alive)}</div>
      <div class="detail-key">Started</div><div>${i.started_at || '\u2014'}</div>
      <div class="detail-key">Ended</div><div>${i.ended_at || '\u2014'}</div>
      ${etaLine}
    </div>${errorHtml}</div>
    ${resultHtml ? `<div class="detail-panel"><h3>Result</h3><div class="detail-grid">${resultHtml}</div></div>` : ''}
    <div class="detail-panel"><h3>Config</h3><div class="detail-grid">${configHtml}</div></div>
  </div>`;
}

async function showRunLogs(targetId, runId) {
  const body = document.getElementById(targetId);
  body.innerHTML = '<div class="empty">Loading...</div>';
  try {
    const text = await (await fetch('/api/logs/' + runId)).text();
    if (!text.trim()) { body.innerHTML = '<div class="empty">No logs. Use <code>capture_output=True</code>.</div>'; return; }
    const hl = text.replace(/^(.*ERROR.*)$/gm, '<span class="error">$1</span>').replace(/^(.*WARNING.*)$/gm, '<span class="warn">$1</span>');
    body.innerHTML = `<div style="display:flex; justify-content:space-between; margin-bottom:8px">
      <span class="run-meta">output.log</span><button class="btn" onclick="showRunLogs('${targetId}','${runId}')">Refresh</button></div>
      <div class="log-viewer">${hl}</div>`;
  } catch (e) { body.innerHTML = '<div class="empty" style="color:var(--red)">Failed</div>'; }
}

function _metricsXLabels(metrics, axis) {
  if (axis === 'time' && metrics[0]?.t != null) {
    const t0 = new Date(metrics[0].t).getTime();
    return { labels: metrics.map(m => (((new Date(m.t)).getTime() - t0) / 60000).toFixed(1)), xTitle: 'Time (min)' };
  }
  return { labels: metrics.map((m, i) => m.epoch ?? m.step ?? i), xTitle: 'Epoch' };
}

function resetAllZoom() {
  Object.values(charts).forEach(c => { try { c.resetZoom(); } catch {} });
}

function switchMetricsAxis(targetId, runId, axis) {
  metricsXAxis = axis;
  showRunMetrics(targetId, runId);
}

async function showRunMetrics(targetId, runId) {
  const body = document.getElementById(targetId);
  // Check if charts already exist (in-place update) or first render
  const existingCharts = Object.keys(charts).filter(k => k.startsWith(targetId.replace('-', '')));
  const isUpdate = existingCharts.length > 0 && body.querySelector('canvas');

  if (!isUpdate) body.innerHTML = '<div class="empty">Loading...</div>';

  try {
    const metrics = await (await fetch('/api/metrics/' + runId)).json();
    metricsCache[runId] = metrics;
    if (!metrics.length) { body.innerHTML = '<div class="empty">No metrics</div>'; return; }
    const last = metrics[metrics.length - 1];
    const summary = Object.entries(last).filter(([k]) => !['t','step'].includes(k))
      .map(([k,v]) => `<span><span class="key">${k}:</span> <strong>${fmtVal(v)}</strong></span>`).join('');
    const keys = new Set();
    metrics.forEach(m => Object.keys(m).forEach(k => { if (!['t','step'].includes(k) && typeof m[k] === 'number') keys.add(k); }));
    const chartGroups = groupMetricKeys(keys);
    const pfx = targetId.replace('-', '');
    const { labels, xTitle } = _metricsXLabels(metrics, metricsXAxis);
    const hasTime = metrics[0]?.t != null;

    const axisToggle = hasTime ? `<div class="metrics-axis-toggle">
      <button class="metrics-axis-btn${metricsXAxis==='epoch'?' active':''}" onclick="switchMetricsAxis('${targetId}','${runId}','epoch')">Epoch</button>
      <button class="metrics-axis-btn${metricsXAxis==='time'?' active':''}" onclick="switchMetricsAxis('${targetId}','${runId}','time')">Time</button>
    </div>` : '';

    if (!isUpdate) {
      body.innerHTML = `<div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <div class="run-metrics">${summary}</div>
        <span class="run-meta">${metrics.length} entries</span>${axisToggle}
        <button class="btn" style="font-size:10px;padding:2px 8px" onclick="resetAllZoom()">Reset Zoom</button>
      </div>
      <div class="chart-grid" id="${pfx}-charts">${chartGroups.map((_, i) => '<div class="chart-wrap" style="height:200px"><canvas id="' + pfx + '-chart-' + i + '"></canvas></div>').join('')}</div>`;
    } else {
      // Update summary + axis buttons only
      const summaryEl = body.querySelector('.run-metrics');
      if (summaryEl) summaryEl.innerHTML = summary;
      const countEl = body.querySelector('.run-meta');
      if (countEl) countEl.textContent = metrics.length + ' entries';
    }

    const run = allRuns.find(r => r.info.id === runId);
    const mu = run?.info?.metric_units || {};
    chartGroups.forEach((group, gi) => {
      const ckey = pfx + '-' + gi;
      const yTitle = _groupYTitle(group, mu);
      const datasets = group.map((k, ki) => ({
        label: k, data: metrics.map(m => m[k] ?? null),
        borderColor: COLORS[ki % COLORS.length], backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, borderWidth: 2
      }));

      if (charts[ckey] && isUpdate) {
        // In-place update (tensorboard-style)
        charts[ckey].data.labels = labels;
        charts[ckey].data.datasets.forEach((ds, di) => { if (datasets[di]) ds.data = datasets[di].data; });
        charts[ckey].options.scales.x.title.text = xTitle;
        charts[ckey].update('none');
      } else {
        const ctx = document.getElementById(pfx + '-chart-' + gi);
        if (!ctx) return;
        if (charts[ckey]) charts[ckey].destroy();
        charts[ckey] = new Chart(ctx, { type: 'line',
          data: { labels, datasets },
          options: { ...baseChartOpts(), scales: chartScalesXY(xTitle, yTitle) } });
      }
    });
  } catch (e) { if (!isUpdate) body.innerHTML = '<div class="empty" style="color:var(--red)">Failed</div>'; }
}

async function deleteCurrentRun() {
  if (!currentModalRunId) return;
  const run = allRuns.find(r => r.info.id === currentModalRunId);
  if (!confirm(`Delete "${run?.info.name || currentModalRunId}"?`)) return;
  const data = await (await fetch('/api/runs/' + currentModalRunId, { method: 'DELETE' })).json();
  if (data.ok) { closeModal(); poll(); } else alert('Cannot delete active run.');
}

// ════════════════════
// POLLING
// ════════════════════

async function poll() {
  try {
    const [runsResp, blResp] = await Promise.all([fetch('/api/runs'), fetch('/api/backlog')]);
    allRuns = await runsResp.json();
    backlogItems = await blResp.json();
    const nActive = backlogItems.filter(b => b.status === 'queued' || b.status === 'in_progress').length;
    const badge = document.getElementById('queue-badge');
    badge.textContent = nActive; badge.style.display = nActive > 0 ? 'inline' : 'none';
    if (activeTab === 'dashboard') renderDashboard();
    else if (activeTab === 'experiments') renderExpListOnly();
    else if (activeTab === 'queue') renderQueue();
    renderRightSidebar();
    // Live-update selected run metrics (tensorboard-style)
    if (selectedRunId && inlineRunTab === 'metrics') {
      const run = allRuns.find(r => r.info.id === selectedRunId);
      if (run && run.info.status === 'running') showRunMetrics('inline-body', selectedRunId);
    }
  } catch (e) { console.error('Poll failed:', e); }
}

async function pollSystem() {
  try { renderSidebar(await (await fetch('/api/system')).json()); } catch (e) {}
}

function renderSidebar(s) {
  const cpu = s.cpu || {};
  const load = cpu.load_1m ?? 0, cores = cpu.cores ?? 1;
  const cpuPct = Math.min(100, (load / cores) * 100);
  const cpuColor = cpuPct > 80 ? 'var(--red)' : cpuPct > 50 ? 'var(--yellow)' : 'var(--green)';

  const mem = s.memory || {};
  const memPct = mem.pct ?? 0;
  const memColor = memPct > 85 ? 'var(--red)' : memPct > 60 ? 'var(--yellow)' : 'var(--green)';

  document.getElementById('sys-info').innerHTML = `
    <div class="sys-cell">
      <div class="sys-head"><span>CPU</span><span>${load.toFixed(1)}/${cores}c</span></div>
      <div class="sys-bar"><div class="sys-fill" style="width:${cpuPct}%;background:${cpuColor}"></div></div>
    </div>
    <div class="sys-cell">
      <div class="sys-head"><span>MEM</span><span>${mem.used_gb??0}/${mem.total_gb??0}G</span></div>
      <div class="sys-bar"><div class="sys-fill" style="width:${memPct}%;background:${memColor}"></div></div>
    </div>`;

  const gpus = s.gpus || [];
  if (!gpus.length) { document.getElementById('gpu-info').innerHTML = '<span class="run-meta">No GPU</span>'; return; }
  document.getElementById('gpu-info').innerHTML = gpus.map(g => {
    const pct = g.mem_pct;
    const clr = pct > 90 ? 'var(--red)' : pct > 60 ? 'var(--yellow)' : 'var(--blue)';
    const dot = g.ours ? '<span class="gpu-dot"></span>' : '';
    return `<div class="gpu-row${g.ours ? ' ours' : ''}">
      <span class="gpu-id">${dot}${g.index}</span>
      <div class="gpu-bar"><div class="gpu-fill" style="width:${pct}%;background:${clr}"></div></div>
      <span class="gpu-num">${g.mem_used_mb}/${g.mem_total_mb}M</span>
      <span class="gpu-num">${g.util_pct}%</span>
    </div>`;
  }).join('');

  // Mini sidebar (collapsed view)
  let mini = `
    <div class="mini-item"><span>C</span><div class="mini-bar"><div class="mini-fill" style="width:${cpuPct}%;background:${cpuColor}"></div></div></div>
    <div class="mini-item"><span>M</span><div class="mini-bar"><div class="mini-fill" style="width:${memPct}%;background:${memColor}"></div></div></div>`;
  gpus.forEach(g => {
    const pct = g.mem_pct;
    const clr = pct > 90 ? 'var(--red)' : pct > 60 ? 'var(--yellow)' : 'var(--blue)';
    mini += `<div class="mini-item"><span>${g.index}</span><div class="mini-bar"><div class="mini-fill" style="width:${pct}%;background:${clr}"></div></div></div>`;
  });
  document.getElementById('sidebar-mini').innerHTML = mini;
}

// ── Exp sidebar resize ──
(function() {
  const sidebar = document.getElementById('exp-list');
  if (!sidebar) return;
  sidebar.appendChild(_expResizeHandle);

  // Restore width
  const saved = _st('exp-w');
  if (saved) sidebar.style.width = saved + 'px';

  _expResizeHandle.addEventListener('mousedown', e => { _expDragging = true; _expResizeHandle.classList.add('active'); e.preventDefault(); });
  document.addEventListener('mousemove', e => {
    if (!_expDragging) return;
    const rect = sidebar.parentElement.getBoundingClientRect();
    const w = Math.max(140, Math.min(400, e.clientX - rect.left));
    sidebar.style.width = w + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (!_expDragging) return;
    _expDragging = false; _expResizeHandle.classList.remove('active');
    _ss('exp-w', parseInt(sidebar.style.width));
  });
})();

// ── Init ──
poll().then(() => renderRightSidebar());
pollSystem();
setInterval(poll, 5000);
setInterval(pollSystem, 10000);
