if (typeof supabase === 'undefined') {
  document.getElementById('login-screen').innerHTML = `
    <div class="login-card">
      <div class="logo">Face<span>Attend</span></div>
      <div class="error-msg" style="display:block;margin-top:16px;">
        Failed to load. Check your internet connection and refresh.
      </div>
    </div>`;
  throw new Error('Supabase SDK failed to load');
}

const SUPABASE_URL      = "https://xrlsltunfgjxooyyrora.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhybHNsdHVuZmdqeG9veXlyb3JhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwNDczODEsImV4cCI6MjA4MDYyMzM4MX0.BWr27wHWGt6a3gWnD2ocGdQBL0_sH0HK-YHUcJsrlC0";
const API_URL           = "https://faceattend.app";
const DAZZLING_URL      = "https://faceattend.app";

const { createClient } = supabase;
const client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

let currentToken         = null;
let aiSummaryData        = null;
let currentInstitutionId = null;
let isSuperAdmin         = false;
let allRecords           = [];
let allSecurityData      = [];
let pendingDeleteId      = null;
let pendingDeleteName    = null;
let pendingRemoveCoordId = null;
let pendingRemoveCoordName = null;
let pendingRevokeKeyId   = null;
let pendingRevokeKeyName = null;
let newKeyRawValue       = null;
let courseUnitsCache     = [];
let coordinatorsData     = [];
let auditLoaded          = false;

document.getElementById('ai-scope')?.addEventListener('change', function() {
  const wrap = document.getElementById('ai-scope-id-wrap');
  if (wrap) wrap.style.display = this.value === 'course_unit' ? '' : 'none';
});

// ── Toast ──────────────────────────────────────────────────────────────────
function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `show toast-${type}`;
  setTimeout(() => { t.className = ''; }, 3000);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, '&#96;');
}

function escapeJsString(value) {
  return String(value ?? '')
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\r/g, '\\r')
    .replace(/\n/g, '\\n');
}

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

  const btn = document.getElementById(`tab-btn-${tab}`);
  if (btn) btn.classList.add('active');
  const panel = document.getElementById(`tab-${tab}`);
  if (panel) panel.classList.add('active');

  if (tab === 'students')   loadStudents();
  if (tab === 'superadmin') loadInstitutions();
  if (tab === 'team')       loadCoordinators();
  if (tab === 'apikeys')    loadApiKeys();
  if (tab === 'billing')    loadBilling();
  if (tab === 'security')   loadSecurityData();
  if (tab === 'units')      loadCourseUnits();
  if (tab === 'aisummary') { populateAIScopeUnits(); }
  if (tab === 'audit') {
    if (!auditLoaded) {
      auditLoaded = true;
      auditLoadActionFilter();
      if (isSuperAdmin) {
        document.getElementById('audit-institution-filter-wrap').style.display = '';
        fetch(`${DAZZLING_URL}/admin/institutions`, {
          headers: { 'Authorization': `Bearer ${currentToken}` }
        }).then(r => r.json()).then(data => {
          const insts = data.institutions || [];
          const sel = document.getElementById('audit-filter-institution');
          sel.innerHTML = '<option value="">All Institutions</option>' +
            insts.map(i => `<option value="${escapeAttr(i.id)}">${escapeHtml(i.id)} — ${escapeHtml(i.name)}</option>`).join('');
        }).catch(() => {});
      }
    }
    auditLoadPage(1);
  }
}

// ── Auto-login from Flutter WebView token injection ────────────────────────
async function autoLoginWithToken(accessToken, refreshToken) {
  if (!accessToken) return;
  try {
    const { data, error } = await client.auth.setSession({
      access_token:  accessToken,
      refresh_token: refreshToken,
    });
    if (data?.session) await initDashboard(data.session, true);
  } catch (e) {
    console.error('Auto-login failed:', e);
  }
}

// ── Page load: check for Supabase recovery hash OR existing session ────────
window.addEventListener('load', async () => {
  // Detect Supabase password-recovery redirect (hash contains type=recovery)
  const hash = window.location.hash;
  if (hash && hash.includes('type=recovery')) {
    let recoveryScreenShown = false;
    const showRecoveryScreenOnce = () => {
      if (recoveryScreenShown) return;
      recoveryScreenShown = true;
      showSetNewPasswordScreen();
    };

    // Register listener first so PASSWORD_RECOVERY cannot be missed.
    const { data: authListener } = client.auth.onAuthStateChange((event, session) => {
      if (event === 'PASSWORD_RECOVERY' || (event === 'SIGNED_IN' && session)) {
        showRecoveryScreenOnce();
      }
    });

    // First try current session (hash may already be processed by Supabase).
    const { data: sessionData } = await client.auth.getSession();
    if (sessionData?.session) {
      showRecoveryScreenOnce();
    } else {
      // Fallback: explicitly parse tokens from hash and set recovery session.
      const hashParams = new URLSearchParams(hash.substring(1));
      const accessToken = hashParams.get('access_token');
      const refreshToken = hashParams.get('refresh_token') || '';
      if (accessToken) {
        const { data: setData } = await client.auth.setSession({
          access_token: accessToken,
          refresh_token: refreshToken,
        });
        if (setData?.session) showRecoveryScreenOnce();
      }
    }

    if (recoveryScreenShown) authListener?.subscription?.unsubscribe?.();
    return;
  }

  const { data: { session } } = await client.auth.getSession();
  if (session) await initDashboard(session);
});

// ── Set New Password Screen (recovery flow) ────────────────────────────────
function showSetNewPasswordScreen() {
  document.getElementById('login-screen').style.display       = 'none';
  document.getElementById('dashboard-screen').style.display   = 'none';
  document.getElementById('set-new-pw-screen').style.display  = 'flex';
  setTimeout(() => document.getElementById('set-pw-new').focus(), 100);
}

function updatePwStrength(pw) {
  const fill = document.getElementById('pw-strength-fill');
  let score = 0;
  if (pw.length >= 8)          score++;
  if (pw.length >= 12)         score++;
  if (/[A-Z]/.test(pw))        score++;
  if (/[0-9]/.test(pw))        score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  const colors = ['#EF4444','#F59E0B','#F59E0B','#10B981','#10B981'];
  fill.style.width      = `${(score / 5) * 100}%`;
  fill.style.background = colors[score - 1] || 'rgba(255,255,255,0.06)';
}

async function confirmSetNewPassword() {
  const pw1  = document.getElementById('set-pw-new').value;
  const pw2  = document.getElementById('set-pw-confirm').value;
  const btn  = document.getElementById('set-pw-btn');
  const msg  = document.getElementById('set-pw-msg');
  msg.style.display = 'none';

  if (!pw1 || pw1.length < 8) {
    msg.textContent = 'Password must be at least 8 characters.';
    msg.className = 'set-pw-msg error'; msg.style.display = 'block'; return;
  }
  if (pw1 !== pw2) {
    msg.textContent = 'Passwords do not match.';
    msg.className = 'set-pw-msg error'; msg.style.display = 'block'; return;
  }

  btn.disabled = true; btn.textContent = 'Updating…';

  try {
    const { error } = await client.auth.updateUser({ password: pw1 });
    if (error) throw error;

    msg.textContent = '✓ Password updated! Redirecting to sign in…';
    msg.className = 'set-pw-msg success'; msg.style.display = 'block';
    document.getElementById('set-pw-fields').style.display = 'none';

    // Clear the hash and redirect to login after a short delay
    setTimeout(() => {
      window.location.hash = '';
      document.getElementById('set-new-pw-screen').style.display = 'none';
      document.getElementById('login-screen').style.display      = 'flex';
    }, 2000);
  } catch (err) {
    msg.textContent = err.message || 'Failed to update password.';
    msg.className = 'set-pw-msg error'; msg.style.display = 'block';
    btn.disabled = false; btn.textContent = 'Update Password';
  }
}

// ── Forgot Password Modal ──────────────────────────────────────────────────
function openForgotModal() {
  document.getElementById('forgot-email').value    = '';
  document.getElementById('forgot-email').disabled = false;
  const msg = document.getElementById('forgot-msg');
  msg.style.display = 'none'; msg.className = 'forgot-pw-msg';
  const btn = document.getElementById('forgot-submit-btn');
  btn.disabled = false; btn.textContent = 'Send Link'; btn.style.display = '';
  document.getElementById('forgot-fields').style.display = '';
  document.getElementById('forgot-modal').classList.add('visible');
  setTimeout(() => document.getElementById('forgot-email').focus(), 100);
}

function closeForgotModal() {
  document.getElementById('forgot-modal').classList.remove('visible');
}

async function sendResetLink() {
  const email = document.getElementById('forgot-email').value.trim();
  const msg   = document.getElementById('forgot-msg');
  const btn   = document.getElementById('forgot-submit-btn');
  msg.style.display = 'none';

  if (!email) {
    msg.textContent = 'Please enter your email address.';
    msg.className = 'forgot-pw-msg error'; msg.style.display = 'block'; return;
  }

  btn.disabled = true; btn.textContent = 'Sending…';

  try {
    const resp = await fetch(`${DAZZLING_URL}/auth/forgot-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });

    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data?.detail || data?.message || 'Failed to send reset link.');

    msg.textContent = data?.message || '✓ Reset link sent — check your inbox.';
    msg.className = 'forgot-pw-msg success'; msg.style.display = 'block';
    btn.style.display = 'none';
    document.getElementById('forgot-email').disabled = true;
  } catch (err) {
    msg.textContent = err.message || 'Failed to send reset link.';
    msg.className = 'forgot-pw-msg error'; msg.style.display = 'block';
    btn.disabled = false; btn.textContent = 'Send Link';
  }
}

// ── Manual login ──────────────────────────────────────────────────────────
async function login() {
  const emailEl    = document.getElementById('login-email');
  const passwordEl = document.getElementById('login-password');
  const email      = emailEl.value.trim();
  const password   = passwordEl.value;
  const btn        = document.getElementById('login-btn');
  const errEl      = document.getElementById('login-error');

  if (!email || !password) { showLoginError("Please enter email and password."); return; }

  btn.disabled    = true;
  btn.textContent = 'Signing in...';
  errEl.style.display = 'none';

  const { data, error } = await client.auth.signInWithPassword({ email, password });

  if (error) {
    showLoginError(error.message);
    btn.disabled    = false;
    btn.textContent = 'Sign In';
    return;
  }

  const { data: profile } = await client
    .from('profiles')
    .select('is_admin, is_super_admin')
    .eq('id', data.user.id)
    .single();

  const isAdmin           = profile && (profile.is_admin === true || profile.is_admin === 'true');
  const isSuperAdminCheck = profile && (profile.is_super_admin === true || profile.is_super_admin === 'true');

  if (!profile || !(isAdmin || isSuperAdminCheck)) {
    showLoginError("Access denied. Admin account required.");
    await client.auth.signOut();
    btn.disabled    = false;
    btn.textContent = 'Sign In';
    return;
  }

  emailEl.value    = '';
  passwordEl.value = '';
  btn.disabled     = false;
  btn.textContent  = 'Sign In';

  await initDashboard(data.session, true);
}

function showLoginError(msg) {
  const el = document.getElementById('login-error');
  el.textContent   = msg;
  el.style.display = 'block';
}

// ── Init dashboard ─────────────────────────────────────────────────────────
async function initDashboard(session, isFreshLogin = false) {
  currentToken = session.access_token;

// 1. Fire non-blocking analytics asynchronously
  if (isFreshLogin) {
    fetch(`${DAZZLING_URL}/auth/log-login`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${session.access_token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ source: 'dashboard'}),
    }).catch(e => console.warn('log-login failed:', e));
  }

  try {
    // 2. Resolve critical profile architecture early
    const { data: profile, error } = await client
      .from('profiles')
      .select('institution_id, is_super_admin, role')
      .eq('id', session.user.id)
      .single();

    if (error) throw error;

    currentInstitutionId = profile?.institution_id || null;
    isSuperAdmin         = profile?.is_super_admin === true;

    // 3. Clear existing dynamic items before building UI
    const tabs = document.getElementById('tabs');
    tabs.innerHTML = ''; 

    // Helper to generate tabs compliant with your strict CSP
    const createSecureTab = (id, text, targetTab, classNames) => {
      const btn = document.createElement('button');
      btn.className = `tab-btn ${classNames}`;
      btn.id = id;
      btn.textContent = text;
      btn.addEventListener('click', () => switchTab(targetTab));
      return btn;
    };

    // 4. Construct navigation structure dynamically
    tabs.appendChild(createSecureTab('tab-btn-attendance', '📅 Attendance', 'attendance', 'attendance-tab'));
    tabs.appendChild(createSecureTab('tab-btn-students', '👥 Students', 'students', ''));
    tabs.appendChild(createSecureTab('tab-btn-aisummary', '🤖 AI Summary', 'aisummary', ''));

    if (!isSuperAdmin && currentInstitutionId) {
      tabs.appendChild(createSecureTab('tab-btn-units', '📚 Course Units', 'units', 'units-tab'));
    }

    tabs.appendChild(createSecureTab('tab-btn-team', 'Team', 'team', ''));

        

    if (isSuperAdmin) {
      const teamInstFilter = document.getElementById('filter-team-inst');
      teamInstFilter.style.display = '';
      fetch(`${DAZZLING_URL}/admin/institutions`, {
        headers: { 'Authorization': `Bearer ${currentToken}` }
      })
      .then(r => r.json())
      .then(data => {
        const insts = data.institutions || [];
        teamInstFilter.innerHTML = '<option value="">All Institutions</option>' +
          insts.map(i => `<option value="${i.id}">${i.id} — ${i.name}</option>`).join('');
      }).catch(() => {});
    }

    if (!isSuperAdmin) {
      tabs.appendChild(createSecureTab('tab-btn-billing', '💳 Billing', 'billing', 'billing-tab'));
    }

    tabs.appendChild(createSecureTab('tab-btn-apikeys', '⚙ API Keys', 'apikeys', 'dev-tab'));

    if (isSuperAdmin) {
      const adminTabBtn = document.createElement('button');
      adminTabBtn.className = 'tab-btn superadmin-tab';
      adminTabBtn.id = 'tab-btn-superadmin';
      adminTabBtn.innerHTML = '🏢 Institutions <span class="pending-count" id="pending-count" style="display:none"></span>';
      adminTabBtn.addEventListener('click', () => switchTab('superadmin'));
      tabs.appendChild(adminTabBtn);

      tabs.appendChild(createSecureTab('tab-btn-security', '🛡 Security', 'security', 'security-tab'));
    }

    tabs.appendChild(createSecureTab('tab-btn-audit', '📋 Audit Logs', 'audit', 'audit-tab'));

    // 5. Pre-fetch essential local dataset caches
    if (!isSuperAdmin && currentInstitutionId) {
      const { data: units } = await client
        .from('course_units')
        .select('id, name, code')
        .eq('institution_id', currentInstitutionId)
        .order('created_at', { ascending: true });
      courseUnitsCache = units || [];
      populateAIScopeUnits();
    }

    // 6. Execute global async routines
    await loadData();
    if (isSuperAdmin) await fetchPendingCount();

    // 7. Transition interface context only after complete initialization success
    document.getElementById('nav-user').textContent = session.user.email;
    document.getElementById('nav-superadmin').style.display = isSuperAdmin ? '' : 'none';
    
    const navInst = document.getElementById('nav-inst');
    if (currentInstitutionId) {
      navInst.textContent = currentInstitutionId;
      navInst.style.display = '';
    } else {
      navInst.style.display = 'none';
    }

    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('dashboard-screen').style.display = 'block';

  } catch (err) {
    console.error('Critical initialization failure:', err);
    alert('Failed to securely initialize your dashboard session. Please login again.');
  }
}

// ── AI SUMMARY TAB ─────────────────────────────────────────────────────────
function populateAIScopeUnits() {
  const sel = document.getElementById('ai-scope-id');
  const scopeSel = document.getElementById('ai-scope');
  if (scopeSel && !currentInstitutionId && isSuperAdmin) {
    const institutionOption = scopeSel.querySelector('option[value="institution"]');
    if (institutionOption) institutionOption.textContent = 'All Institutions';
  }
  if (!sel) return;
  sel.innerHTML = '<option value="">— Select unit —</option>' +
    courseUnitsCache.map(u =>
      `<option value="${escapeAttr(u.id)}">${escapeHtml(u.name)}${u.code ? ' (' + escapeHtml(u.code) + ')' : ''}</option>`
    ).join('');
}

async function generateAISummary() {
  const btn      = document.getElementById('ai-generate-btn');
  const preview  = document.getElementById('ai-summary-preview');
  const scope    = document.getElementById('ai-scope').value;
  const scopeId  = document.getElementById('ai-scope-id').value;
  const dateFrom = document.getElementById('ai-date-from').value;
  const dateTo   = document.getElementById('ai-date-to').value;

  if (scope === 'course_unit' && !scopeId) {
    showToast('Please select a course unit', 'error'); return;
  }

  if (btn) { btn.disabled = true; btn.textContent = '⏳ Generating...'; }
  if (preview) preview.classList.remove('visible');

  try {
    const params = new URLSearchParams({ scope });
    if (scopeId)  params.set('scope_id', scopeId);
    if (dateFrom) params.set('date_from', dateFrom);
    if (dateTo)   params.set('date_to', dateTo);
    if (currentInstitutionId) params.set('institution_id', currentInstitutionId);

    const resp = await fetch(`${API_URL}/admin/ai-attendance-summary?${params}`, {
      headers: { 'Authorization': `Bearer ${currentToken}` }
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Failed to generate summary');

    aiSummaryData = data;
    renderAISummary(data);
    if (preview) preview.classList.add('visible');
    preview?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    showToast('✓ Summary generated', 'success');
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '✦ Generate'; }
  }
}

function renderAISummary(data) {
  const stats = data.stats || {};
  const metaParts = [data.institution];
  if (data.scope === 'course_unit' && data.scope_id) metaParts.push(`Course Unit: ${data.scope_id}`);
  if (data.date_from || data.date_to) metaParts.push(`${data.date_from || '...'} → ${data.date_to || 'today'}`);
  metaParts.push(new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }));
  document.getElementById('ai-preview-meta').textContent = metaParts.join(' · ');

  const riskColor = stats.at_risk_count > 0 ? 'var(--red)' : 'var(--cyan)';
  document.getElementById('ai-stats-strip').innerHTML = `
    <div class="ai-stat">
      <div class="ai-stat-label">Students</div>
      <div class="ai-stat-value">${stats.total_students ?? '—'}</div>
    </div>
    <div class="ai-stat">
      <div class="ai-stat-label">Total Records</div>
      <div class="ai-stat-value">${stats.total_records ?? '—'}</div>
    </div>
    <div class="ai-stat">
      <div class="ai-stat-label">Avg Attendance</div>
      <div class="ai-stat-value" style="color:var(--cyan)">${stats.overall_attendance_pct ?? '—'}%</div>
    </div>
    <div class="ai-stat">
      <div class="ai-stat-label">At Risk</div>
      <div class="ai-stat-value" style="color:${riskColor}">${stats.at_risk_count ?? '—'}</div>
    </div>
  `;

  document.getElementById('ai-summary-text').textContent = data.summary || '';
  const atRisk = data.at_risk || [];
  if (atRisk.length) {
    document.getElementById('ai-at-risk-table').innerHTML = `
      <div class="ai-at-risk-title">⚠ At-Risk Students (below 75%)</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Student</th><th>Attendance</th><th>Rate</th></tr></thead>
          <tbody>${atRisk.map(s => `
            <tr>
              <td>${escapeHtml(s.name)}</td>
              <td style="color:var(--red);font-weight:600;">${s.attendance_pct}%</td>
              <td><div style="height:6px;border-radius:3px;background:rgba(255,255,255,0.06);overflow:hidden;"><div style="height:6px;border-radius:3px;background:var(--red);width:${s.attendance_pct}%;"></div></div></td>
            </tr>`).join('')}</tbody>
        </table>
      </div>`;
  } else {
    document.getElementById('ai-at-risk-table').innerHTML =
      `<div class="ai-at-risk-title" style="color:var(--cyan)">✓ No at-risk students</div>`;
  }
}

function downloadSummaryPDF() {
  if (!aiSummaryData) return;
  const data  = aiSummaryData;
  const stats = data.stats || {};
  const now   = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
  const atRiskRows = (data.at_risk || []).map(s =>
    `<tr><td>${escapeHtml(s.name)}</td><td style="color:#ef4444;font-weight:600;">${s.attendance_pct}%</td></tr>`
  ).join('');

  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"/>
  <title>AI Attendance Report — ${data.institution}</title>
  <style>
    body { font-family: Arial, sans-serif; color: #111; max-width: 750px; margin: 40px auto; padding: 0 24px; }
    h1 { font-size: 1.4rem; margin-bottom: 4px; }
    .meta { color: #666; font-size: 0.82rem; margin-bottom: 24px; }
    .stats { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
    .stat { border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; flex: 1; min-width: 100px; }
    .stat-label { font-size: 0.68rem; color: #999; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
    .stat-value { font-size: 1.5rem; font-weight: 800; }
    h2 { font-size: 1rem; border-bottom: 2px solid #eee; padding-bottom: 6px; margin: 24px 0 12px; }
    .summary { font-size: 0.88rem; line-height: 1.8; white-space: pre-wrap; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th { text-align: left; font-size: 0.72rem; color: #999; text-transform: uppercase; padding: 8px; border-bottom: 1px solid #eee; }
    td { padding: 8px; border-bottom: 1px solid #f0f0f0; font-size: 0.85rem; }
    .footer { margin-top: 40px; font-size: 0.72rem; color: #aaa; text-align: center; }
  </style></head><body>
  <h1>AI Attendance Report — ${escapeHtml(data.institution)}</h1>
  <div class="meta">Generated ${now} · Powered by FaceAttend AI</div>
  <div class="stats">
    <div class="stat"><div class="stat-label">Students</div><div class="stat-value">${stats.total_students ?? '—'}</div></div>
    <div class="stat"><div class="stat-label">Records</div><div class="stat-value">${stats.total_records ?? '—'}</div></div>
    <div class="stat"><div class="stat-label">Avg Attendance</div><div class="stat-value">${stats.overall_attendance_pct ?? '—'}%</div></div>
    <div class="stat"><div class="stat-label">At Risk</div><div class="stat-value" style="color:#ef4444">${stats.at_risk_count ?? '—'}</div></div>
  </div>
  <h2>AI Analysis</h2>
  <div class="summary">${escapeHtml(data.summary || '')}</div>
  ${atRiskRows ? `<h2>At-Risk Students</h2><table><thead><tr><th>Student</th><th>Attendance</th></tr></thead><tbody>${atRiskRows}</tbody></table>` : ''}
  <div class="footer">FaceAttend · faceattend.app · Confidential</div>
  </body></html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `attendance-report-${data.institution}-${new Date().toISOString().slice(0,10)}.html`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Report downloaded — open in browser and print to PDF', 'success');
}

// ── Pending count badge ────────────────────────────────────────────────────
async function fetchPendingCount() {
  try {
    const resp = await fetch(`${DAZZLING_URL}/admin/institutions?status=pending`, {
      headers: { 'Authorization': `Bearer ${currentToken}` }
    });
    if (!resp.ok) return;
    const data = await resp.json();
    const count = data.count || 0;
    const badge = document.getElementById('pending-count');
    if (badge && count > 0) {
      badge.textContent   = count;
      badge.style.display = 'inline-block';
    }
  } catch(e) { console.error(e); }
}

async function logout() {
  if (currentToken) {
    try {
      await fetch(`${DAZZLING_URL}/auth/log-logout`, {
        method: 'POST', headers: { 'Authorization': `Bearer ${currentToken}` }
      });
    } catch (e) { console.warn('log-logout failed:', e); }
  }

  await client.auth.signOut();
  localStorage.removeItem('lastActivity');
  currentToken         = null;
  currentInstitutionId = null;
  isSuperAdmin         = false;
  allRecords           = [];
  allSecurityData      = [];
  courseUnitsCache     = [];
  coordinatorsData     = [];
  auditLoaded          = false;

  ['tab-btn-attendance','tab-btn-students','tab-btn-team','tab-btn-units',
   'tab-btn-billing','tab-btn-apikeys','tab-btn-superadmin',
   'tab-btn-security','tab-btn-audit'].forEach(id => {
    document.getElementById(id)?.remove();
  });

  document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === 0));
  document.querySelectorAll('.tab-panel').forEach((p, i) => p.classList.toggle('active', i === 0));

  document.getElementById('nav-superadmin').style.display = 'none';
  document.getElementById('nav-inst').style.display       = 'none';
  document.getElementById('filter-inst').style.display    = '';

  document.getElementById('login-email').value    = '';
  document.getElementById('login-password').value = '';
  document.getElementById('login-error').style.display = 'none';

  document.getElementById('new-key-banner').classList.remove('visible');
  newKeyRawValue = null;

  document.getElementById('login-screen').style.display    = 'flex';
  document.getElementById('dashboard-screen').style.display = 'none';
}

// ── Load attendance data ───────────────────────────────────────────────────
async function loadData() {
  if (!currentToken) return;
  const headers = { 'Authorization': `Bearer ${currentToken}` };

  document.getElementById('table-wrap').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading records...</div>';

  try {
    const instParam = currentInstitutionId ? `&institution_id=${currentInstitutionId}` : '';
    const instQuery = currentInstitutionId ? `?institution_id=${currentInstitutionId}` : '';

    const [recResp, studResp] = await Promise.all([
      fetch(`${API_URL}/admin/attendance-records?limit=1000${instParam}`, { headers }),
      fetch(`${DAZZLING_URL}/students${instQuery}`, { headers }),
    ]);

    const records  = recResp.ok  ? await recResp.json()  : [];
    const studData = studResp.ok ? await studResp.json() : { count: 0 };

    allRecords = Array.isArray(records) ? records : [];
    const totalStudents = studData.count || 0;

    const total    = allRecords.length;
    const verified = allRecords.filter(r => r.verified === 'success').length;
    const failed   = allRecords.filter(r => r.verified === 'failed').length;
    const spoofs   = allRecords.filter(r => r.verified === 'spoof').length;
    const rate     = total > 0 ? Math.round(verified / total * 100) : 0;

    document.getElementById('metrics').innerHTML = `
      <div class="metric-card cyan">
        <div class="metric-label">Verified</div>
        <div class="metric-value">${verified}</div>
        <div class="metric-sub">successful scans</div>
      </div>
      <div class="metric-card blue">
        <div class="metric-label">Total Students</div>
        <div class="metric-value">${totalStudents}</div>
        <div class="metric-sub">enrolled</div>
      </div>
      <div class="metric-card orange">
        <div class="metric-label">Failed + Spoofs</div>
        <div class="metric-value">${failed + spoofs}</div>
        <div class="metric-sub">${spoofs} spoof attempts</div>
      </div>
      <div class="metric-card purple">
        <div class="metric-label">Success Rate</div>
        <div class="metric-value">${rate}%</div>
        <div class="metric-sub">recognition accuracy</div>
      </div>
    `;

    if (!currentInstitutionId) {
      const insts = [...new Set(allRecords.map(r => r.institution_id).filter(Boolean))].sort();
      document.getElementById('filter-inst').innerHTML =
        '<option value="">All Institutions</option>' +
        insts.map(i => `<option value="${i}">${i}</option>`).join('');
    }

    renderTable(allRecords);

  } catch (e) {
    document.getElementById('table-wrap').innerHTML =
      `<div class="loading" style="color:var(--red)">Failed to load: ${e.message}</div>`;
  }
}

// ── SECURITY TAB ───────────────────────────────────────────────────────────
async function loadSecurityData() {
  if (!currentToken || !isSuperAdmin) return;

  document.getElementById('security-table-wrap').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading security data...</div>';
  document.getElementById('security-summary-grid').innerHTML = '';

  try {
    const periodDays = document.getElementById('security-filter-period').value;
    const resp = await fetch(`${API_URL}/admin/attendance-records?limit=5000`, {
      headers: { 'Authorization': `Bearer ${currentToken}` }
    });
    if (!resp.ok) throw new Error('Failed to fetch records');
    let records = await resp.json();
    records = Array.isArray(records) ? records : [];

    if (periodDays !== 'all') {
      const cutoff = Date.now() - parseInt(periodDays) * 24 * 60 * 60 * 1000;
      records = records.filter(r => r.timestamp && new Date(r.timestamp).getTime() >= cutoff);
    }

    allSecurityData = records;

    const byInst = {};
    for (const r of records) {
      const id = r.institution_id || 'Unknown';
      if (!byInst[id]) byInst[id] = { spoof: 0, failed: 0, success: 0, total: 0 };
      byInst[id].total++;
      if (r.verified === 'spoof')   byInst[id].spoof++;
      if (r.verified === 'failed')  byInst[id].failed++;
      if (r.verified === 'success') byInst[id].success++;
    }

    const rows = Object.entries(byInst).sort((a, b) => b[1].spoof - a[1].spoof);

    const totalSpoofs  = rows.reduce((s, [, v]) => s + v.spoof, 0);
    const totalFailed  = rows.reduce((s, [, v]) => s + v.failed, 0);
    const totalSuccess = rows.reduce((s, [, v]) => s + v.success, 0);
    const flagged      = rows.filter(([, v]) => v.total > 0 && (v.spoof / v.total) >= 0.1).length;

    document.getElementById('security-summary-grid').innerHTML = `
      <div class="security-card red">
        <div class="metric-label">Total Spoofs</div>
        <div class="metric-value" style="color:var(--red)">${totalSpoofs}</div>
        <div class="metric-sub">spoof attempts</div>
      </div>
      <div class="security-card orange">
        <div class="metric-label">Total Failed</div>
        <div class="metric-value" style="color:var(--orange)">${totalFailed}</div>
        <div class="metric-sub">failed verifications</div>
      </div>
      <div class="security-card cyan">
        <div class="metric-label">Successful</div>
        <div class="metric-value" style="color:var(--cyan)">${totalSuccess}</div>
        <div class="metric-sub">verified scans</div>
      </div>
      <div class="security-card yellow">
        <div class="metric-label">Flagged Institutions</div>
        <div class="metric-value" style="color:var(--yellow)">${flagged}</div>
        <div class="metric-sub">spoof rate ≥ 10%</div>
      </div>
    `;

    if (!rows.length) {
      document.getElementById('security-table-wrap').innerHTML =
        '<div class="loading">No security records found for this period.</div>';
      return;
    }

    const maxSpoof = Math.max(...rows.map(([, v]) => v.spoof), 1);

    const tableRows = rows.map(([instId, v]) => {
      const spoofRate = v.total > 0 ? ((v.spoof / v.total) * 100).toFixed(1) : '0.0';
      const isFlagged = v.total > 0 && (v.spoof / v.total) >= 0.1;
      const barWidth  = Math.round((v.spoof / maxSpoof) * 100);
      const barColor  = isFlagged ? 'var(--red)' : v.spoof > 0 ? 'var(--orange)' : 'var(--muted)';
      const flagMark  = isFlagged ? '<span class="flag-badge"></span>' : '';
      const rowClass  = isFlagged ? 'flag-row' : '';

      return `
        <tr class="${rowClass}" id="row-${instId}">
          <td><strong>${flagMark}${instId}</strong></td>
          <td style="color:var(--red);font-weight:600;">${v.spoof}</td>
          <td style="color:var(--orange)">${v.failed}</td>
          <td style="color:var(--cyan)">${v.success}</td>
          <td>
            <div class="spoof-rate-row">
              <div class="spoof-rate-bg">
                <div class="spoof-rate-fill" style="width:${barWidth}%;background:${barColor};"></div>
              </div>
              <span style="font-size:0.78rem;color:${isFlagged ? 'var(--red)' : 'var(--muted)'};min-width:38px;">${spoofRate}%</span>
            </div>
          </td>
          <td>
            <button class="drill-btn" id="drill-btn-${instId}" onclick="toggleDrill('${instId}')">View Logs</button>
          </td>
        </tr>
        <tr id="drill-${instId}" style="display:none;">
          <td colspan="6" style="padding:0;">
            <div class="drill-panel" id="drill-content-${instId}"></div>
          </td>
        </tr>
      `;
    }).join('');

    document.getElementById('security-table-wrap').innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Institution</th><th>Spoof Attempts</th><th>Failed Verifications</th>
            <th>Successful Scans</th><th>Spoof Rate</th><th>Drill Down</th>
          </tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>
      <div class="security-note">
        <span class="flag-badge"></span>
        Institutions with spoof rate ≥ 10% are flagged in red. Click "View Logs" to see individual attempts.
      </div>
    `;

  } catch (e) {
    document.getElementById('security-table-wrap').innerHTML =
      `<div class="loading" style="color:var(--red)">Failed to load: ${e.message}</div>`;
  }
}

function toggleDrill(instId) {
  const drillRow     = document.getElementById(`drill-${instId}`);
  const drillContent = document.getElementById(`drill-content-${instId}`);
  const drillBtn     = document.getElementById(`drill-btn-${instId}`);

  const isOpen = drillRow.style.display !== 'none';
  if (isOpen) {
    drillRow.style.display = 'none';
    drillBtn.classList.remove('active');
    drillBtn.textContent = 'View Logs';
    return;
  }

  drillRow.style.display = 'table-row';
  drillBtn.classList.add('active');
  drillBtn.textContent = 'Hide Logs';

  const periodDays = document.getElementById('security-filter-period').value;
  let cutoff = null;
  if (periodDays !== 'all') cutoff = Date.now() - parseInt(periodDays) * 24 * 60 * 60 * 1000;

  const records = allSecurityData.filter(r => {
    const matchInst   = (r.institution_id || 'Unknown') === instId;
    const matchStatus = r.verified === 'spoof' || r.verified === 'failed';
    const matchTime   = !cutoff || (r.timestamp && new Date(r.timestamp).getTime() >= cutoff);
    return matchInst && matchStatus && matchTime;
  }).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)).slice(0, 50);

  if (!records.length) {
    drillContent.innerHTML = '<div style="padding:16px;color:var(--muted);font-size:0.82rem;">No records found.</div>';
    return;
  }

  const rows = records.map(r => {
    const ts    = r.timestamp ? new Date(r.timestamp).toLocaleString('en-GB', { timeZone: 'Africa/Nairobi' }) : '—';
    const badge = r.verified === 'spoof'
      ? '<span class="badge badge-spoof">Spoof</span>'
      : '<span class="badge badge-failed">Failed</span>';
    const conf = r.confidence ? (r.confidence * 100).toFixed(1) + '%' : '—';
    return `<tr><td>${badge}</td><td>${r.student_id || '—'}</td><td>${conf}</td><td style="color:var(--muted)">${ts}</td></tr>`;
  }).join('');

  drillContent.innerHTML = `
    <table>
      <thead><tr><th>Type</th><th>Student ID</th><th>Confidence</th><th>Time (EAT)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="table-footer">Showing ${records.length} record${records.length !== 1 ? 's' : ''}${records.length === 50 ? ' (capped at 50)' : ''}</div>
  `;
}

function exportSecurityCSV() {
  const records = allSecurityData.filter(r => r.verified === 'spoof' || r.verified === 'failed');
  if (!records.length) { showToast('No security records to export', 'error'); return; }
  const header = ['institution_id', 'verified', 'student_id', 'confidence', 'timestamp'];
  const rows   = records.map(r => header.map(k => r[k] ?? '').join(','));
  const csv    = [header.join(','), ...rows].join('\n');
  const blob   = new Blob([csv], { type: 'text/csv' });
  const url    = URL.createObjectURL(blob);
  const a      = document.createElement('a');
  a.href = url; a.download = `security_${new Date().toISOString().slice(0,10)}.csv`; a.click();
  URL.revokeObjectURL(url);
}

// ── AUDIT LOGS TAB ─────────────────────────────────────────────────────────
const AUDIT_ACTION_LABELS = {
  'auth.login':                { label: 'Login',                 color: '#4a9eff' },
  'auth.logout':               { label: 'Logout',                color: '#5a5a7a' },
  'auth.password_reset':       { label: 'Password Reset',        color: '#f5c400' },
  'auth.password_change':      { label: 'Password Changed',      color: '#f5c400' },
  'coordinator.invite':        { label: 'Coordinator Invited',   color: '#a78bfa' },
  'coordinator.remove':        { label: 'Coordinator Removed',   color: '#ff5050' },
  'student.create':            { label: 'Student Registered',    color: '#00f5c4' },
  'student.delete':            { label: 'Student Deleted',       color: '#ff5050' },
  'api_key.create':            { label: 'API Key Created',       color: '#00f5c4' },
  'api_key.revoke':            { label: 'API Key Revoked',       color: '#ff5050' },
  'institution.approve':       { label: 'Institution Approved',  color: '#00f5c4' },
  'institution.suspend':       { label: 'Institution Suspended', color: '#ff5050' },
  'attendance.verify':         { label: 'Attendance Verified',   color: '#4a9eff' },
  'attendance.spoof_detected': { label: '⚠ Spoof Detected',      color: '#ff5050' },
};

let auditCurrentPage = 1;

function auditBadge(action) {
  const def = AUDIT_ACTION_LABELS[action] || { label: action, color: '#5a5a7a' };
  return `<span class="audit-action-badge" style="background:${def.color}1a;color:${def.color};border:1px solid ${def.color}33;">${def.label}</span>`;
}

function auditFormatTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  d.setHours(d.getHours() + 3);
  const pad = n => String(n).padStart(2, '0');
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())} <span style="color:var(--muted)">${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}</span>`;
}

function auditFormatMeta(meta) {
  if (!meta || !Object.keys(meta).length) return '—';
  const skip = new Set(['ip_address']);
  const parts = Object.entries(meta)
    .filter(([k, v]) => !skip.has(k) && v)
    .map(([k, v]) => `<span style="color:var(--muted)">${escapeHtml(k.replace(/_/g,' '))}:</span> ${escapeHtml(v)}`)
    .join(' &nbsp;·&nbsp; ');
  return parts || '—';
}

async function auditLoadPage(page = 1) {
  auditCurrentPage = page;
  const wrap = document.getElementById('audit-table-wrap');
  wrap.innerHTML = '<div class="loading"><div class="spinner"></div>Loading audit logs...</div>';

  const action      = document.getElementById('audit-filter-action')?.value || '';
  const start       = document.getElementById('audit-filter-start')?.value || '';
  const end         = document.getElementById('audit-filter-end')?.value || '';
  const institution = document.getElementById('audit-filter-institution')?.value || '';

  const params = new URLSearchParams({ page, limit: 50 });
  if (action)      params.set('action', action);
  if (start)       params.set('start_date', start);
  if (end)         params.set('end_date', end);
  if (institution) params.set('institution_id', institution);

  let data;
  try {
    const res = await fetch(`${DAZZLING_URL}/audit-logs?${params}`, {
      headers: { 'Authorization': `Bearer ${currentToken}` }
    });
    if (!res.ok) throw new Error(await res.text());
    data = await res.json();
  } catch (e) {
    wrap.innerHTML = `<div class="loading" style="color:var(--red)">Error loading logs: ${escapeHtml(e.message)}</div>`;
    return;
  }

  document.getElementById('audit-stats').innerHTML =
    `<span><strong>${data.total || 0}</strong> total events</span>
     <span>Page <strong>${data.page || 1}</strong> of <strong>${data.pages || 1}</strong></span>`;

  if (!data.data?.length) {
    wrap.innerHTML = '<div class="loading">No audit events found.</div>';
  } else {
    const rows = data.data.map(ev => `
      <tr>
        <td style="white-space:nowrap;font-size:0.78rem;">${auditFormatTime(ev.created_at)}</td>
        <td>${auditBadge(ev.action)}</td>
        <td style="color:var(--muted);font-size:0.8rem;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeAttr(ev.actor_email || '')}">
          ${ev.actor_email ? escapeHtml(ev.actor_email) : '<span style="color:var(--muted)">—</span>'}
        </td>
        <td style="font-family:'DM Mono',monospace;font-size:0.75rem;color:var(--muted);">${ev.ip_address || '—'}</td>
        <td style="font-size:0.78rem;color:var(--muted);max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeAttr(JSON.stringify(ev.metadata||{}))}">
          ${auditFormatMeta(ev.metadata)}
        </td>
      </tr>
    `).join('');

    wrap.innerHTML = `
      <table>
        <thead><tr><th>Time (EAT)</th><th>Action</th><th>Actor</th><th>IP Address</th><th>Details</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="table-footer">${data.data.length} events on this page</div>
    `;
  }

  const totalPages = data.pages || 1;
  document.getElementById('audit-pagination').innerHTML = `
    <button onclick="auditLoadPage(${page - 1})" ${page <= 1 ? 'disabled' : ''}>← Prev</button>
    <span>Page ${page} of ${totalPages}</span>
    <button onclick="auditLoadPage(${page + 1})" ${page >= totalPages ? 'disabled' : ''}>Next →</button>
  `;
}

async function auditLoadActionFilter() {
  try {
    const res  = await fetch(`${DAZZLING_URL}/audit-logs/actions`, {
      headers: { 'Authorization': `Bearer ${currentToken}` }
    });
    if (!res.ok) return;
    const data = await res.json();
    const sel  = document.getElementById('audit-filter-action');
    (data.actions || []).forEach(a => {
      const opt = document.createElement('option');
      opt.value = a; opt.textContent = AUDIT_ACTION_LABELS[a]?.label || a;
      sel.appendChild(opt);
    });
  } catch (e) { /* silent */ }
}

function auditExportCSV() {
  const action      = document.getElementById('audit-filter-action')?.value || '';
  const start       = document.getElementById('audit-filter-start')?.value || '';
  const end         = document.getElementById('audit-filter-end')?.value || '';
  const institution = document.getElementById('audit-filter-institution')?.value || '';
  const params = new URLSearchParams({ page: 1, limit: 200 });
  if (action)      params.set('action', action);
  if (start)       params.set('start_date', start);
  if (end)         params.set('end_date', end);
  if (institution) params.set('institution_id', institution);

  fetch(`${DAZZLING_URL}/audit-logs?${params}`, {
    headers: { 'Authorization': `Bearer ${currentToken}` }
  })
    .then(r => r.json())
    .then(data => {
      const rows = [['Time (UTC)', 'Action', 'Actor', 'IP Address', 'Resource', 'Details']];
      (data.data || []).forEach(ev => {
        rows.push([ev.created_at||'', ev.action||'', ev.actor_email||'', ev.ip_address||'',
          ev.resource_type ? `${ev.resource_type} ${ev.resource_id||''}`.trim() : '',
          JSON.stringify(ev.metadata || {})]);
      });
      const csv  = rows.map(r => r.map(c => `"${String(c).replace(/"/g,'""')}"`).join(',')).join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href = url; a.download = `faceattend-audit-${new Date().toISOString().slice(0,10)}.csv`; a.click();
      URL.revokeObjectURL(url);
    })
    .catch(e => showToast('Export failed: ' + e.message, 'error'));
}

// ── Load students ──────────────────────────────────────────────────────────
async function loadStudents() {
  if (!currentToken) return;
  const headers   = { 'Authorization': `Bearer ${currentToken}` };
  const instQuery = currentInstitutionId ? `?institution_id=${currentInstitutionId}` : '';

  document.getElementById('students-table-wrap').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading students...</div>';

  try {
    const resp = await fetch(`${DAZZLING_URL}/students${instQuery}`, { headers });
    const data = resp.ok ? await resp.json() : { students: [] };
    const students = data.students || [];

    if (!students.length) {
      document.getElementById('students-table-wrap').innerHTML =
        '<div class="loading">No students registered.</div>';
      return;
    }

    const rows = students.map(s => `
      <tr>
        <td>${escapeHtml(s.id || '—')}</td>
        <td>${escapeHtml(s.name || '—')}</td>
        <td>${escapeHtml(s.institution_id || '—')}</td>
        <td>
          <button class="btn-delete" onclick="openDeleteModal('${escapeJsString(s.id)}', '${escapeJsString(s.name || '')}')">Delete</button>
        </td>
      </tr>
    `).join('');

    document.getElementById('students-table-wrap').innerHTML = `
      <table>
        <thead><tr><th>Student ID</th><th>Name</th><th>Institution</th><th>Action</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="table-footer">${students.length} students</div>
    `;
  } catch (e) {
    document.getElementById('students-table-wrap').innerHTML =
      `<div class="loading" style="color:var(--red)">Failed to load: ${escapeHtml(e.message)}</div>`;
  }
}

// ── Load coordinators ──────────────────────────────────────────────────────
async function loadCoordinators() {
  if (!currentToken) return;

  document.getElementById('team-table-wrap').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading team...</div>';

  try {
    const teamInstFilter = document.getElementById('filter-team-inst');
    const teamInstVal    = teamInstFilter ? teamInstFilter.value : '';
    const teamInstQuery  = teamInstVal ? `?institution_id=${teamInstVal}` : '';
    const resp = await fetch(`${DAZZLING_URL}/coordinators${teamInstQuery}`, {
      headers: { 'Authorization': `Bearer ${currentToken}` }
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      document.getElementById('team-table-wrap').innerHTML =
        `<div class="loading" style="color:var(--red)">${escapeHtml(err.detail || 'Failed to load team.')}</div>`;
      return;
    }

    const data = await resp.json();
    const coordinators = data.coordinators || [];
    coordinatorsData = coordinators;

    if (!coordinators.length) {
      document.getElementById('team-table-wrap').innerHTML = `
        <div class="loading" style="padding:48px;">
          <div style="margin-bottom:12px;font-size:1.5rem;">👥</div>
          No coordinators yet. Invite someone to get started.
        </div>
      `;
      return;
    }

    if (!isSuperAdmin && !courseUnitsCache.length && currentInstitutionId) {
      const { data: units, error } = await client
        .from('course_units').select('id, name, code')
        .eq('institution_id', currentInstitutionId).order('created_at', { ascending: true });
      if (!error) courseUnitsCache = units || [];
    }

    const rows = coordinators.map(c => {
      const joined   = c.created_at
        ? new Date(c.created_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
        : '—';
      const safeName = escapeJsString(c.full_name || '');
      const instCell = isSuperAdmin
        ? `<td style="color:var(--muted);font-size:0.8rem;">${escapeHtml(c.institution_id || '—')}</td>`
        : '';

      let assignedUnitIds = [];
      if (Array.isArray(c.course_unit_id)) assignedUnitIds = c.course_unit_id;
      else if (c.course_unit_id) assignedUnitIds = [c.course_unit_id];

      const assignedUnits = assignedUnitIds
        .map(id => courseUnitsCache.find(u => u.id == id)).filter(u => u)
        .map(u => `${escapeHtml(u.name)}${u.code ? ' (' + escapeHtml(u.code) + ')' : ''}`);

      const unitCell = !isSuperAdmin ? `<td>
            <select class="unit-assign-select" multiple id="unit-select-${c.id}">
              ${courseUnitsCache.map(u =>
                `<option value="${escapeAttr(u.id)}">${escapeHtml(u.name)}${u.code ? ' (' + escapeHtml(u.code) + ')' : ''}</option>`
              ).join('')}
            </select>
            <button onclick="assignCoordinatorUnit('${escapeJsString(c.id)}')"
              style="margin-top:6px;display:block;background:transparent;border:1px solid rgba(74,158,255,0.4);color:var(--blue);padding:4px 12px;border-radius:6px;font-size:0.75rem;cursor:pointer;transition:background 0.2s;"
              onmouseover="this.style.background='rgba(74,158,255,0.1)'"
              onmouseout="this.style.background='transparent'">Save</button>
           </td>` : '';

      return `
        <tr>
          <td>
            ${c.full_name ? escapeHtml(c.full_name) : '<span style="color:var(--muted)">Pending setup</span>'}
            <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">
              ${assignedUnits.length > 0
                ? assignedUnits.map(u => `<span style="background:rgba(74,158,255,0.1);border:1px solid rgba(74,158,255,0.2);color:var(--blue);border-radius:100px;padding:2px 10px;font-size:0.68rem;font-weight:600;letter-spacing:0.05em;">${u}</span>`).join('')
                : '<span style="color:var(--muted);font-size:0.78rem;">No units assigned</span>'}
            </div>
          </td>
          <td><span class="badge badge-coord">Coordinator</span></td>
          ${instCell}${unitCell}
          <td style="color:var(--muted);font-size:0.8rem;">${joined}</td>
          <td><button class="btn-delete" onclick="openRemoveCoordModal('${escapeJsString(c.id)}', '${safeName}')">Remove</button></td>
        </tr>
      `;
    }).join('');

    const unitHeader = !isSuperAdmin ? '<th>Assign Units</th>' : '';
    document.getElementById('team-table-wrap').innerHTML = `
      <table>
        <thead>
          <tr><th>Name</th><th>Role</th>${isSuperAdmin ? '<th>Institution</th>' : ''}${unitHeader}<th>Added</th><th>Action</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="table-footer">${coordinators.length} coordinator${coordinators.length !== 1 ? 's' : ''}</div>
    `;
  } catch (e) {
    document.getElementById('team-table-wrap').innerHTML =
      `<div class="loading" style="color:var(--red)">Failed to load: ${escapeHtml(e.message)}</div>`;
  }
}

// ── COURSE UNITS ───────────────────────────────────────────────────────────
async function loadCourseUnits() {
  if (!currentToken || !currentInstitutionId) return;
  document.getElementById('units-table-wrap').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading course units...</div>';

  try {
    const { data: units, error } = await client
      .from('course_units').select('id, name, code, created_at')
      .eq('institution_id', currentInstitutionId).order('created_at', { ascending: true });
    if (error) throw error;
    courseUnitsCache = units || [];

    if (!courseUnitsCache.length) {
      document.getElementById('units-table-wrap').innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📚</div>
          <p>No course units yet.<br>Add your first unit above.</p>
        </div>`;
      return;
    }

    const rows = courseUnitsCache.map(u => {
      const created = new Date(u.created_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
      return `<tr>
        <td><strong>${escapeHtml(u.name)}</strong></td>
        <td style="color:var(--muted)">${escapeHtml(u.code || '—')}</td>
        <td style="color:var(--muted);font-size:0.78rem;">${created}</td>
        <td><button class="btn-delete" onclick="deleteCourseUnit('${escapeJsString(u.id)}', '${escapeJsString(u.name)}')">Delete</button></td>
      </tr>`;
    }).join('');

    document.getElementById('units-table-wrap').innerHTML = `
      <table>
        <thead><tr><th>Unit Name</th><th>Code</th><th>Created</th><th>Action</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="table-footer">${courseUnitsCache.length} unit${courseUnitsCache.length !== 1 ? 's' : ''}</div>
    `;
  } catch (e) {
    document.getElementById('units-table-wrap').innerHTML =
      `<div class="loading" style="color:var(--red)">Failed to load: ${e.message}</div>`;
  }
}

async function createCourseUnit() {
  const nameEl = document.getElementById('unit-name-input');
  const codeEl = document.getElementById('unit-code-input');
  const errEl  = document.getElementById('units-create-error');
  const btn    = document.getElementById('add-unit-btn');
  const name   = nameEl.value.trim();
  const code   = codeEl.value.trim();
  errEl.style.display = 'none';
  if (!name) { errEl.textContent = 'Unit name is required.'; errEl.style.display = 'block'; return; }
  btn.disabled = true; btn.textContent = 'Adding...';
  try {
    const { error } = await client.from('course_units')
      .insert({ institution_id: currentInstitutionId, name, code: code || null });
    if (error) throw error;
    nameEl.value = ''; codeEl.value = '';
    showToast(`✓ "${name}" added`, 'success');
    await loadCourseUnits(); populateInviteUnitDropdown(); populateAIScopeUnits();
  } catch (e) { errEl.textContent = e.message || 'Failed to create unit.'; errEl.style.display = 'block'; }
  finally { btn.disabled = false; btn.textContent = '+ Add Unit'; }
}

async function deleteCourseUnit(unitId, unitName) {
  if (!confirm(`Delete "${unitName}"? Coordinators assigned to it will become unassigned.`)) return;
  try {
    const { error } = await client.from('course_units').delete().eq('id', unitId);
    if (error) throw error;
    showToast(`"${unitName}" deleted`, 'success');
    await loadCourseUnits(); populateInviteUnitDropdown(); populateAIScopeUnits();
  } catch (e) { showToast(`Failed: ${e.message}`, 'error'); }
}

async function assignCoordinatorUnit(coordId) {
  const selectElement = document.getElementById(`unit-select-${coordId}`);
  if (!selectElement) return;
  const newUnitIds = Array.from(selectElement.selectedOptions).map(o => o.value).filter(id => id);
  if (!newUnitIds.length) { showToast('Select at least one unit first', 'error'); return; }
  const coord = coordinatorsData.find(c => c.id === coordId);
  let existing = Array.isArray(coord?.course_unit_id) ? coord.course_unit_id : coord?.course_unit_id ? [coord.course_unit_id] : [];
  const merged = [...new Set([...existing, ...newUnitIds])];
  try {
    const body = new URLSearchParams();
    body.append('course_unit_ids', merged.join(','));
    const resp = await fetch(`${DAZZLING_URL}/admin/coordinators/${coordId}/course-unit`, {
      method: 'PATCH',
      headers: { 'Authorization': `Bearer ${currentToken}`, 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
    });
    const result = await resp.json();
    if (!resp.ok) throw new Error(result.detail || 'Failed to update.');
    const unitNames = Array.from(selectElement.selectedOptions).map(o => o.text).join(', ');
    showToast(`✓ Assigned: ${unitNames}`, 'success');
    await loadCoordinators();
  } catch (e) { showToast(`Failed: ${e.message}`, 'error'); await loadCoordinators(); }
}

function populateInviteUnitDropdown() {
  const sel = document.getElementById('invite-unit');
  if (!sel) return;
  sel.innerHTML = '<option value="">— Select a unit —</option>' +
    courseUnitsCache.map(u => `<option value="${u.id}">${u.name}${u.code ? ' (' + u.code + ')' : ''}</option>`).join('');
}

// ── Load billing ───────────────────────────────────────────────────────────
async function loadBilling() {
  if (!currentToken || !currentInstitutionId) return;
  document.getElementById('billing-wrap').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading billing info...</div>';
  try {
    const { data: inst, error } = await client.from('institutions')
      .select('plans, status, trial_ends_at, name').eq('id', currentInstitutionId).limit(1).single();
    if (error || !inst) {
      document.getElementById('billing-wrap').innerHTML =
        '<div class="loading" style="color:var(--red)">Failed to load billing info.</div>';
      return;
    }
    const plan = (inst.plans || 'trial').toLowerCase();
    const trialEnds = inst.trial_ends_at ? new Date(inst.trial_ends_at) : null;
    const daysLeft = trialEnds ? Math.max(0, Math.ceil((trialEnds - Date.now()) / (1000 * 60 * 60 * 24))) : null;
    const planLabel = plan === 'enterprise' ? 'Enterprise' : plan === 'standard' ? 'Standard' : plan === 'starter' ? 'Starter' : 'Trial';
    const planColor = plan === 'enterprise' ? 'var(--purple)' : plan === 'standard' ? 'var(--cyan)' : 'var(--yellow)';
    const expiryCard = trialEnds ? `
      <div class="billing-card orange">
        <div class="metric-label">Trial Expires</div>
        <div class="metric-value" style="font-size:1.1rem;">${trialEnds.toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric' })}</div>
        <div class="metric-sub" style="margin-top:8px;">${daysLeft > 0 ? `${daysLeft} day${daysLeft !== 1 ? 's' : ''} remaining` : 'Expired — upgrade to restore access'}</div>
      </div>` : '';
    const upgradeBlock = (plan === 'trial' || plan === 'starter') ? `
      <div class="upgrade-section">
        <div class="upgrade-plan-tag">Standard Plan — $30 / month</div>
        <div class="upgrade-title">Upgrade to Standard</div>
        <div class="upgrade-desc">Get full access to all FaceAttend features — unlimited attendance records, coordinator invites, and priority support.</div>
        <div class="upgrade-features">
          <span class="upgrade-feature">Unlimited attendance records</span>
          <span class="upgrade-feature">Up to 5 coordinators</span>
          <span class="upgrade-feature">CSV export</span>
          <span class="upgrade-feature">Priority support</span>
        </div>
        <div class="pesapal-row">
          <button onclick="handleUpgrade('standard')" style="background:#00bcd4;color:#000;border:none;padding:12px 28px;border-radius:8px;font-weight:700;font-size:1rem;cursor:pointer;width:100%;">Pay Now — $30/month</button>
          <span class="pesapal-note">Secure payment via Pesapal · MoMo, cards &amp; bank</span>
        </div>
      </div>` : plan === 'standard' ? `
      <div class="enterprise-cta">
        <p>You're on the Standard plan. Need custom integrations, a dedicated account manager, or direct API access? Upgrade to Enterprise.</p>
        <button class="btn-contact-enterprise" onclick="window.open('mailto:abubaker@faceattend.app?subject=Enterprise Plan Inquiry','_blank')">Contact Us for Enterprise</button>
      </div>` : `
      <div class="enterprise-cta" style="border-color:rgba(167,139,250,0.3);">
        <p style="color:var(--purple);">You're on the Enterprise plan. For billing inquiries or changes, contact us directly.</p>
        <button class="btn-contact-enterprise" onclick="window.open('mailto:abubaker@faceattend.app?subject=Enterprise Billing','_blank')">Contact Support</button>
      </div>`;
    document.getElementById('billing-wrap').innerHTML = `
      <div class="billing-grid">
        <div class="billing-card cyan">
          <div class="metric-label">Current Plan</div>
          <div class="metric-value" style="color:${planColor};">${planLabel}</div>
          <div class="metric-sub" style="margin-top:8px;">${inst.name || currentInstitutionId}</div>
        </div>
        ${expiryCard}
      </div>
      ${upgradeBlock}`;
  } catch (e) {
    document.getElementById('billing-wrap').innerHTML =
      `<div class="loading" style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

// ── API KEYS ───────────────────────────────────────────────────────────────
async function loadApiKeys() {
  if (!currentToken) return;
  document.getElementById('apikeys-table-wrap').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading keys...</div>';
  try {
    const orgId = currentInstitutionId;
    const query = orgId ? `?org_id=${orgId}` : '';
    const resp = await fetch(`${DAZZLING_URL}/v1/keys${query}`, {
      headers: { 'Authorization': `Bearer ${currentToken}` }
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      document.getElementById('apikeys-table-wrap').innerHTML =
        `<div class="loading" style="color:var(--red)">${err.detail || 'Failed to load API keys.'}</div>`;
      return;
    }
    const data = await resp.json();
    const keys = data.keys || [];
    if (!keys.length) {
      document.getElementById('apikeys-table-wrap').innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">🔑</div>
          <p>No API keys yet.<br>Generate your first key to start using the FaceAttend Enterprise API.</p>
        </div>`;
      return;
    }
    const rows = keys.map(k => {
      const created  = k.created_at ? new Date(k.created_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : '—';
      const lastUsed = k.last_used_at ? new Date(k.last_used_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : 'Never';
      const planBadge = k.plan === 'enterprise' ? '<span class="badge badge-enterprise">Enterprise</span>' :
        k.plan === 'pro' ? '<span class="badge badge-pro">Pro</span>' : '<span class="badge badge-starter">Starter</span>';
      const statusBadge = k.is_active ? '<span class="badge badge-active">Active</span>' : '<span class="badge badge-revoked">Revoked</span>';
      const revokeBtn = k.is_active
        ? `<button class="btn-revoke" onclick="openRevokeModal('${k.id}', '${(k.name || '').replace(/'/g, "\\'")}')">Revoke</button>`
        : `<span style="color:var(--muted);font-size:0.75rem;">—</span>`;
      const instCell = isSuperAdmin ? `<td style="color:var(--muted);font-size:0.78rem;">${k.org_id || '—'}</td>` : '';
      const suffix = k.key_suffix || '????';
      return `<tr>
        <td><strong>${k.name || 'Unnamed Key'}</strong></td>
        <td><span class="key-masked">fa_live_••••••••••••${suffix}</span></td>
        <td>${planBadge}</td><td>${statusBadge}</td>${instCell}
        <td class="last-used-cell">${lastUsed}</td>
        <td style="color:var(--muted);font-size:0.78rem;">${created}</td>
        <td>${revokeBtn}</td>
      </tr>`;
    }).join('');
    document.getElementById('apikeys-table-wrap').innerHTML = `
      <table>
        <thead><tr><th>Name</th><th>Key</th><th>Plan</th><th>Status</th>${isSuperAdmin ? '<th>Org</th>' : ''}<th>Last Used</th><th>Created</th><th>Action</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="table-footer">${keys.length} key${keys.length !== 1 ? 's' : ''}</div>`;
  } catch (e) {
    document.getElementById('apikeys-table-wrap').innerHTML =
      `<div class="loading" style="color:var(--red)">Failed to load: ${e.message}</div>`;
  }
}

function openGenKeyModal() {
  document.getElementById('genkey-name').value = '';
  document.getElementById('genkey-error').style.display = 'none';
  document.getElementById('confirm-gen-btn').disabled = false;
  document.getElementById('confirm-gen-btn').textContent = 'Generate';
  document.getElementById('genkey-modal').classList.add('visible');
  setTimeout(() => document.getElementById('genkey-name').focus(), 100);
}
function closeGenKeyModal() { document.getElementById('genkey-modal').classList.remove('visible'); }

async function confirmGenerateKey() {
  const nameEl = document.getElementById('genkey-name');
  const errEl  = document.getElementById('genkey-error');
  const btn    = document.getElementById('confirm-gen-btn');
  const name   = nameEl.value.trim();
  errEl.style.display = 'none';
  if (!name) { errEl.textContent = 'Please enter a name.'; errEl.style.display = 'block'; return; }
  btn.disabled = true; btn.textContent = 'Generating...';
  try {
    const resp = await fetch(`${DAZZLING_URL}/v1/keys`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${currentToken}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const result = await resp.json();
    if (resp.ok && result.api_key) {
      newKeyRawValue = result.api_key;
      closeGenKeyModal();
      const banner = document.getElementById('new-key-banner');
      document.getElementById('new-key-value').textContent = result.api_key;
      document.getElementById('copy-key-btn').textContent = 'Copy';
      document.getElementById('copy-key-btn').classList.remove('copied');
      banner.classList.add('visible');
      banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      showToast('✓ API key generated', 'success');
      await loadApiKeys();
    } else {
      errEl.textContent = result.detail || 'Failed to generate key.'; errEl.style.display = 'block';
      btn.disabled = false; btn.textContent = 'Generate';
    }
  } catch (e) {
    errEl.textContent = `Error: ${e.message}`; errEl.style.display = 'block';
    btn.disabled = false; btn.textContent = 'Generate';
  }
}

function copyNewKey() {
  if (!newKeyRawValue) return;
  navigator.clipboard.writeText(newKeyRawValue).then(() => {
    const btn = document.getElementById('copy-key-btn');
    btn.textContent = '✓ Copied'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2500);
  }).catch(() => {
    const el = document.createElement('textarea');
    el.value = newKeyRawValue; el.style.position = 'fixed'; el.style.opacity = '0';
    document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
    const btn = document.getElementById('copy-key-btn');
    btn.textContent = '✓ Copied'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2500);
  });
}

function openRevokeModal(keyId, keyName) {
  pendingRevokeKeyId = keyId; pendingRevokeKeyName = keyName;
  document.getElementById('revoke-modal-msg').textContent =
    `Are you sure you want to revoke "${keyName || 'this key'}"? This cannot be undone.`;
  document.getElementById('revoke-modal').classList.add('visible');
}
function closeRevokeModal() {
  pendingRevokeKeyId = null; pendingRevokeKeyName = null;
  document.getElementById('revoke-modal').classList.remove('visible');
}
async function confirmRevoke() {
  if (!pendingRevokeKeyId || !currentToken) return;
  const keyId = pendingRevokeKeyId; closeRevokeModal();
  try {
    const resp = await fetch(`${DAZZLING_URL}/v1/keys/${keyId}`, {
      method: 'DELETE', headers: { 'Authorization': `Bearer ${currentToken}` },
    });
    if (resp.ok) { showToast('API key revoked', 'success'); await loadApiKeys(); }
    else { const err = await resp.json().catch(() => ({})); showToast(`Failed: ${err.detail || 'Unknown error'}`, 'error'); }
  } catch (e) { showToast(`Error: ${e.message}`, 'error'); }
}

function openInviteModal() {
  document.getElementById('invite-name').value  = '';
  document.getElementById('invite-email').value = '';
  document.getElementById('invite-error').style.display = 'none';
  document.getElementById('send-invite-btn').disabled   = false;
  document.getElementById('send-invite-btn').textContent = 'Send Invite';
  populateInviteUnitDropdown();
  document.getElementById('invite-unit').value = '';
  document.getElementById('invite-modal').classList.add('visible');
  setTimeout(() => document.getElementById('invite-name').focus(), 100);
}
function closeInviteModal() { document.getElementById('invite-modal').classList.remove('visible'); }

async function sendInvite() {
  const nameEl  = document.getElementById('invite-name');
  const emailEl = document.getElementById('invite-email');
  const unitEl  = document.getElementById('invite-unit');
  const errEl   = document.getElementById('invite-error');
  const btn     = document.getElementById('send-invite-btn');
  const name    = nameEl.value.trim();
  const email   = emailEl.value.trim();
  const unitId  = unitEl.value;
  errEl.style.display = 'none';
  if (!name)  { showInviteError('Please enter a full name.'); return; }
  if (!email) { showInviteError('Please enter an email address.'); return; }
  btn.disabled = true; btn.textContent = 'Sending...';
  try {
    const formData = new FormData();
    formData.append('full_name', name); formData.append('email', email);
    if (unitId) formData.append('course_unit_id', unitId);
    const resp = await fetch(`${DAZZLING_URL}/invite-coordinator`, {
      method: 'POST', headers: { 'Authorization': `Bearer ${currentToken}` }, body: formData,
    });
    const result = await resp.json();
    if (resp.ok) { closeInviteModal(); showToast(`✓ Invite sent to ${email}`, 'success'); await loadCoordinators(); }
    else { showInviteError(result.detail || 'Invite failed. Please try again.'); btn.disabled = false; btn.textContent = 'Send Invite'; }
  } catch (e) { showInviteError(`Error: ${e.message}`); btn.disabled = false; btn.textContent = 'Send Invite'; }
}
function showInviteError(msg) { const el = document.getElementById('invite-error'); el.textContent = msg; el.style.display = 'block'; }

function openRemoveCoordModal(coordId, coordName) {
  pendingRemoveCoordId = coordId; pendingRemoveCoordName = coordName;
  document.getElementById('remove-coord-modal-msg').textContent =
    `Are you sure you want to remove ${coordName || 'this coordinator'}? They will lose access immediately.`;
  document.getElementById('remove-coord-modal').classList.add('visible');
}
function closeRemoveCoordModal() {
  pendingRemoveCoordId = null; pendingRemoveCoordName = null;
  document.getElementById('remove-coord-modal').classList.remove('visible');
}
async function confirmRemoveCoord() {
  if (!pendingRemoveCoordId || !currentToken) return;
  const coordId = pendingRemoveCoordId; closeRemoveCoordModal();
  try {
    const resp = await fetch(`${DAZZLING_URL}/coordinators/${coordId}`, {
      method: 'DELETE', headers: { 'Authorization': `Bearer ${currentToken}` },
    });
    if (resp.ok) { showToast('Coordinator removed', 'success'); await loadCoordinators(); }
    else { const err = await resp.json().catch(() => ({})); showToast(`Failed: ${err.detail || 'Unknown error'}`, 'error'); }
  } catch (e) { showToast(`Error: ${e.message}`, 'error'); }
}

// ── Load institutions ──────────────────────────────────────────────────────
async function loadInstitutions() {
  if (!currentToken) return;
  if (!isSuperAdmin && !currentInstitutionId) return;
  const statusFilter = document.getElementById('filter-inst-status').value;
  const filterQuery  = statusFilter ? `status=${statusFilter}` : '';
  const instQuery    = !isSuperAdmin && currentInstitutionId ? `institution_id=${currentInstitutionId}` : '';
  const sep          = filterQuery && instQuery ? '&' : '';
  const query        = filterQuery || instQuery ? `?${filterQuery}${sep}${instQuery}` : '';
  document.getElementById('institutions-table-wrap').innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading institutions...</div>';
  try {
    const resp = await fetch(`${DAZZLING_URL}/admin/institutions${query}`, {
      headers: { 'Authorization': `Bearer ${currentToken}` }
    });
    const data = resp.ok ? await resp.json() : { institutions: [] };
    const institutions = data.institutions || [];
    if (!institutions.length) {
      document.getElementById('institutions-table-wrap').innerHTML = '<div class="loading">No institutions found.</div>';
      return;
    }
    const rows = institutions.map(inst => {
      const statusBadge =
        inst.status === 'active'    ? '<span class="badge badge-active">Active</span>' :
        inst.status === 'pending'   ? '<span class="badge badge-pending">Pending</span>' :
        inst.status === 'suspended' ? '<span class="badge badge-suspended">Suspended</span>' : '—';
      let actions = '';
      if (isSuperAdmin) {
        if (inst.status === 'pending') actions = `<button class="btn-approve" onclick="updateInstStatus('${inst.id}', 'active')">Approve</button><button class="btn-suspend" onclick="updateInstStatus('${inst.id}', 'suspended')">Suspend</button>`;
        else if (inst.status === 'active') actions = `<button class="btn-suspend" onclick="updateInstStatus('${inst.id}', 'suspended')">Suspend</button>`;
        else if (inst.status === 'suspended') actions = `<button class="btn-reactivate" onclick="updateInstStatus('${inst.id}', 'active')">Reactivate</button>`;
      }
      return `<tr>
        <td><strong>${escapeHtml(inst.id)}</strong></td>
        <td>${escapeHtml(inst.name || '—')}</td>
        <td style="color:var(--muted)">${escapeHtml(inst.admin_email || '—')}</td>
        <td>${escapeHtml(inst.plan || '—')}</td>
        <td>${statusBadge}</td><td>${actions}</td>
      </tr>`;
    }).join('');
    document.getElementById('institutions-table-wrap').innerHTML = `
      <table>
        <thead><tr><th>ID</th><th>Name</th><th>Admin Email</th><th>Plan</th><th>Status</th><th>Actions</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="table-footer">${institutions.length} institution${institutions.length !== 1 ? 's' : ''}</div>`;
  } catch (e) {
    document.getElementById('institutions-table-wrap').innerHTML =
      `<div class="loading" style="color:var(--red)">Failed to load: ${escapeHtml(e.message)}</div>`;
  }
}

async function updateInstStatus(institutionId, newStatus) {
  if (!currentToken || !isSuperAdmin) return;
  const actionLabel = newStatus === 'active' ? 'Approving' : newStatus === 'suspended' ? 'Suspending' : 'Updating';
  showToast(`${actionLabel} ${institutionId}...`, 'success');
  try {
    const formData = new FormData();
    formData.append('status', newStatus);
    const resp = await fetch(`${DAZZLING_URL}/admin/institutions/${institutionId}/status`, {
      method: 'PATCH', headers: { 'Authorization': `Bearer ${currentToken}` }, body: formData,
    });
    if (resp.ok) {
      const label = newStatus === 'active' ? 'approved' : newStatus === 'suspended' ? 'suspended' : 'updated';
      showToast(`✓ ${institutionId} ${label}`, 'success');
      await loadInstitutions(); await fetchPendingCount();
    } else {
      const err = await resp.json(); showToast(`Failed: ${err.detail || 'Unknown error'}`, 'error');
    }
  } catch (e) { showToast(`Error: ${e.message}`, 'error'); }
}

function openDeleteModal(studentId, studentName) {
  pendingDeleteId = studentId; pendingDeleteName = studentName;
  document.getElementById('delete-modal-msg').textContent =
    `Are you sure you want to delete ${studentName}? This will remove all their photos and attendance records permanently.`;
  document.getElementById('delete-modal').classList.add('visible');
}
function closeDeleteModal() {
  pendingDeleteId = null; pendingDeleteName = null;
  document.getElementById('delete-modal').classList.remove('visible');
}
async function confirmDelete() {
  if (!pendingDeleteId || !currentToken) return;
  const studentId = pendingDeleteId; closeDeleteModal();
  try {
    const resp = await fetch(`${DAZZLING_URL}/students/${studentId}`, {
      method: 'DELETE', headers: { 'Authorization': `Bearer ${currentToken}` },
    });
    if (resp.ok) { showToast('Student deleted', 'success'); await loadStudents(); await loadData(); }
    else { const err = await resp.json(); showToast(`Failed: ${err.detail || 'Unknown error'}`, 'error'); }
  } catch (e) { showToast(`Error: ${e.message}`, 'error'); }
}

function applyFilters() {
  const inst   = document.getElementById('filter-inst').value;
  const status = document.getElementById('filter-status').value;
  let filtered = allRecords;
  if (inst)   filtered = filtered.filter(r => r.institution_id === inst);
  if (status) filtered = filtered.filter(r => r.verified === status);
  renderTable(filtered);
}

function renderTable(records) {
  if (!records.length) {
    document.getElementById('table-wrap').innerHTML = '<div class="loading">No records found.</div>';
    return;
  }
  const rows = records.map(r => {
    const ts   = r.timestamp ? new Date(r.timestamp).toLocaleString('en-GB', { timeZone: 'Africa/Nairobi' }) : '—';
    const conf = r.confidence ? (r.confidence * 100).toFixed(1) + '%' : '—';
    const badge = r.verified === 'success' ? '<span class="badge badge-success">Success</span>' :
      r.verified === 'spoof' ? '<span class="badge badge-spoof">Spoof</span>' : '<span class="badge badge-failed">Failed</span>';
    const unitCell = r.course_unit_name
      ? `<span style="color:var(--blue);font-size:0.78rem;">${r.course_unit_name}</span>`
      : `<span style="color:var(--muted);font-size:0.78rem;">—</span>`;
    return `<tr><td>${r.student_id || '—'}</td><td>${badge}</td><td>${conf}</td><td>${unitCell}</td><td>${r.institution_id || '—'}</td><td style="color:var(--muted)">${ts}</td></tr>`;
  }).join('');
  document.getElementById('table-wrap').innerHTML = `
    <table>
      <thead><tr><th>Student ID</th><th>Status</th><th>Confidence</th><th>Course Unit</th><th>Institution</th><th>Time (EAT)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="table-footer">${records.length} records</div>`;
}

function exportCSV() {
  const inst   = document.getElementById('filter-inst').value;
  const status = document.getElementById('filter-status').value;
  let filtered = allRecords;
  if (inst)   filtered = filtered.filter(r => r.institution_id === inst);
  if (status) filtered = filtered.filter(r => r.verified === status);
  const header = ['student_id', 'verified', 'confidence', 'institution_id', 'course_unit_name', 'timestamp_EAT'];
  const rows = filtered.map(r => [
    r.student_id ?? '',
    r.verified ?? '',
    r.confidence ?? '',
    r.institution_id ?? '',
    r.course_unit_name ?? 'N/A',
    r.timestamp ? new Date(r.timestamp).toLocaleString('en-GB', { timeZone: 'Africa/Nairobi' }) : ''
  ].join(','));
  const csv    = [header.join(','), ...rows].join('\n');
  const blob   = new Blob([csv], { type: 'text/csv' });
  const url    = URL.createObjectURL(blob);
  const a      = document.createElement('a');
  a.href = url; a.download = `attendance_${new Date().toISOString().slice(0,10)}.csv`; a.click();
  URL.revokeObjectURL(url);
}

async function handleUpgrade(plan) {
  const { data: { user } } = await client.auth.getUser();
  if (!user) return alert('Please log in first.');
  const { data: profile } = await client.from('profiles').select('full_name, institution_id').eq('id', user.id).single();
  const nameParts = (profile?.full_name || 'User Name').split(' ');
  const resp = await fetch('/api/cart/create-cart', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plan, email: user.email, first_name: nameParts[0] || 'User', last_name: nameParts[1] || 'Name', phone: '', institution_id: profile?.institution_id || '' })
  });
  const data = await resp.json();
  if (data.redirect_url) window.location.href = data.redirect_url;
  else alert('Payment failed: ' + JSON.stringify(data));
}

// ── Idle timeout (5 mins) ──────────────────────────────────────────────────
const IDLE_TIMEOUT_MS = 5 * 60 * 1000;
let idleTimer = null;

function resetIdleTimer() {
  clearTimeout(idleTimer);
  if (!currentToken) return;
  localStorage.setItem('lastActivity', Date.now());
  idleTimer = setTimeout(async () => {
    if (currentToken) {
      showToast('Session expired due to inactivity. Signing out...', 'error');
      await new Promise(r => setTimeout(r, 2000));
      await logout();
    }
  }, IDLE_TIMEOUT_MS);
}

['mousemove', 'keydown', 'click', 'scroll', 'touchstart'].forEach(event => {
  document.addEventListener(event, resetIdleTimer, { passive: true });
});

const _origInitDashboard = initDashboard;
initDashboard = async function(session, isFreshLogin = false) {
  if (!isFreshLogin) {
    const lastActivityRaw = localStorage.getItem('lastActivity');
    const lastActivity = lastActivityRaw ? parseInt(lastActivityRaw, 10) : null;
    if (lastActivity && Date.now() - lastActivity > IDLE_TIMEOUT_MS) {
      showToast('Session expired due to inactivity. Signing out...', 'error');
      await new Promise(r => setTimeout(r, 2000));
      await logout();
      return;
    }
  }
  await _origInitDashboard(session);
  resetIdleTimer();
};

// ── Enter key on login ─────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && document.getElementById('login-screen').style.display !== 'none') {
    login();
  }
});
