const API = "/api";

function token() {
  return document.getElementById("token").value;
}

async function get(path) {
  const r = await fetch(API + path, {
    headers: { "x-harvy-token": token() }
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function post(path, body = {}) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "x-harvy-token": token(), "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadStatus() {
  try {
    const data = await get("/status");
    renderPlayers(data.players);
    renderDeaths(data.recent_deaths);
    document.getElementById("status").textContent =
      `${data.players.length} players known | ${data.recent_combat.length} recent combats`;
  } catch (e) {
    document.getElementById("status").textContent = "Error: " + e.message;
  }
}

function renderPlayers(players) {
  const tbody = document.getElementById("players");
  tbody.innerHTML = "";
  for (const p of players) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${p.name}</td>
      <td>${p.threat_level}</td>
      <td>${new Date(p.last_seen_time * 1000).toLocaleString()}</td>
      <td>
        <button onclick="setThreat('${p.name}', 'ally')">Trust</button>
        <button onclick="setThreat('${p.name}', 'hostile')">Hostile</button>
        <button onclick="setThreat('${p.name}', 'neutral')">Neutral</button>
      </td>
    `;
    tbody.appendChild(tr);
  }
}

function renderDeaths(deaths) {
  const tbody = document.getElementById("deaths");
  tbody.innerHTML = "";
  for (const d of deaths) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${new Date(d.time * 1000).toLocaleString()}</td>
      <td>${d.cause || "-"}</td>
      <td>${d.killer_name || "-"}</td>
      <td>${[d.pos_x, d.pos_y, d.pos_z].map(v => v?.toFixed(1)).join(", ")}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function sendCmd(cmd) {
  try {
    const args = document.getElementById("cmdArgs").value;
    await post(`/command/${cmd}?args=${encodeURIComponent(args)}`);
    document.getElementById("cmdStatus").textContent = `Sent: ${cmd}`;
  } catch (e) {
    document.getElementById("cmdStatus").textContent = "Error: " + e.message;
  }
}

async function sendCustom() {
  const raw = document.getElementById("cmdArgs").value;
  const parts = raw.split(" ");
  sendCmd(parts[0]);
}

async function setThreat(name, threat) {
  try {
    await post(`/players/${encodeURIComponent(name)}/threat?threat=${threat}`);
    loadStatus();
  } catch (e) {
    alert(e.message);
  }
}

setInterval(loadStatus, 2000);
loadStatus();
