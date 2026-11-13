// When served by FastAPI the API is on the same address. If index.html is opened directly, use localhost.
const API = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
const $ = (id) => document.getElementById(id);
let selectedFile = null;

// ---------- Tabs ----------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab, .view").forEach((el) => el.classList.remove("active"));
    tab.classList.add("active");
    $(tab.dataset.view).classList.add("active");
    if (tab.dataset.view === "history") loadHistory();
  });
});

// ---------- File selection ----------
const dropzone = $("dropzone");
dropzone.addEventListener("click", () => $("fileInput").click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter") $("fileInput").click(); });
$("fileInput").addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

function setFile(file) {
  hideError(); $("result").hidden = true;
  if (!file) return;
  if (!["image/jpeg", "image/png"].includes(file.type)) return showError("Please choose a JPG or PNG image.");
  if (file.size > 10 * 1024 * 1024) return showError("File is too large (max 10 MB).");
  selectedFile = file;
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false; $("dropPrompt").hidden = true;
  $("analyzeBtn").disabled = false; $("resetBtn").disabled = false;
}

$("resetBtn").addEventListener("click", () => {
  selectedFile = null; $("fileInput").value = "";
  $("preview").hidden = true; $("dropPrompt").hidden = false;
  $("analyzeBtn").disabled = true; $("resetBtn").disabled = true;
  $("result").hidden = true; hideError();
});

// ---------- Analyze ----------
$("analyzeBtn").addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError(); $("result").hidden = true; $("loader").hidden = false; $("analyzeBtn").disabled = true;
  const form = new FormData();
  form.append("file", selectedFile);
  try {
    const res = await fetch(`${API}/predict`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Something went wrong.");
    showResult(data);
  } catch (err) {
    showError(err.message === "Failed to fetch" ? "Cannot reach the server. Is `python run.py` running?" : err.message);
  } finally {
    $("loader").hidden = true; $("analyzeBtn").disabled = false;
  }
});

function showResult(d) {
  const cls = d.label.toLowerCase();
  const pct = (d.confidence * 100).toFixed(1);
  $("badge").textContent = d.label; $("badge").className = `badge ${cls}`;
  $("confText").textContent = `${pct}% confidence`;
  $("barFill").className = `bar-fill ${cls}`; $("barFill").style.width = "0";
  setTimeout(() => ($("barFill").style.width = pct + "%"), 50);
  $("warning").hidden = !d.low_confidence;
  $("origImg").src = $("preview").src; $("heatImg").src = d.heatmap;
  $("pNormal").textContent = (d.probabilities.NORMAL * 100).toFixed(1) + "%";
  $("pPneumonia").textContent = (d.probabilities.PNEUMONIA * 100).toFixed(1) + "%";
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth" });
}

// ---------- History ----------
async function loadHistory() {
  try {
    const rows = await (await fetch(`${API}/history`)).json();
    const body = $("historyBody"); body.innerHTML = "";
    $("historyEmpty").hidden = rows.length > 0;
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      [r.created_at, r.filename, null, (r.confidence * 100).toFixed(1) + "%"].forEach((val, i) => {
        const td = document.createElement("td");
        if (i === 2) { const s = document.createElement("span"); s.className = `tag ${r.label}`; s.textContent = r.label; td.appendChild(s); }
        else td.textContent = val;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  } catch { $("historyEmpty").hidden = false; }
}

function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }
function hideError() { $("error").hidden = true; }
