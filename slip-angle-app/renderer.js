const BASE = "http://localhost:5000";

const form = document.getElementById("analyzeForm");
const results = document.getElementById("results");
const plotImg = document.getElementById("plotImg");
const statsDiv = document.getElementById("stats");
const csvLink = document.getElementById("csvLink");
const rawJson = document.getElementById("rawJson");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  results.classList.remove("hidden");
  statsDiv.textContent = "Processing…";

  const fd = new FormData(form);

  try {
    const res = await fetch(`${BASE}/api/analyze`, {
      method: "POST",
      body: fd
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    const plotURL = `${BASE}${data.links.plot_png}?t=${Date.now()}`;
    const csvURL = `${BASE}${data.links.csv}`;

    const mean = data.stats.mean_abs_beta_deg.toFixed(2);
    const peak = data.stats.peak_abs_beta_deg.toFixed(2);

    statsDiv.textContent = `mean |β| = ${mean}°,  peak |β| = ${peak}°`;

    plotImg.src = plotURL;
    csvLink.href = csvURL;
    rawJson.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    statsDiv.textContent = "Error: " + err.message;
  }
});
