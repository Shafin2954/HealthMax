let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let diseaseChart = null;

// ── Text submission ──
async function submitText() {
  const text = document.getElementById("symptom-input").value.trim();
  if (!text) {
    alert("অনুগ্রহ করে উপসর্গ লিখুন।");
    return;
  }
  await runTriage({ mode: "text", text });
}

// ── Voice recording toggle ──
async function toggleRecording() {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      await runTriage({ mode: "voice", audioBlob });
    };

    mediaRecorder.start();
    isRecording = true;
    document.getElementById("record-btn").textContent = "⏹ থামান";
    document.getElementById("record-btn").classList.add("recording");
    document.getElementById("recording-indicator").classList.remove("hidden");
  } catch (err) {
    alert("মাইক্রোফোন অ্যাক্সেস করতে পারছি না: " + err.message);
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    isRecording = false;
    document.getElementById("record-btn").textContent = "🎤 ভয়েস রেকর্ড";
    document.getElementById("record-btn").classList.remove("recording");
    document.getElementById("recording-indicator").classList.add("hidden");
  }
}

// ── Core triage runner ──
async function runTriage({ mode, text, audioBlob }) {
  showLoading(true);
  hideResults();

  try {
    let data;
    if (mode === "text") {
      const res = await fetch("/api/triage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      data = await res.json();
    } else {
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.wav");
      const res = await fetch("/api/triage/voice", { method: "POST", body: formData });
      data = await res.json();

      if (data.transcript) {
        document.getElementById("transcript-text").textContent = data.transcript;
        document.getElementById("transcript-box").classList.remove("hidden");
        document.getElementById("symptom-input").value = data.transcript;
      }

      if (data.low_confidence) {
        showLoading(false);
        alert(data.fallback_message);
        return;
      }
    }

    renderResults(data);
  } catch (err) {
    alert("সার্ভারে সমস্যা হয়েছে। আবার চেষ্টা করুন।");
    console.error(err);
  } finally {
    showLoading(false);
  }
}

// ── Render results ──
function renderResults(data) {
  document.getElementById("results").classList.remove("hidden");

  // Emergency banner
  const emergencyBanner = document.getElementById("emergency-banner");
  if (data.emergency_override) {
    emergencyBanner.classList.remove("hidden");
  } else {
    emergencyBanner.classList.add("hidden");
  }

  // NER entities
  renderNEREntities(data.ner_entities || {});

  // Disease chart
  renderDiseaseChart(data.top_diseases || []);

  // Urgency badge
  const urgencyEl = document.getElementById("urgency-badge");
  urgencyEl.textContent = data.urgency_label_bn || "জরুরি";
  urgencyEl.className = "urgency-badge urgency-" + (data.urgency_level || "URGENT").toLowerCase();

  // Facility
  document.getElementById("facility-text").textContent =
    data.facility_recommendation || "উপজেলা স্বাস্থ্য কমপ্লেক্স";

  // Drug cards
  renderDrugCards(data.drug_recommendations || []);

  // LLM response
  document.getElementById("llm-response").textContent =
    data.llm_response || "পরামর্শ লোড হচ্ছে...";

  // Scroll to results
  document.getElementById("results").scrollIntoView({ behavior: "smooth" });
}

function renderNEREntities(entities) {
  const container = document.getElementById("ner-entities");
  container.innerHTML = "";

  const typeMap = {
    symptoms: { label: "উপসর্গ", cls: "tag-symptom" },
    diseases: { label: "রোগ", cls: "tag-disease" },
    medicines: { label: "ওষুধ", cls: "tag-medicine" },
  };

  let hasEntities = false;
  for (const [type, config] of Object.entries(typeMap)) {
    const items = entities[type] || [];
    items.forEach((item) => {
      hasEntities = true;
      const tag = document.createElement("span");
      tag.className = `entity-tag ${config.cls}`;
      tag.textContent = `${item} (${config.label})`;
      container.appendChild(tag);
    });
  }

  if (!hasEntities) {
    container.textContent = "কোনো নির্দিষ্ট উপসর্গ শনাক্ত হয়নি।";
  }
}

function renderDiseaseChart(diseases) {
  const ctx = document.getElementById("disease-chart").getContext("2d");
  if (diseaseChart) diseaseChart.destroy();

  if (!diseases.length) {
    ctx.canvas.parentElement.innerHTML += "<p>পর্যাপ্ত তথ্য নেই।</p>";
    return;
  }

  diseaseChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: diseases.map((d) => d.disease),
      datasets: [
        {
          label: "সম্ভাবনা (%)",
          data: diseases.map((d) => (d.probability * 100).toFixed(1)),
          backgroundColor: ["#4A90D9", "#5BA85F", "#E8A838"],
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: (v) => v + "%" } },
      },
    },
  });
}

function renderDrugCards(drugs) {
  const container = document.getElementById("drug-cards");
  container.innerHTML = "";

  if (!drugs.length) {
    container.textContent = "ডাক্তারের পরামর্শ অনুযায়ী ওষুধ নিন।";
    return;
  }

  drugs.forEach((drug) => {
    const card = document.createElement("div");
    card.className = "drug-card";
    card.innerHTML = `
      <div class="drug-name">💊 ${drug.generic_name}</div>
      <div class="drug-brand">ব্র্যান্ড: ${drug.brand_example}</div>
      <div class="drug-price">৳${drug.price_bdt} / ${drug.unit}
        ${drug.affordable ? '<span class="affordable-badge">সাশ্রয়ী 💚</span>' : ""}
      </div>
    `;
    container.appendChild(card);
  });
}

function showLoading(show) {
  document.getElementById("loading").classList.toggle("hidden", !show);
  document.getElementById("submit-btn").disabled = show;
}

function hideResults() {
  document.getElementById("results").classList.add("hidden");
  document.getElementById("transcript-box").classList.add("hidden");
}
