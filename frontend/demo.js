/**
 * HealthMax — Browser Demo JavaScript
 *
 * Responsibilities:
 *  1. MediaRecorder API — record voice, send audio blob to /api/triage/voice
 *  2. Text input — send Bangla text to /api/triage
 *  3. Render results: NER entity highlights, disease chart, urgency badge, drug list
 *  4. TTS playback — fetch /api/tts and play MP3 audio (FLEX feature)
 *
 * Collaborator instructions:
 *  - API_BASE must point to the EC2 backend URL (or localhost for development).
 *  - The renderNerEntities() function highlights symptoms (blue), diseases (red),
 *    and medicines (green) inline in the transcript text.
 *  - The disease probability chart uses Chart.js bar chart.
 *  - Keep all user-facing strings in Bangla.
 */

"use strict";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const API_BASE = window.location.origin; // Same origin when served by FastAPI
// For local dev without backend running, override:
// const API_BASE = "http://localhost:8000";

const URGENCY_COLORS = {
    EMERGENCY: "#e53e3e",  // Red
    URGENT: "#d69e2e",  // Amber
    "SELF-CARE": "#38a169", // Green
};

const URGENCY_LABELS_BN = {
    EMERGENCY: "🚨 জরুরি — এখনই হাসপাতালে যান!",
    URGENT: "⚠️ দরকারি — আজই ডাক্তার দেখান",
    "SELF-CARE": "✅ স্বাভাবিক — ঘরে যত্ন নিন",
};

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const btnRecord = document.getElementById("btn-record");
const btnRecordLabel = document.getElementById("btn-record-label");
const recordStatus = document.getElementById("record-status");
const symptomInput = document.getElementById("symptom-input");
const btnSubmit = document.getElementById("btn-submit");
const transcriptBox = document.getElementById("transcript-box");
const transcriptText = document.getElementById("transcript-text");
const loadingSpinner = document.getElementById("loading-spinner");
const resultsPanel = document.getElementById("results-panel");
const nerDisplay = document.getElementById("ner-display");
const urgencyBadge = document.getElementById("urgency-badge");
const facilityText = document.getElementById("facility-text");
const drugList = document.getElementById("drug-list");
const fullResponse = document.getElementById("full-response");
const btnTts = document.getElementById("btn-tts");
const ttsPlayer = document.getElementById("tts-player");
const btnReset = document.getElementById("btn-reset");

let diseaseChart = null;   // Chart.js instance
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// ---------------------------------------------------------------------------
// Voice recording — MediaRecorder API
// ---------------------------------------------------------------------------

/**
 * Toggle voice recording on/off.
 * When stopped, sends the recorded audio blob to /api/triage/voice.
 *
 * TODO (collaborator):
 *  - Test on Chrome (preferred), Firefox, and Safari mobile.
 *  - Confirm MIME type support: prefer 'audio/webm;codecs=opus', fallback to 'audio/ogg'.
 *  - Show live waveform animation while recording (FLEX).
 */
async function toggleRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("আপনার ব্রাউজার মাইক্রোফোন সমর্থন করে না। অনুগ্রহ করে Chrome ব্যবহার করুন।");
        return;
    }

    if (!isRecording) {
        // Start recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
                ? "audio/webm;codecs=opus"
                : "audio/ogg";

            mediaRecorder = new MediaRecorder(stream, { mimeType });
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                stream.getTracks().forEach((t) => t.stop());
                await submitVoice(audioBlob);
            };

            mediaRecorder.start(250); // collect data every 250ms
            isRecording = true;
            btnRecord.classList.add("recording");
            btnRecordLabel.textContent = "থামান";
            recordStatus.textContent = "রেকর্ডিং হচ্ছে...";
        } catch (err) {
            console.error("Microphone access denied:", err);
            alert("মাইক্রোফোনে অনুমতি দিন এবং আবার চেষ্টা করুন।");
        }
    } else {
        // Stop recording
        mediaRecorder.stop();
        isRecording = false;
        btnRecord.classList.remove("recording");
        btnRecordLabel.textContent = "কথা বলুন";
        recordStatus.textContent = "প্রক্রিয়া হচ্ছে...";
    }
}

/**
 * Send a recorded audio Blob to the voice triage endpoint.
 * @param {Blob} audioBlob — recorded audio blob
 */
async function submitVoice(audioBlob) {
    showLoading();
    try {
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        const res = await fetch(`${API_BASE}/api/triage/voice`, {
            method: "POST",
            body: formData,
        });
        await handleApiResponse(res);
    } catch (err) {
        console.error("Voice triage error:", err);
        showError("নেটওয়ার্ক সমস্যা। আবার চেষ্টা করুন।");
    } finally {
        hideLoading();
        recordStatus.textContent = "";
    }
}

// ---------------------------------------------------------------------------
// Text input submission
// ---------------------------------------------------------------------------

/**
 * Send the Bangla text from the text input to /api/triage.
 */
async function submitText() {
    const text = symptomInput.value.trim();
    if (!text) {
        alert("অনুগ্রহ করে আপনার লক্ষণ লিখুন।");
        return;
    }

    showLoading();
    try {
        const res = await fetch(`${API_BASE}/api/triage`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        await handleApiResponse(res);
    } catch (err) {
        console.error("Text triage error:", err);
        showError("নেটওয়ার্ক সমস্যা। আবার চেষ্টা করুন।");
    } finally {
        hideLoading();
    }
}

// ---------------------------------------------------------------------------
// API response handler
// ---------------------------------------------------------------------------

/**
 * Parse the triage API response and render all result widgets.
 * @param {Response} res — Fetch API response object
 */
async function handleApiResponse(res) {
    if (!res.ok) {
        const errText = await res.text();
        showError(`সার্ভার ত্রুটি (${res.status}): ${errText}`);
        return;
    }

    const data = await res.json();
    renderResults(data);
}

// ---------------------------------------------------------------------------
// Result rendering
// ---------------------------------------------------------------------------

/**
 * Render all result widgets from the API response.
 * @param {Object} data — TriageResponse JSON from the API
 */
function renderResults(data) {
    // Show transcript
    if (data.transcript) {
        transcriptText.textContent = data.transcript;
        transcriptBox.classList.remove("hidden");
    }

    // Render NER entities
    renderNerEntities(data.transcript || "", data.entities || {});

    // Render disease probability chart
    renderDiseaseChart(data.top_diseases || []);

    // Render urgency badge
    renderUrgencyBadge(data.urgency || "SELF-CARE");

    // Render facility recommendation
    facilityText.textContent = data.facility || "";

    // Render drug list
    renderDrugList(data.drugs || []);

    // Render full response text
    fullResponse.textContent = data.response_text || "";

    // Show results panel
    resultsPanel.classList.remove("hidden");
    resultsPanel.scrollIntoView({ behavior: "smooth" });
}

/**
 * Highlight NER entities inline in the transcript text.
 * Symptoms → blue tags, Diseases → red tags, Medicines → green tags.
 *
 * TODO (collaborator):
 *  - Replace simple span list rendering with inline text highlighting
 *    that shows entities in context within the transcript sentence.
 *  - Use the 'start' and 'end' character offsets from the NER API if available.
 *
 * @param {string} transcript — Full Bangla transcript text
 * @param {Object} entities   — {symptoms: [], diseases: [], medicines: []}
 */
function renderNerEntities(transcript, entities) {
    nerDisplay.innerHTML = "";

    const groups = [
        { label: "লক্ষণ", items: entities.symptoms || [], cssClass: "ner-symptom" },
        { label: "রোগ", items: entities.diseases || [], cssClass: "ner-disease" },
        { label: "ওষুধ", items: entities.medicines || [], cssClass: "ner-medicine" },
    ];

    groups.forEach(({ label, items, cssClass }) => {
        if (!items.length) return;
        const groupEl = document.createElement("div");
        groupEl.className = "ner-group";

        const labelEl = document.createElement("span");
        labelEl.className = "ner-group-label";
        labelEl.textContent = `${label}: `;
        groupEl.appendChild(labelEl);

        items.forEach((item) => {
            const tag = document.createElement("span");
            tag.className = `ner-tag ${cssClass}`;
            tag.textContent = item;
            groupEl.appendChild(tag);
        });

        nerDisplay.appendChild(groupEl);
    });

    if (nerDisplay.children.length === 0) {
        nerDisplay.textContent = "কোনো লক্ষণ শনাক্ত হয়নি।";
    }
}

/**
 * Render horizontal bar chart for disease probabilities.
 * @param {Array} diseases — [{name, probability}, ...]
 */
function renderDiseaseChart(diseases) {
    if (diseaseChart) {
        diseaseChart.destroy();
        diseaseChart = null;
    }

    if (!diseases.length) return;

    const ctx = document.getElementById("disease-chart").getContext("2d");
    diseaseChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: diseases.map((d) => d.name),
            datasets: [{
                label: "সম্ভাব্যতা",
                data: diseases.map((d) => Math.round(d.probability * 100)),
                backgroundColor: ["#4299e1", "#48bb78", "#ed8936"],
                borderRadius: 6,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    ticks: { callback: (v) => `${v}%` },
                },
            },
        },
    });
}

/**
 * Render the urgency badge with the appropriate color.
 * @param {string} urgency — 'EMERGENCY' | 'URGENT' | 'SELF-CARE'
 */
function renderUrgencyBadge(urgency) {
    urgencyBadge.textContent = URGENCY_LABELS_BN[urgency] || urgency;
    urgencyBadge.style.backgroundColor = URGENCY_COLORS[urgency] || "#718096";
    urgencyBadge.className = `urgency-badge urgency-${urgency.toLowerCase().replace("-", "")}`;
}

/**
 * Render the drug recommendation list.
 * Marks affordable items (below ৳5) with the "সাশ্রয়ী" badge.
 * @param {Array} drugs — [{generic_name, brand_example, price_bdt, unit, affordable}, ...]
 */
function renderDrugList(drugs) {
    drugList.innerHTML = "";
    if (!drugs.length) {
        drugList.innerHTML = "<li>কোনো ওষুধ পরামর্শ নেই।</li>";
        return;
    }
    drugs.forEach((drug) => {
        const li = document.createElement("li");
        li.className = "drug-item";
        li.innerHTML = `
      <span class="drug-name">${drug.generic_name}</span>
      <span class="drug-brand">(${drug.brand_example})</span>
      <span class="drug-price">৳${drug.price_bdt}/${drug.unit}</span>
      ${drug.affordable ? '<span class="drug-affordable-badge">সাশ্রয়ী</span>' : ""}
    `;
        drugList.appendChild(li);
    });
}

// ---------------------------------------------------------------------------
// TTS playback (FLEX feature)
// ---------------------------------------------------------------------------

/**
 * Fetch TTS audio from /api/tts and play it in the browser.
 *
 * TODO (collaborator — FLEX):
 *  - Implement the /api/tts endpoint in main.py (see tts.py).
 *  - Test TTS audio on mobile Chrome — ensure autoplay policies allow it.
 */
async function playTts() {
    const text = fullResponse.textContent;
    if (!text) return;

    btnTts.disabled = true;
    btnTts.textContent = "লোড হচ্ছে...";

    try {
        const res = await fetch(`${API_BASE}/api/tts`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error("TTS unavailable");
        const audioBlob = await res.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        ttsPlayer.src = audioUrl;
        ttsPlayer.hidden = false;
        await ttsPlayer.play();
    } catch (err) {
        console.warn("TTS playback failed:", err);
        alert("TTS পরিষেবা এই মুহূর্তে উপলব্ধ নয়।");
    } finally {
        btnTts.disabled = false;
        btnTts.textContent = "🔊 শুনুন";
    }
}

// ---------------------------------------------------------------------------
// UI utility helpers
// ---------------------------------------------------------------------------

function showLoading() {
    loadingSpinner.classList.remove("hidden");
    resultsPanel.classList.add("hidden");
    btnSubmit.disabled = true;
    btnRecord.disabled = true;
}

function hideLoading() {
    loadingSpinner.classList.add("hidden");
    btnSubmit.disabled = false;
    btnRecord.disabled = false;
}

function showError(message) {
    hideLoading();
    alert(`ত্রুটি: ${message}`);
}

function resetForm() {
    symptomInput.value = "";
    transcriptText.textContent = "";
    transcriptBox.classList.add("hidden");
    resultsPanel.classList.add("hidden");
    nerDisplay.innerHTML = "";
    facilityText.textContent = "";
    drugList.innerHTML = "";
    fullResponse.textContent = "";
    if (diseaseChart) { diseaseChart.destroy(); diseaseChart = null; }
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

btnRecord.addEventListener("click", toggleRecording);
btnSubmit.addEventListener("click", submitText);
btnTts.addEventListener("click", playTts);
btnReset.addEventListener("click", resetForm);

// Allow Enter key (without Shift) to submit text
symptomInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submitText();
    }
});
