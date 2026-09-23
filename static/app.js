const recordButton = document.getElementById("record-button");
const results = document.getElementById("results");

let recorder;
let chunks = [];
let isStarting = false;

function showMessage(message) {
  results.replaceChildren();
  const messageElement = document.createElement("p");
  messageElement.textContent = message;
  results.append(messageElement);
}

function showVerification(result) {
  results.replaceChildren();
  const claim = result.results?.[0];
  if (!claim) {
    showMessage("No checkable claim was found in the transcription.");
    return;
  }

  const title = document.createElement("h2");
  title.textContent = "Verification result";
  results.append(title);

  const spoken = document.createElement("p");
  spoken.className = "spoken-claim";
  spoken.textContent = `Spoken claim: ${claim.claim_text}`;
  results.append(spoken);

  const verdict = document.createElement("p");
  verdict.className = `verdict verdict-${claim.verdict.toLowerCase()}`;
  verdict.textContent = `${claim.verdict} (${Math.round(claim.confidence * 100)}% confidence)`;
  results.append(verdict);

  const explanation = document.createElement("p");
  explanation.className = "explanation";
  explanation.textContent = claim.explanation;
  results.append(explanation);

  if (claim.citations?.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Evidence";
    results.append(heading);
    const evidenceList = document.createElement("ul");
    evidenceList.className = "evidence-list";
    claim.citations.forEach((citation) => {
      const item = document.createElement("li");
      const source = document.createElement("strong");
      source.textContent = citation.source_name;
      item.append(source);
      const text = document.createElement("p");
      text.textContent = citation.text;
      item.append(text);
      if (citation.source_url) {
        const link = document.createElement("a");
        link.href = citation.source_url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = citation.source_url;
        item.append(link);
      }
      evidenceList.append(item);
    });
    results.append(evidenceList);
  }
}

async function verifyText(text) {
  const response = await fetch("/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Verification failed.");
  showVerification(result);
}

function resetRecordButton() {
  recordButton.disabled = false;
  recordButton.textContent = "RECORD";
}

recordButton.addEventListener("click", async () => {
  if (recorder?.state === "recording") {
    recorder.stop();
    return;
  }
  if (isStarting) return;

  results.replaceChildren();
  showMessage("Listening...");
  recordButton.textContent = "STOP";
  isStarting = true;

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showMessage("Audio recording is not supported in this browser.");
    resetRecordButton();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const audio = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      recorder = undefined;
      showMessage("Transcribing...");
      recordButton.disabled = true;
      try {
        const form = new FormData();
        form.append("file", audio, "recording.webm");
        const response = await fetch("/transcribe", { method: "POST", body: form });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Transcription failed.");
        showMessage(`Spoken claim: ${result.text}\n\nVerifying...`);
        await verifyText(result.text);
      } catch (error) {
        showMessage(error.message || "Transcription or verification failed.");
      } finally {
        resetRecordButton();
      }
    };
    recorder.start();
    isStarting = false;
    recordButton.textContent = "STOP";
  } catch (error) {
    isStarting = false;
    showMessage(error.name === "NotAllowedError"
      ? "Microphone permission was denied."
      : "Unable to access the microphone.");
    resetRecordButton();
  }
});
