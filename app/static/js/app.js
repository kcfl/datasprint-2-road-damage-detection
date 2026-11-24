/**
 * Civic Road Damage Grievance Portal Frontend Logic
 * Implements mobile camera capture, Geolocation API, AI analysis preview, and SMTP dispatch.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropzone = document.getElementById("dropzone");
  const photoInput = document.getElementById("photo-input");
  const btnCameraTrigger = document.getElementById("btn-camera-trigger");
  const dropzonePrompt = document.getElementById("dropzone-prompt");
  const previewBox = document.getElementById("preview-box");
  const imagePreview = document.getElementById("image-preview");
  const btnRetake = document.getElementById("btn-retake");

  const btnGps = document.getElementById("btn-gps");
  const gpsStatus = document.getElementById("gps-status");
  const inputLat = document.getElementById("input-lat");
  const inputLon = document.getElementById("input-lon");
  const inputCity = document.getElementById("input-city");
  const inputLandmark = document.getElementById("input-landmark");
  const inputName = document.getElementById("input-name");
  const inputPhone = document.getElementById("input-phone");
  const inputEmail = document.getElementById("input-email");
  const mapsLinkWrapper = document.getElementById("maps-link-wrapper");
  const mapsLink = document.getElementById("maps-link");

  const btnAnalyze = document.getElementById("btn-analyze");
  const sectionStep3 = document.getElementById("section-step3");
  const priorityBadge = document.getElementById("priority-badge");
  const urgencyNoteText = document.getElementById("urgency-note-text");
  const countD40 = document.getElementById("count-d40");
  const countD20 = document.getElementById("count-d20");
  const countD00 = document.getElementById("count-d00");
  const countD10 = document.getElementById("count-d10");
  const annotatedImage = document.getElementById("annotated-image");

  const muniName = document.getElementById("muni-name");
  const muniReason = document.getElementById("muni-reason");
  const muniHelpline = document.getElementById("muni-helpline");
  const muniPortal = document.getElementById("muni-portal");
  const recipientEmail = document.getElementById("recipient-email");

  const draftSubject = document.getElementById("draft-subject");
  const draftBody = document.getElementById("draft-body");
  const btnCopyDraft = document.getElementById("btn-copy-draft");
  const btnBackToStep2 = document.getElementById("btn-back-to-step2");
  const btnConfirmSend = document.getElementById("btn-confirm-send");

  const statusBanner = document.getElementById("status-banner");
  const modalOverlay = document.getElementById("modal-overlay");
  const modalIcon = document.getElementById("modal-icon");
  const modalTitle = document.getElementById("modal-title");
  const modalMessage = document.getElementById("modal-message");
  const btnModalClose = document.getElementById("btn-modal-close");

  const stepNode1 = document.getElementById("step-node-1");
  const stepNode2 = document.getElementById("step-node-2");
  const stepNode3 = document.getElementById("step-node-3");
  const statusModelBadge = document.getElementById("status-model-badge");

  let selectedFile = null;
  let lastAnalysisResult = null;

  // 1. Fetch Backend System Status
  async function checkSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        if (data.active_weights) {
          statusModelBadge.textContent = `AI: ${data.active_weights}`;
        }
      }
    } catch (e) {
      console.warn("Could not fetch status from backend:", e);
    }
  }
  checkSystemStatus();

  // 2. Banner Notification Helper
  function showBanner(message, type = "info") {
    statusBanner.className = `banner banner-${type}`;
    statusBanner.textContent = message;
    statusBanner.classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function hideBanner() {
    statusBanner.classList.add("hidden");
  }

  // 3. Photo Capture & Upload Handling
  btnCameraTrigger.addEventListener("click", () => {
    photoInput.click();
  });

  photoInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  // Drag and Drop
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
      showBanner("Please select a valid image file (JPEG, PNG, WEBP).", "error");
      return;
    }
    selectedFile = file;
    hideBanner();

    const reader = new FileReader();
    reader.onload = (event) => {
      imagePreview.src = event.target.result;
      dropzonePrompt.classList.add("hidden");
      previewBox.classList.remove("hidden");
      stepNode1.classList.add("completed");
      stepNode2.classList.add("active");
    };
    reader.readAsDataURL(file);
  }

  btnRetake.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    photoInput.value = "";
    previewBox.classList.add("hidden");
    dropzonePrompt.classList.remove("hidden");
    sectionStep3.classList.add("hidden");
    stepNode1.classList.remove("completed");
    stepNode3.classList.remove("completed", "active");
  });

  // 4. Geolocation API Integration
  btnGps.addEventListener("click", () => {
    if (!navigator.geolocation) {
      showBanner("Geolocation is not supported by your browser.", "error");
      return;
    }

    gpsStatus.textContent = "⏳ Acquiring high-precision GPS lock...";
    btnGps.disabled = true;

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        const accuracy = Math.round(position.coords.accuracy);

        inputLat.value = lat.toFixed(6);
        inputLon.value = lon.toFixed(6);
        gpsStatus.textContent = `📍 GPS Fixed (±${accuracy}m accuracy)`;
        btnGps.disabled = false;

        const mapsUrl = `https://www.google.com/maps?q=${lat},${lon}`;
        mapsLink.href = mapsUrl;
        mapsLinkWrapper.classList.remove("hidden");

        // Reverse-geocode to auto-populate city if field is currently blank
        try {
          const res = await fetch("/api/geocode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lat, lon }),
          });
          if (res.ok) {
            const data = await res.json();
            if (data.geo && data.geo.city && !inputCity.value) {
              inputCity.value = data.geo.city;
            }
          }
        } catch (err) {
          console.warn("Reverse geocode lookup warning:", err);
        }
      },
      (error) => {
        btnGps.disabled = false;
        let errMsg = "Could not retrieve GPS location.";
        if (error.code === error.PERMISSION_DENIED) {
          errMsg = "Location permission denied. Please enter your city and landmark manually.";
        } else if (error.code === error.POSITION_UNAVAILABLE) {
          errMsg = "GPS position unavailable. Please enter your city manually.";
        } else if (error.code === error.TIMEOUT) {
          errMsg = "GPS request timed out. Please enter your city manually.";
        }
        gpsStatus.textContent = "⚠️ " + errMsg;
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  });

  // 5. Run AI Analysis
  btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) {
      showBanner("Please upload or take a photo of the damaged road first (Step 1).", "error");
      return;
    }

    hideBanner();
    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `<span>⏳ Auditing Distress with AI (CPU)...</span>`;

    const formData = new FormData();
    formData.append("image", selectedFile);

    const latVal = parseFloat(inputLat.value);
    const lonVal = parseFloat(inputLon.value);
    if (!isNaN(latVal)) formData.append("lat", latVal);
    if (!isNaN(lonVal)) formData.append("lon", lonVal);

    if (inputCity.value.trim()) formData.append("city", inputCity.value.trim());
    if (inputLandmark.value.trim()) formData.append("landmark", inputLandmark.value.trim());
    if (inputName.value.trim()) formData.append("citizen_name", inputName.value.trim());
    if (inputPhone.value.trim()) formData.append("citizen_phone", inputPhone.value.trim());
    if (inputEmail.value.trim()) formData.append("citizen_email", inputEmail.value.trim());

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Analysis request failed.");
      }

      const result = await response.json();
      lastAnalysisResult = result;
      renderAnalysisResult(result);

    } catch (err) {
      showBanner(`Analysis failed: ${err.message}`, "error");
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = `🚀 Run AI Damage Audit`;
    }
  });

  // 6. Render Analysis Result in Step 3
  function renderAnalysisResult(data) {
    const det = data.detection;
    const muni = data.municipality;
    const draft = data.draft_email;

    // Priority pill
    priorityBadge.textContent = det.priority.toUpperCase();
    priorityBadge.style.backgroundColor = det.priority_color || "#ef4444";
    urgencyNoteText.textContent = det.urgency_note;

    // Counts
    countD40.textContent = det.counts["D40"] || 0;
    countD20.textContent = det.counts["D20"] || 0;
    countD00.textContent = det.counts["D00"] || 0;
    countD10.textContent = det.counts["D10"] || 0;

    // Annotated Image
    annotatedImage.src = det.annotated_image_data_url;

    // Municipality details
    muniName.textContent = muni.name;
    muniReason.textContent = muni.match_reason || "Assigned municipal authority";
    muniHelpline.textContent = muni.helpline || "112";
    if (muni.portal_url) {
      muniPortal.href = muni.portal_url;
      muniPortal.textContent = muni.portal_url.replace("https://", "");
      muniPortal.classList.remove("hidden");
    } else {
      muniPortal.classList.add("hidden");
    }
    recipientEmail.value = muni.grievance_email || "grievance-roads@municipal.gov.in";

    // Draft email
    draftSubject.value = draft.subject;
    draftBody.value = draft.body;

    // Reveal step 3
    sectionStep3.classList.remove("hidden");
    stepNode2.classList.add("completed");
    stepNode3.classList.add("active");

    // Scroll to Step 3 smoothly
    sectionStep3.scrollIntoView({ behavior: "smooth" });
  }

  // 7. Copy Draft Button
  btnCopyDraft.addEventListener("click", () => {
    const textToCopy = `SUBJECT: ${draftSubject.value}\n\n${draftBody.value}`;
    navigator.clipboard.writeText(textToCopy).then(() => {
      btnCopyDraft.textContent = "✅ Copied!";
      setTimeout(() => { btnCopyDraft.textContent = "📋 Copy Text"; }, 2000);
    });
  });

  // 8. Back to Step 2
  btnBackToStep2.addEventListener("click", () => {
    document.getElementById("section-step2").scrollIntoView({ behavior: "smooth" });
  });

  // 9. Confirm & Send Report via SMTP
  btnConfirmSend.addEventListener("click", async () => {
    const targetEmail = recipientEmail.value.trim();
    if (!targetEmail) {
      showBanner("Recipient municipal grievance email cannot be empty.", "error");
      return;
    }

    btnConfirmSend.disabled = true;
    btnConfirmSend.innerHTML = `<span>⏳ Dispatching Email via SMTP...</span>`;

    const payload = {
      to_email: targetEmail,
      subject: draftSubject.value,
      body: draftBody.value,
      image_base64: lastAnalysisResult ? lastAnalysisResult.detection.annotated_image_data_url : null,
      cc_citizen_email: inputEmail.value.trim() || null,
      citizen_name: inputName.value.trim() || null,
    };

    try {
      const response = await fetch("/api/send-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const resData = await response.json();

      if (resData.success) {
        modalTitle.textContent = resData.simulated ? "Report Dispatched (Demo Mode)" : "Official Grievance Dispatched!";
        modalIcon.textContent = resData.simulated ? "ℹ️" : "✅";
        modalMessage.textContent = resData.message;
        modalOverlay.classList.remove("hidden");
      } else {
        showBanner(`Dispatch error: ${resData.message}`, "error");
      }

    } catch (err) {
      showBanner(`Could not dispatch report: ${err.message}`, "error");
    } finally {
      btnConfirmSend.disabled = false;
      btnConfirmSend.innerHTML = `✉️ Confirm & Dispatch Municipal Report`;
    }
  });

  // 10. Close Modal and Reset
  btnModalClose.addEventListener("click", () => {
    modalOverlay.classList.add("hidden");
    btnRetake.click();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

});
