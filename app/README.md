# 🛣️ Civic Road Damage Grievance Portal

A citizen-to-municipal grievance dispatch application that transforms smartphone pavement photos into formal, geolocated road repair complaints with AI-annotated photographic evidence.

> **Architecture Guarantee:** This web application is 100% decoupled in `app/`. It imports our trained competition model weights (`weights/best_yolo11s.pt` or `weights/best.pt`) strictly in CPU mode, and has zero impact on the offline Kaggle competition inference pipeline (`inference.py`).

---

## 🌟 Key Features

1. **Smartphone Camera & Photo Capture:**
   - Native mobile camera trigger (`capture="environment"`) or desktop photo upload.
   - Instant client-side preview with retake functionality.

2. **One-Tap High-Precision GPS Geolocation:**
   - Captures high-accuracy device coordinates via HTML5 `navigator.geolocation`.
   - Generates clickable Google Maps pin links (`https://www.google.com/maps?q=lat,lon`).
   - Reverse-geocodes coordinates via OpenStreetMap Nominatim with local city fallback.

3. **AI Structural Distress Detection (Strictly CPU-Safe):**
   - Automatically loads `weights/best_yolo11s.pt` (or falls back to baseline `weights/best.pt`).
   - Detects and color-codes all 4 competition distress classes:
     - 🔴 **D40 (Pothole):** High hazard for 2-wheelers and vehicle axles
     - 🟠 **D20 (Alligator Crack):** Sub-base structural fatigue
     - 🔵 **D00 (Longitudinal Crack):** Centerline joint distress
     - 🟢 **D10 (Transverse Crack):** Thermal contraction distress
   - Computes an Urgency Priority Grade:
     - **HIGH:** If any Pothole (D40) or $\ge 3$ distress zones
     - **MEDIUM:** If Alligator Crack (D20)
     - **LOW:** Surface cracks (D00, D10)

4. **Automated Municipal Grievance Routing (`app/municipalities.json`):**
   - Automatically routes complaints to official municipal corporations across major Indian metropolitan hubs:
     - **Mysuru:** Mysuru City Corporation (MCC)
     - **Bengaluru:** Bruhat Bengaluru Mahanagara Palike (BBMP)
     - **Delhi:** Municipal Corporation of Delhi (MCD)
     - **Mumbai:** Brihanmumbai Municipal Corporation (BMC)
     - **Guwahati:** Guwahati Municipal Corporation (GMC)
     - **Pune:** Pune Municipal Corporation (PMC)
     - **Hyderabad:** Greater Hyderabad Municipal Corporation (GHMC)
     - **Chennai:** Greater Chennai Corporation (GCC)
     - **Kolkata:** Kolkata Municipal Corporation (KMC)
     - **National / State Fallback:** General Public Works Grievance Cell
   - All email destinations remain citizen-editable before dispatch.

5. **Preview-Before-Send & Formal Complaint Generation:**
   - Citizen reviews the annotated image, priority rating, and drafted administrative complaint letter.
   - Includes full complainant contact info and coordinates.

6. **Automated SMTP Dispatch with Attached Evidence:**
   - Dispatches the report and attaches the annotated JPEG directly to the municipal grievance inbox via Gmail SMTP (`smtp.gmail.com:587` TLS).
   - CCs the complainant for their tracking records.
   - Includes graceful **Simulation Mode** if SMTP credentials are not configured.

---

## 🚀 Quickstart Guide

### 1. Install Requirements
The application uses FastAPI and standard imaging libraries already present in the workspace:
```bash
pip install fastapi uvicorn python-multipart python-dotenv requests pillow torch ultralytics
```

### 2. Configure Gmail SMTP (Optional for Live Dispatch)
If you wish to send real emails to municipal grievance cells:
1. Go to your Google Account security settings: [https://myaccount.google.com/security](https://myaccount.google.com/security)
2. Ensure **2-Step Verification** is enabled.
3. Visit [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).
4. Create a new App Password (name it e.g. `Road Damage Reporter`).
5. Copy the generated 16-character password (format: `xxxx xxxx xxxx xxxx`).
6. Copy `app/.env.example` to `.env` in the repository root:
   ```env
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_EMAIL=your_email@gmail.com
   SMTP_APP_PASSWORD=your_16_character_app_password
   ```

*(If `.env` is omitted or empty, the app will run in safe Demo / Simulation Mode and print the full drafted email to the screen).*

### 3. Launch the Web Server
Run Uvicorn from the repository root:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
```
http://127.0.0.1:8000
```
On a mobile device on the same local Wi-Fi, open `http://<your-lan-ip>:8000`.

---

## 📁 File Structure

```
app/
├── __init__.py
├── main.py               # FastAPI endpoints (/api/analyze, /api/send-report, /api/geocode)
├── detector.py           # CPU YOLO model loader, bounding box renderer, and priority calculator
├── router.py             # OpenStreetMap Nominatim reverse geocoder & municipal mapper
├── mailer.py             # Gmail SMTP dispatcher (TLS) with attachment support
├── municipalities.json   # Seeded municipal contact database
├── .env.example          # Template for Gmail credentials
├── README.md             # Application guide
└── static/
    ├── index.html        # Mobile-first responsive UI
    ├── css/style.css     # Clean CSS with design tokens
    └── js/app.js         # Geolocation, camera trigger, API client
```
