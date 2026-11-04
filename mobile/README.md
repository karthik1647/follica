# Follica Mobile Client (React Native / Expo)

Cross-platform mobile application for the Follica Scalp Health Assessment and Clinical Triage System.

---

## 1. Features
- **Standardized Viewfinder Guidelines**: Clean lens reminders, ambient light guidance (no flash washouts), and focus distance targets.
- **4-Step Clinical Intake**: Captures concern area, borewell hard water, daily motorcycle helmet friction, hair pull test, and digestive/metabolic markers.
- **Explainable Health Scores**: Displays 0-100 Scalp Health Score (SHS) alongside 4 discrete Computer Vision sub-scores (Flake, Redness, Sebum, Hair Coverage).
- **Molecule-First Prescriptions**: Displays targeted chemical active schedules (e.g. Ketoconazole 2% wash, Salicylic Acid descalers) and lifestyle advice.

---

## 2. Running Locally for Free via Expo Go

You do not need a paid developer account to test on physical Android or iOS devices.

### Prerequisites
1. Install Node.js (version 18 or newer).
2. Install the free **Expo Go** application on your physical smartphone from Google Play Store or Apple App Store.

### Quickstart
```bash
# 1. Navigate to the mobile directory
cd mobile

# 2. Install dependencies
npm install

# 3. Start Expo development server
npx expo start
```

### Connecting to Your Phone
1. Ensure your smartphone and development machine are connected to the same Wi-Fi network (or use a Cloudflare tunnel URL).
2. Scan the QR code displayed in your terminal using:
   - **Android**: The Expo Go app's built-in QR scanner.
   - **iOS**: The native iOS Camera application (tap the Expo prompt).
3. In the mobile interface, set the **Backend Server Endpoint** to your local machine IP (e.g., `http://192.168.1.50:8000/api/v1`) or your public tunnel URL (`https://your-tunnel.trycloudflare.com/api/v1`).
