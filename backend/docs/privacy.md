# Follica Privacy & Consent Architecture

## 1. Overview
Follica is designed with a privacy-first, minimal-retention architecture. The application processes user-uploaded scalp images and self-reported clinical questionnaire responses exclusively in-memory for real-time triage scoring.

---

## 2. What Is Stored vs. What Is Never Stored

### Stored in SQLite Consent Store (`consent_records` table)
| Field | Type | Purpose |
|---|---|---|
| `id` | `TEXT (UUID v4)` | Random, non-sequential consent record identifier. |
| `created_at` | `TEXT (ISO-8601 UTC)` | Timestamp of consent registration. |
| `terms_version` | `TEXT` | Version string of terms and privacy policy active at time of agreement. |
| `client_token` | `TEXT (SHA-256)` | Salted cryptographic hash of an anonymous user/device identifier. |
| `scope_analysis` | `INTEGER (0/1)` | Explicit opt-in to execute transient scalp health scoring. |
| `scope_progress` | `INTEGER (0/1)` | Optional opt-in for longitudinal tracking / calibration. |
| `retention_days` | `INTEGER` | User-selected data retention window (default: 30 days; max: 365 days). |
| `revoked_at` | `TEXT (ISO-8601 UTC) / NULL` | Revocation timestamp if user exercises right to withdraw consent. |

### What Is NEVER Stored in the Consent Database
- **No Scalp Photos or Trichoscopy Images**: Uploaded images are held strictly in transient memory (RAM) during dual-pipeline processing and discarded immediately after response transmission.
- **No Questionnaire Responses or Symptom Data**: Responses (pull test, shedding pattern, dietary habits, sensations) are processed in-memory and are never written to the consent database.
- **No Derived Health Scores**: Scalp Health Scores, sub-scores, and clinical triage recommendations are returned directly to the client and never logged to the database.
- **No Direct Identifiers**: No raw IP addresses, physical device serials, names, email addresses, or phone numbers are persisted. The `client_token` is salted and hashed using SHA-256 before disk write.

---

## 3. Revocation and Data Deletion

Users have the right to revoke consent at any time.

- **Revocation API Endpoint**: `DELETE /api/v1/consent/{consent_id}` (or `DELETE /consent/{consent_id}`)
- **Soft-Delete Mechanism**: Sets `revoked_at` to the current UTC timestamp immediately.
- **Automated Hard Purge**: The `purge_expired()` maintenance routine executes:
  ```sql
  DELETE FROM consent_records
  WHERE revoked_at IS NOT NULL
     OR (julianday('now') - julianday(created_at)) > retention_days;
  ```
- **Provisional / Uncalibrated Notice**: The default 30-day retention window and automated cleanup rules are provisional operational engineering defaults, not legally certified retention schedules.

---

## 4. Encryption at Rest & Security Claims Notice

- The system persists consent metadata to a local SQLite database (`FOLLICA_CONSENT_DB`).
- Unverified, unimplemented security claims (such as "AES-256 encrypted at rest in cloud buckets") have been explicitly removed from documentation and API contracts to avoid misleading users.
- Production deployments requiring disk encryption should leverage OS-level full-disk encryption (e.g., BitLocker, LUKS) or encrypted storage volumes.

---

## 5. Regulatory Compliance Notice for Deployers

> **IMPORTANT NOTICE FOR PROJECT OWNERS AND DEPLOYERS:**
>
> Follica is provided as an open-source clinical engineering prototype and educational tool.
> **Before exposing this service to real users or patients in any jurisdiction, the project owner must verify and enforce compliance with applicable healthcare and data protection laws:**
>
> - **India**: Digital Personal Data Protection Act, 2023 (DPDP Act) regarding processing of digital personal data, valid consent mechanisms, and notice requirements.
> - **European Union**: General Data Protection Regulation (GDPR) Articles 6, 7, and 9 governing special category health data and explicit consent.
> - **United States**: Health Insurance Portability and Accountability Act (HIPAA) if integrated with covered entities or handling Protected Health Information (PHI).
