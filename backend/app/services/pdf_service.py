"""
Follica Generic Clinical Summary PDF Service.

Architectural Design Intent & Rationale:
1. Universal Non-Affiliated Clinical Hand-Off:
   - Rationale: A key failure in student healthcare is information loss between the patient
     and the clinic. When a student visits a general clinic or dermatologist, they often forget
     how long symptoms lasted, what products they tried, or whether hard water was a factor.
   - This service produces a standardized 1-page summary ("For your treating physician / dermatologist")
     containing timestamped objective CV sub-scores, reported context, and suspected patterns.
   - It intentionally omits any specific clinic or hospital names to ensure geographic portability
     and avoid institutional endorsement assumptions.
"""

import io
from app.models.schemas import AnalysisResponse, SymptomContext


class ClinicalPDFService:
    """Creates a printable medical summary for clinical hand-off."""

    @staticmethod
    def generate_report(analysis: AnalysisResponse, context: SymptomContext, patient_ref_id: str = "ANON-USER") -> bytes:
        """
        Generates PDF bytes using reportlab if available, or structured formatted report bytes.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor=colors.HexColor('#1E293B'),
                spaceAfter=6
            )
            subtitle_style = ParagraphStyle(
                'DocSub',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#64748B'),
                spaceAfter=12
            )
            heading_style = ParagraphStyle(
                'Heading2',
                parent=styles['Heading2'],
                fontSize=12,
                textColor=colors.HexColor('#0F172A'),
                spaceBefore=8,
                spaceAfter=4
            )
            body_style = ParagraphStyle(
                'Body',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#334155'),
                leading=13
            )
            disclaimer_style = ParagraphStyle(
                'Disclaimer',
                parent=styles['Normal'],
                fontSize=7.5,
                textColor=colors.HexColor('#94A3B8'),
                leading=10
            )

            story = []

            # Header
            story.append(Paragraph("Follica Scalp Assessment Summary", title_style))
            story.append(Paragraph(f"<b>Recipient:</b> For your treating physician / dermatologist | <b>Ref ID:</b> {patient_ref_id} | <b>Date:</b> {analysis.analysis_timestamp[:10]}", subtitle_style))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=10))

            # Triage & Condition Box
            story.append(Paragraph("Assessment & Triage Overview", heading_style))
            triage_data = [
                ["Suggested Pattern:", analysis.suggested_condition],
                ["Confidence Metric:", f"{analysis.confidence_score * 100:.1f}% {'(Confidence-Gated)' if analysis.confidence_gated else ''}"],
                ["Triage Category:", analysis.triage_level],
                ["Composite Scalp Health Score:", f"{analysis.scalp_health_score:.1f} / 100"]
            ]
            triage_table = Table(triage_data, colWidths=[160, 360])
            triage_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1E293B')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1'))
            ]))
            story.append(triage_table)
            story.append(Spacer(1, 10))

            # Classical CV Sub-Scores Breakdown
            story.append(Paragraph("Computer Vision Micro-Feature Sub-Scores", heading_style))
            sub_data = [
                ["Flake & Scaling Score (0-25)", f"{analysis.sub_scores.flake_score:.1f}", "High-contrast particle count & area ratio"],
                ["Erythema / Redness Score (0-25)", f"{analysis.sub_scores.redness_score:.1f}", "Skin chrominance delta from baseline"],
                ["Sebum / Lipid Score (0-25)", f"{analysis.sub_scores.sebum_score:.1f}", "Specular highlight and luster distribution"],
                ["Hair Coverage (indicative) (0-25)", f"{analysis.sub_scores.density_score:.1f}", "Estimated visible fiber area ratio"]
            ]
            sub_table = Table([["Metric", "Score", "Extraction Method"]] + sub_data, colWidths=[180, 70, 270])
            sub_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284C7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1'))
            ]))
            story.append(sub_table)
            story.append(Spacer(1, 10))

            # Self-Reported Context
            story.append(Paragraph("Patient-Reported Context", heading_style))
            context_text = (
                f"<b>Symptom Duration:</b> {context.duration_months} Months ({context.duration_weeks} Weeks) | "
                f"<b>Itching:</b> {'Yes' if context.itching else 'No'} | "
                f"<b>Burning/Pain:</b> {'Yes' if context.burning_or_pain else 'No'} | "
                f"<b>Hard Water:</b> {'Yes' if context.hard_water_exposure else 'No'} | "
                f"<b>Stress:</b> {context.stress_level.capitalize()}<br/>"
                f"<b>Gut Health:</b> {context.gut_health.replace('_', ' ').title()} | "
                f"<b>Metabolic Energy:</b> {context.metabolic_energy.replace('_', ' ').title()} | "
                f"<b>Endocrine/Hormonal:</b> {context.hormonal_profile.replace('_', ' ').title()}<br/>"
                f"<b>Additional Notes / Past Products:</b> {context.symptom_notes or 'None specified'}"
            )
            story.append(Paragraph(context_text, body_style))
            story.append(Spacer(1, 10))

            # Clinical Escalation Note
            story.append(Paragraph("Clinical Hand-Off Notes", heading_style))
            story.append(Paragraph(f"<i>{analysis.triage_reason}</i>", body_style))
            if analysis.consultation_referral.clinical_hand_off_notes:
                story.append(Paragraph(f"<b>Key Observations:</b> {analysis.consultation_referral.clinical_hand_off_notes}", body_style))
            story.append(Spacer(1, 14))

            # Mandatory Universal Disclaimer
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceAfter=6))
            story.append(Paragraph(f"<b>Notice:</b> {analysis.disclaimer}", disclaimer_style))

            doc.build(story)
            return buffer.getvalue()

        except ImportError:
            report_str = f"""
================================================================================
                      FOLLICA SCALP ASSESSMENT REPORT
================================================================================
Addressed To: {analysis.consultation_referral.addressed_to}
Reference ID: {patient_ref_id}
Timestamp   : {analysis.analysis_timestamp}

ASSESSMENT SUMMARY:
- Suggested Pattern: {analysis.suggested_condition}
- Confidence Score: {analysis.confidence_score * 100:.1f}% (Gated: {analysis.confidence_gated})
- Triage Category: {analysis.triage_level}
- Composite Scalp Health Score: {analysis.scalp_health_score:.1f} / 100

CV SUB-SCORES (0-25):
- Flake & Scaling Score    : {analysis.sub_scores.flake_score:.1f}
- Erythema & Redness Score : {analysis.sub_scores.redness_score:.1f}
- Sebum Balance Score      : {analysis.sub_scores.sebum_score:.1f}
- Hair Coverage (indicative): {analysis.sub_scores.density_score:.1f}

PATIENT CONTEXT:
- Duration: {context.duration_weeks} weeks
- Itching: {context.itching}, Burning/Pain: {context.burning_or_pain}
- Hard Water Exposure: {context.hard_water_exposure}

CLINICAL HAND-OFF NOTES:
{analysis.triage_reason}
{analysis.consultation_referral.clinical_hand_off_notes}

LEGAL & MEDICAL DISCLAIMER:
{analysis.disclaimer}
================================================================================
"""
            return report_str.encode('utf-8')


pdf_service = ClinicalPDFService()
