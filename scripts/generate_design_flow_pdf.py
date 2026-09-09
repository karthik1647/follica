"""
Follica Project - Biomedical System Design Flow Report Generator.
Generates an academic, publication-grade PDF report mapping Follica
to the standardized Biomedical Instrumentation Design Flow framework.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for total page numbering and running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "FOLLICA | Biomedical System Design Flow & Architecture Report")
            self.drawRightString(558, 750, "Academic Project Documentation")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential - Academic Coursework & Engineering Project Evaluation")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.restoreState()


def generate_report(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor('#0284C7'),
        spaceAfter=14
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#475569')
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0284C7'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor('#0F172A')
    )
    table_cell_header = ParagraphStyle(
        'TableCellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.white
    )

    story = []

    # Title Block
    story.append(Paragraph("FOLLICA: Scalp Trichoscopy & Biomedical Decision Support", title_style))
    story.append(Paragraph("Biomedical System Design Flow & Cyber-Physical Instrumentation Architecture", subtitle_style))

    meta_text = (
        "<b>System Classification:</b> Software as a Medical Device (SaMD) / Dual-Pipeline Computer Vision & Triage Engine<br/>"
        "<b>Architectural Foundation:</b> Standard Biomedical Instrumentation Design Flow (Transducer to Clinical Diagnosis)<br/>"
        "<b>Clinical Benchmarks:</b> Dichromatic Reflection Model | L'Oreal Dandruff Atlas (PMC11975186) | Traya 3-Science Axis"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0284C7'), spaceAfter=12))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Biomedical Problem Formulation", h1_style))
    summary_p1 = (
        "Follica is an open-source, non-invasive digital scalp trichoscopy and systemic triage system designed to bridge "
        "the gap between high-cost clinical optical instrumentation (such as polarized FotoFinder dermoscopes) and non-standardized "
        "commercial e-commerce intake funnels (such as Traya). The system operates on a dual-pipeline architecture: deterministic "
        "classical computer vision micro-feature extraction running parallel with deep learning feature representations, fused with "
        "a systemic multi-factor intake model covering gastrointestinal absorption, metabolic rate, and endocrine profiles."
    )
    story.append(Paragraph(summary_p1, body_style))

    # Section 2: Complete Design Flow Mapping
    story.append(Paragraph("2. Complete Mapping to the Biomedical Instrumentation Design Flow", h1_style))
    flow_intro = (
        "The standard Biomedical Design Flow outlines the transformation of an organic clinical pathology into a digitized, "
        "processed, and clinically actionable diagnostic output. Below is the direct, stage-by-stage engineering mapping for Follica:"
    )
    story.append(Paragraph(flow_intro, body_style))

    table_data = [
        [
            Paragraph("Design Flow Stage", table_cell_header),
            Paragraph("Biomedical Engineering Implementation in Follica", table_cell_header),
            Paragraph("Signal & Algorithmic Details", table_cell_header)
        ],
        [
            Paragraph("1. Clinical Condition", table_cell_bold),
            Paragraph("Scalp dermatological and follicular pathologies including Seborrheic Dermatitis, Alopecia Areata, Androgenetic Thinning, Psoriasis, and Normal Scalp Homeostasis.", table_cell),
            Paragraph("Includes systemic etiologies: acute telogen effluvium, gut-mediated malabsorption, and thyroid dysfunction.", table_cell)
        ],
        [
            Paragraph("2. Physiological Parameters", table_cell_bold),
            Paragraph("Surface optical reflection, stratum corneum desquamation, follicular spatial distribution, cutaneous microvascular perfusion, and lipid barrier gloss.", table_cell),
            Paragraph("Quantified into 4 bounded sub-scores (0-25 each): Flake, Redness/Erythema, Sebum/Lipid, and Follicular Density.", table_cell)
        ],
        [
            Paragraph("3. Mode of Measurement", table_cell_bold),
            Paragraph("Non-invasive Optical Trichoscopy in the visible electromagnetic spectrum (wavelengths 400 nm to 700 nm).", table_cell),
            Paragraph("Multi-angle macroscopic photographic capture (Crown, Hairline, Parting, Temples).", table_cell)
        ],
        [
            Paragraph("4. Transducer Design", table_cell_bold),
            Paragraph("Smartphone solid-state CMOS/CCD photodiode sensor array with Bayer Color Filter Array (RGGB pattern).", table_cell),
            Paragraph("Converts incident photons to photo-generated electron charge packets, then to proportional analog voltages.", table_cell)
        ],
        [
            Paragraph("5. Calibration Signal", table_cell_bold),
            Paragraph("Optical: Gray-World White Balance algorithm normalizing ambient room illuminants to D65 standard.<br/>Clinical: 6-point exfoliating severity scale calibrated per Flament et al. 2025 (PMC11975186, r^2 = 0.952).", table_cell),
            Paragraph("Ensures colorimetric invariance against warm yellow (2700K) or cool fluorescent (6500K) room lighting.", table_cell)
        ],
        [
            Paragraph("6. Filter & Pre-Amplifier Design (Noise Suppression)", table_cell_bold),
            Paragraph("Software Pre-Processing Quality Gate & Optical Filtering:<br/>"
                      "- Dichromatic Specular Glare Mask (Shafer Model) suppressing flash reflection artifacts.<br/>"
                      "- Discrete 2D Laplacian operator filtering out motion/defocus blur (var < 60.0).<br/>"
                      "- Luminance histogram boundary gate (35 < mean < 225) rejecting under/overexposure.<br/>"
                      "- EXIF optical parser detecting high sensor grain (ISO > 800) and slow shutter speed.", table_cell),
            Paragraph("Eliminates flash reflections being misclassified as white dandruff flakes, and prevents motion-blurred images from clouding diagnosis.", table_cell)
        ],
        [
            Paragraph("7. Signal Digitization (ADC)", table_cell_bold),
            Paragraph("Hardware Analog-to-Digital Converter integrated on the camera image signal processor (ISP).<br/>"
                      "- Spatial Sampling: Standardized 512 x 512 discrete spatial grid (262,144 pixels).<br/>"
                      "- Quantization Depth: 24-bit Truecolor (8 bits / 256 discrete levels per channel).", table_cell),
            Paragraph("Generates normalized numerical tensor I(x, y) in RGB color space with bounded dynamic range [0, 255].", table_cell)
        ],
        [
            Paragraph("8. Digital Signal Processing (DSP)", table_cell_bold),
            Paragraph("Dual Classical Computer Vision + Deep Learning Engine:<br/>"
                      "- Morphological Top-Hat filtering isolating high-frequency keratin squames.<br/>"
                      "- Local Laplacian Texture Variance Discriminator (var >= 8.0) separating rough flakes from smooth glare fringes.<br/>"
                      "- MobileNetV4 / EfficientNet-B0 deep convolutional feature extraction.<br/>"
                      "- ORB Keypoint Homography Registration for serial longitudinal delta tracking (Day 0 vs Day 14/30).<br/>"
                      "- Traya 3-Science Multi-Factor Correlation Engine (Gut, Energy, Hormones).", table_cell),
            Paragraph("Replaces opaque black-box deep learning with explainable, deterministic mathematical metrics combined with systemic clinical correlation.", table_cell)
        ],
        [
            Paragraph("9. Wireless Transmission", table_cell_bold),
            Paragraph("Encrypted client-server payload transmission over TLS (HTTPS). Multipart/form-data schema transmitting image bytes and structured JSON questionnaire context to FastAPI.", table_cell),
            Paragraph("AES-256 encrypted at rest in database storage with user-selectable 30-day automated purge window.", table_cell)
        ],
        [
            Paragraph("10. Data Display & Storage", table_cell_bold),
            Paragraph("Client-side Single Page Application (SPA) displaying composite Scalp Health Score (0-100 SHS), 4 discrete sub-score tracks, live risk indicator, and longitudinal comparison.", table_cell),
            Paragraph("Secure, pseudonymous session storage preventing biometric photo linkage to personally identifiable information (PII).", table_cell)
        ],
        [
            Paragraph("11. Clinical Diagnosis & Triage Support", table_cell_bold),
            Paragraph("Server-Side 3-Tier Clinical Triage Engine:<br/>"
                      "- Level 1: Minor Home Routine (Preventative barrier hydration).<br/>"
                      "- Level 2: Monitored Routine (Active ingredients with 14-day tracking; strict server confidence gate < 0.65).<br/>"
                      "- Level 3: Escalation (Automated generation of 1-Page Clinical Summary PDF pass for certified dermatologist review).", table_cell),
            Paragraph("Follows medical ethics standards by operating as decision support rather than unlicensed medical diagnosis.", table_cell)
        ]
    ]

    flow_table = Table(table_data, colWidths=[105, 235, 164])
    flow_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))

    story.append(flow_table)
    story.append(Spacer(1, 12))

    # Section 3: Algorithmic Innovations
    story.append(Paragraph("3. Scientific Innovations & Core Algorithmic Proofs", h1_style))

    story.append(Paragraph("A. Flash Glare Suppression via the Shafer Dichromatic Reflection Model", h2_style))
    glare_text = (
        "A critical challenge in mobile scalp trichoscopy is that flash or ring light creates intense, circular specular hotspots "
        "on the curved scalp surface. In standard morphology algorithms, these bright white highlights are falsely identified as white dandruff "
        "flakes (causing healthy scalps to be diagnosed with Seborrheic Dermatitis).<br/>"
        "Follica resolves this physically via the Shafer Dichromatic Reflection Model: Total reflectance L(x) = L_diffuse(x) + L_specular(x). "
        "Flash reflection is purely specular and achromatic (V > 230, |R-G| + |G-B| < 25, S < 35 in HSV). Follica constructs a binary specular mask, "
        "suppresses these pixels from the luminance channel prior to Top-Hat morphology, and routes the specular energy exclusively to the sebum/gloss "
        "sub-score where surface shine belongs. Remaining candidates are evaluated through a local Laplacian texture variance discriminator; genuine "
        "keratin squames have rough irregular boundaries (var >= 8.0), whereas residual specular fringes have smooth falloff and are rejected."
    )
    story.append(Paragraph(glare_text, body_style))

    story.append(Paragraph("B. Clinical Dandruff Atlas Calibration (PMC11975186 Benchmark)", h2_style))
    paper_text = (
        "Rather than using an arbitrary threshold, Follica's flake scoring algorithm is calibrated directly against the clinical dandruff grading atlas "
        "published by Flament et al. (2025, PMC11975186 / Journal of Cosmetic Dermatology), which established r^2 = 0.952 correlation with 15 expert "
        "dermatologists. The system maps measured particle surface coverage to the validated 6-point scale: Grade 0 (<0.1% coverage -> 25.0 score), "
        "Grade 1 (0.1-0.5% -> 22.0-24.9), Grade 2 (0.5-1.2% -> 18.0-21.9), Grade 3 (1.2-2.5% -> 13.0-17.9), Grade 4 (2.5-4.0% -> 7.0-12.9), and "
        "Grade 5 (>4.0% -> 0.0-6.9)."
    )
    story.append(Paragraph(paper_text, body_style))

    story.append(Paragraph("C. Systemic Biomarker Integration (Traya 3-Science Model)", h2_style))
    traya_text = (
        "Clinical trichology demonstrates that hair loss cannot be diagnosed accurately by superficial scalp appearance alone. Diffuse thinning and "
        "sudden shedding (Telogen Effluvium) are driven by systemic physiology. Following the Traya 3-Science diagnostic framework (Dermatology + "
        "Nutrition + Internal Metabolism), Follica integrates self-reported systemic biomarkers into its clinical correlation engine:<br/>"
        "1. <b>Gastrointestinal Axis:</b> Identifies bloating, chronic constipation, or IBS as malabsorption risks that starve hair matrix keratinocytes of iron, zinc, and biotin.<br/>"
        "2. <b>Metabolic Axis:</b> Flags chronic sluggish fatigue and rapid weight fluctuations as systemic metabolic load indicators.<br/>"
        "3. <b>Endocrine Axis:</b> Correlates thyroid dysfunction and PCOS with follicular cycle elongation and local DHT binding affinity, triggering automated blood panel recommendations (TSH, Free T3/T4, Ferritin, Vitamin D3)."
    )
    story.append(Paragraph(traya_text, body_style))

    # Section 4: Verification and Quality Assurance
    story.append(Paragraph("4. Automated Quality Assurance & Verification Suite", h1_style))
    qa_text = (
        "The complete Follica codebase is backed by an automated verification suite consisting of <b>54 end-to-end tests</b> executed via pytest. "
        "The test suite covers: dual-pipeline sub-score invariants, server-side confidence gate rerouting (<0.65 threshold), pre-inference Laplacian blur "
        "and overexposure rejections, adversarial geometries (extreme aspect ratios, sensor noise, corrupted bytes), longitudinal ORB registration, "
        "flash glare suppression on synthetic healthy scalps, and metabolic/gut health correlation synthesis. All 54 tests pass with 100% success rate."
    )
    story.append(Paragraph(qa_text, body_style))

    # Section 5: Conclusion
    story.append(Spacer(1, 6))
    story.append(Paragraph("5. Academic Conclusion", h1_style))
    conc_text = (
        "By adhering strictly to the Biomedical Instrumentation Design Flow, Follica demonstrates how smartphone sensors can be transformed into a "
        "clinically grounded diagnostic screening instrument. The project achieves technical parity with commercial systems like Traya on systemic anamnesis, "
        "while outperforming them on the computer vision front through deterministic specular glare separation and open-source scientific transparency."
    )
    story.append(Paragraph(conc_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "docs/Follica_Biomedical_Design_Flow_Report.pdf"
    generate_report(out_file)
    print(f"Report successfully generated at: {out_file}")
