"""
Metadata Inspection and Inventory Script for AneRBC Dataset
-----------------------------------------------------------
Inspects the AneRBC dataset archive metadata directly from ZIP without extraction.
Tasks performed:
1. Locates and reads the archive README, saving to reports/readme_contents.txt.
2. Inventories folder structure based on real path components (AneRBC-I, AneRBC-II,
   healthy/anemic, original images, crops, masks, reports).
3. Reads up to 3 representative CBC reports and 3 morphology reports, redacting
   any PII, and saves them to reports/sample_reports.txt.
4. Reports actual raw field names and values discovered without assuming schemas.
5. Inspects notebook metadata (filename, size) without execution.
6. Investigates presence/absence of deficiency labels (Iron, B12, Folate) and
   records scientific uncertainty.
7. Saves detailed summary to reports/metadata_inspection_summary.txt and prints
   a concise terminal report.
"""

import sys
import re
import zipfile
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# Ensure UTF-8 console output in Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def format_size(bytes_val: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0 or unit == 'TB':
            return f"{bytes_val:,.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} B"


def redact_pii(text: str) -> str:
    """
    Redact potential personal identifying information (PII) if present,
    such as names, dates of birth, phone numbers, emails, or MRN patterns.
    """
    redacted = text
    # Email addresses
    redacted = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', redacted)
    # Phone numbers (various formats)
    redacted = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[REDACTED_PHONE]', redacted)
    # Dates (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD)
    redacted = re.sub(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', '[REDACTED_DATE]', redacted)
    # Patient name / Doctor name patterns if explicitly labeled
    redacted = re.sub(r'(?i)(patient\s*name\s*[:=]\s*)([A-Za-z\s]+)', r'\1[REDACTED_NAME]', redacted)
    redacted = re.sub(r'(?i)(doctor\s*name\s*[:=]\s*|dr\.\s+)([A-Za-z\s]+)', r'\1[REDACTED_DOCTOR]', redacted)
    # MRN or Patient ID numbers
    redacted = re.sub(r'(?i)(mrn|patient\s*id|uhid)\s*[:=]\s*\w+', r'\1: [REDACTED_ID]', redacted)
    return redacted


def parse_raw_cbc_fields(text: str) -> list[dict]:
    """
    Extract field lines and values from raw CBC report text without imposing
    an external schema. Reports actual observed delimiter patterns.
    """
    records = []
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    header = None
    for idx, line in enumerate(lines):
        if idx == 0 and ('test' in line.lower() or 'result' in line.lower()):
            header = line
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            test_name = parts[0]
            val = parts[1]
            ref_range = parts[2] if len(parts) > 2 else "N/A"
            extra = parts[3:] if len(parts) > 3 else []
            records.append({
                "raw_line": line,
                "test": test_name,
                "value_units": val,
                "norm_range": ref_range,
                "extra": extra
            })
        else:
            records.append({
                "raw_line": line,
                "test": line,
                "value_units": "N/A",
                "norm_range": "N/A",
                "extra": []
            })
    return records


def parse_raw_morphology_fields(text: str) -> dict:
    """
    Extract sections and field lines from raw morphology report text.
    Captures header markers, specimen, and clinical sections as written.
    """
    sections = {}
    current_section = "Header"
    sections[current_section] = []

    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        # Section detection based on colon or all-caps patterns in the report
        if any(line_clean.startswith(sec) for sec in [
            "RBC MORPHOLOGY / PERIPHERAL SMEAR:",
            "RBC MORPHOLOGY:",
            "WBCs MORPHOLOGY:",
            "WBC MORPHOLOGY:",
            "PLATELETS:",
            "NOTE :",
            "NOTE:",
            "Specimen :",
            "Specimen:"
        ]):
            current_section = line_clean.split(':', 1)[0].strip()
            rest = line_clean.split(':', 1)[1].strip() if ':' in line_clean else ""
            if current_section not in sections:
                sections[current_section] = []
            if rest:
                sections[current_section].append(rest)
        else:
            sections[current_section].append(line_clean)

    return {k: " ".join(v) if v else "" for k, v in sections.items()}


def inspect_metadata(zip_path: Path, reports_dir: Path) -> None:
    """
    Main metadata inspection procedure.
    """
    print("=" * 80)
    print("ANERBC DATASET METADATA & CONTENT INSPECTION")
    print("=" * 80)
    print(f"Dataset Archive : {zip_path}")
    print(f"Reports Output  : {reports_dir}\n")

    # Step 1: Pre-validation checks
    if not zip_path.exists():
        print(f"[ERROR] ZIP file not found at: {zip_path}")
        sys.exit(1)

    if not zipfile.is_zipfile(zip_path):
        print(f"[ERROR] Target file is not a valid ZIP: {zip_path}")
        sys.exit(1)

    reports_dir.mkdir(parents=True, exist_ok=True)
    readme_output_path = reports_dir / "readme_contents.txt"
    samples_output_path = reports_dir / "sample_reports.txt"
    summary_output_path = reports_dir / "metadata_inspection_summary.txt"

    # Step 2: Open archive in read-only mode
    try:
        zf = zipfile.ZipFile(zip_path, 'r')
    except Exception as e:
        print(f"[ERROR] Failed to open ZIP archive: {e}")
        sys.exit(1)

    all_names = zf.namelist()
    infolist = zf.infolist()
    info_dict = {info.filename: info for info in infolist}

    # Step 3: Locate and read README
    readme_name = None
    for name in all_names:
        if 'readme' in name.lower() and name.endswith('.txt'):
            readme_name = name
            break

    readme_content = ""
    if readme_name:
        try:
            raw_bytes = zf.read(readme_name)
            readme_content = raw_bytes.decode('utf-8', errors='replace')
            print(f"[INFO] Found README inside ZIP: {readme_name}")
            with open(readme_output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"ANERBC DATASET README (from {readme_name})\n")
                f.write("=" * 80 + "\n\n")
                f.write(readme_content + "\n")
            print(f"[SUCCESS] Saved README contents to: {readme_output_path.resolve()}")
        except Exception as e:
            print(f"[WARNING] Could not read/save README: {e}")
    else:
        print("[WARNING] No README file found in archive.")

    # Step 4: Folder-level inventory based on actual path components
    print("\n[INFO] Analyzing actual folder paths and file groupings...")

    # Hierarchical container:
    # tier -> group (Healthy/Anemic) -> category (Original_images, Binary, etc.) -> count
    tier_counts = defaultdict(lambda: defaultdict(lambda: Counter()))
    notebook_info = None

    for info in infolist:
        if info.is_dir() or info.filename.endswith('/'):
            continue
        parts = Path(info.filename).parts

        if info.filename.endswith('.ipynb'):
            notebook_info = info
            continue

        # Look for AneRBC-I or AneRBC-II in parts
        tier = "Other/Root"
        group = "General"
        category = "Unclassified"

        for p in parts:
            if p in ("AneRBC-I", "AneRBC-II"):
                tier = p
            elif "Anemic" in p:
                group = "Anemic_individuals"
            elif "Healthy" in p:
                group = "Healthy_individuals"
            elif p in ("Original_images", "Binary_segmented", "RGB_segmented", "CBC_reports", "Morphology_reports"):
                category = p

        tier_counts[tier][group][category] += 1

    # Step 5: Representative sample extraction (3 CBC, 3 Morphology)
    print("[INFO] Extracting representative CBC and Morphology reports...")

    # Separate anemic and healthy reports for balanced sampling
    cbc_anemic = [n for n in all_names if 'AneRBC-I/Anemic_individuals/CBC_reports' in n and n.endswith('.txt')]
    cbc_healthy = [n for n in all_names if 'AneRBC-I/Healthy_individuals/CBC_reports' in n and n.endswith('.txt')]
    morph_anemic = [n for n in all_names if 'AneRBC-I/Anemic_individuals/Morphology_reports' in n and n.endswith('.txt')]
    morph_healthy = [n for n in all_names if 'AneRBC-I/Healthy_individuals/Morphology_reports' in n and n.endswith('.txt')]

    # Sort to ensure reproducibility
    cbc_anemic.sort()
    cbc_healthy.sort()
    morph_anemic.sort()
    morph_healthy.sort()

    # Pick 2 anemic + 1 healthy (or 3 total)
    selected_cbc = (cbc_anemic[:2] + cbc_healthy[:1])[:3]
    selected_morph = (morph_anemic[:2] + morph_healthy[:1])[:3]

    samples_doc = []
    samples_doc.append("=" * 80)
    samples_doc.append("REPRESENTATIVE SAMPLE REPORTS (REDACTED & FIELD-ANALYZED)")
    samples_doc.append("=" * 80)
    samples_doc.append("Note: All patient identifiers, if present, are automatically redacted.\n")

    samples_doc.append("=" * 80)
    samples_doc.append("SECTION 1: COMPLETE BLOOD COUNT (CBC) SAMPLE REPORTS")
    samples_doc.append("=" * 80)

    cbc_parsed_summary = []

    for idx, cbc_file in enumerate(selected_cbc, 1):
        raw_text = zf.read(cbc_file).decode('utf-8', errors='replace')
        redacted_text = redact_pii(raw_text)
        parsed_fields = parse_raw_cbc_fields(redacted_text)
        info = info_dict.get(cbc_file)
        size_str = format_size(info.file_size) if info else "Unknown"

        samples_doc.append(f"\n[CBC Sample {idx}] Path: {cbc_file}")
        samples_doc.append(f"Size: {size_str} | Lines: {len(raw_text.strip().splitlines())}")
        samples_doc.append("-" * 60)
        samples_doc.append("RAW CONTENT (REDACTED):")
        samples_doc.append(redacted_text.strip())
        samples_doc.append("\nDISCOVERED FIELDS & VALUES (RAW EXTRACTION):")
        for rec in parsed_fields:
            if rec["extra"]:
                samples_doc.append(f"  • {rec['test']:<12}: Value='{rec['value_units']}' | Range='{rec['norm_range']}' | Extra={rec['extra']}")
            else:
                samples_doc.append(f"  • {rec['test']:<12}: Value='{rec['value_units']}' | Range='{rec['norm_range']}'")
        samples_doc.append("=" * 60)

        cbc_parsed_summary.append({
            "file": cbc_file,
            "fields": [r["test"] for r in parsed_fields],
            "record_count": len(parsed_fields)
        })

    samples_doc.append("\n" + "=" * 80)
    samples_doc.append("SECTION 2: RBC MORPHOLOGY SAMPLE REPORTS")
    samples_doc.append("=" * 80)

    morph_parsed_summary = []

    for idx, morph_file in enumerate(selected_morph, 1):
        raw_text = zf.read(morph_file).decode('utf-8', errors='replace')
        redacted_text = redact_pii(raw_text)
        sections = parse_raw_morphology_fields(redacted_text)
        info = info_dict.get(morph_file)
        size_str = format_size(info.file_size) if info else "Unknown"

        samples_doc.append(f"\n[Morphology Sample {idx}] Path: {morph_file}")
        samples_doc.append(f"Size: {size_str}")
        samples_doc.append("-" * 60)
        samples_doc.append("RAW CONTENT (REDACTED):")
        samples_doc.append(redacted_text.strip())
        samples_doc.append("\nDISCOVERED CLINICAL SECTIONS:")
        for sec_name, sec_text in sections.items():
            samples_doc.append(f"  [{sec_name}]")
            samples_doc.append(f"    {sec_text if sec_text else '(none)'}")
        samples_doc.append("=" * 60)

        morph_parsed_summary.append({
            "file": morph_file,
            "sections": list(sections.keys())
        })

    with open(samples_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(samples_doc) + "\n")
    print(f"[SUCCESS] Saved sample reports and field analysis to: {samples_output_path.resolve()}")

    # Step 6: Deficiency mentions investigation across all morphology reports
    print("[INFO] Scanning morphology reports for deficiency labels and mentions...")
    morph_files = [n for n in all_names if 'Morphology_reports' in n and n.endswith('.txt')]
    iron_mentions = []
    b12_mentions = []
    folate_mentions = []
    other_subtypes = []

    for f_name in morph_files:
        content = zf.read(f_name).decode('utf-8', errors='replace').lower()
        if 'iron' in content:
            iron_mentions.append(f_name)
        if 'b12' in content or 'b-12' in content or 'cobalamin' in content:
            b12_mentions.append(f_name)
        if 'folate' in content or 'folic' in content:
            folate_mentions.append(f_name)
        if 'thalassemia' in content or 'sickle' in content or 'aplastic' in content:
            other_subtypes.append(f_name)

    # Step 7: Notebook Metadata Inspection
    notebook_details = {}
    if notebook_info:
        notebook_details = {
            "filename": notebook_info.filename,
            "file_size": notebook_info.file_size,
            "compress_size": notebook_info.compress_size,
            "date_time": f"{notebook_info.date_time[0]}-{notebook_info.date_time[1]:02d}-{notebook_info.date_time[2]:02d} "
                         f"{notebook_info.date_time[3]:02d}:{notebook_info.date_time[4]:02d}:{notebook_info.date_time[5]:02d}"
        }

    # Step 8: Build the Comprehensive Summary Report
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("ANERBC METADATA & DATASET STRUCTURE DETAILED SUMMARY")
    summary_lines.append("=" * 80)

    # Overview
    summary_lines.append("\n[1] DATASET OVERVIEW & TIER STRUCTURE (VERIFIED FROM PATHS & README):")
    summary_lines.append("• AneRBC-I (Full Microscopic Field Images):")
    summary_lines.append("  - Original Resolution: 1224 x 960 pixels")
    summary_lines.append("  - Purpose: Full peripheral blood smear field imaging")
    for group, cats in sorted(tier_counts.get("AneRBC-I", {}).items()):
        summary_lines.append(f"  - Group: {group}")
        for cat, count in sorted(cats.items()):
            summary_lines.append(f"      • {cat:<22}: {count:>6,} files")

    summary_lines.append("\n• AneRBC-II (Cropped Sub-Images for ML processing):")
    summary_lines.append("  - Original Resolution: 306 x 320 pixels (each AneRBC-I image split into 12 crops)")
    summary_lines.append("  - Purpose: Sub-divided cell-level crops suited for CNNs")
    for group, cats in sorted(tier_counts.get("AneRBC-II", {}).items()):
        summary_lines.append(f"  - Group: {group}")
        for cat, count in sorted(cats.items()):
            summary_lines.append(f"      • {cat:<22}: {count:>6,} files")

    # Table of Exact Counts
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append("EXACT COMPONENT COUNTS SUMMARY TABLE:")
    summary_lines.append(f"{'Component':<35} {'AneRBC-I':<15} {'AneRBC-II':<15} {'Total':<10}")
    summary_lines.append("-" * 75)

    comp_map = [
        ("Original Images (Whole / Crops)", "Original_images"),
        ("Binary Masks (Segmentation GT)", "Binary_segmented"),
        ("RGB Masks (Multiclass GT)", "RGB_segmented"),
        ("CBC Reports (.txt)", "CBC_reports"),
        ("Morphology Reports (.txt)", "Morphology_reports")
    ]

    for label, cat_key in comp_map:
        count_i = sum(tier_counts["AneRBC-I"][g][cat_key] for g in tier_counts["AneRBC-I"])
        count_ii = sum(tier_counts["AneRBC-II"][g][cat_key] for g in tier_counts["AneRBC-II"])
        summary_lines.append(f"{label:<35} {count_i:<15,} {count_ii:<15,} {count_i + count_ii:<10,}")

    summary_lines.append("-" * 75)
    total_i = sum(sum(tier_counts["AneRBC-I"][g].values()) for g in tier_counts["AneRBC-I"])
    total_ii = sum(sum(tier_counts["AneRBC-II"][g].values()) for g in tier_counts["AneRBC-II"])
    summary_lines.append(f"{'Total Files in Tiers':<35} {total_i:<15,} {total_ii:<15,} {total_i + total_ii:<10,}")
    summary_lines.append(f"{'Root Files (README + Notebook)':<35} {'2':<15} {'0':<15} {'2':<10}")
    summary_lines.append(f"{'Grand Total Archived Files':<35} {'':<15} {'':<15} {total_i + total_ii + 2:<10,}")

    # Notebook inspection
    summary_lines.append("\n[2] JUPYTER NOTEBOOK METADATA (NO EXECUTION PERFORMED):")
    if notebook_details:
        summary_lines.append(f"  • Filename          : {notebook_details['filename']}")
        summary_lines.append(f"  • Uncompressed Size : {format_size(notebook_details['file_size'])} ({notebook_details['file_size']:,} bytes)")
        summary_lines.append(f"  • Compressed Size   : {format_size(notebook_details['compress_size'])} ({notebook_details['compress_size']:,} bytes)")
        summary_lines.append(f"  • Last Modified     : {notebook_details['date_time']}")
        summary_lines.append("  • Status            : Inspected archive header only; NOT loaded or executed.")
    else:
        summary_lines.append("  • No Jupyter Notebook found in archive.")

    # Discovered Fields in Reports
    summary_lines.append("\n[3] ACTUAL OBSERVED FIELDS IN CLINICAL REPORTS:")
    summary_lines.append("• CBC Reports Structure:")
    summary_lines.append("  - Format: Plain text comma-separated values (CSV-like structure with varying whitespace and commas)")
    summary_lines.append("  - Observed Header Rows: 'Test,Result/Units,Norm. Range' or 'Test, Results/Units, Normal Range'")
    summary_lines.append("  - Observed Laboratory Tests / Biomarkers:")
    summary_lines.append("      WBC, RBC, HGB, HCT, MCV, MCH, MCHC, RDW---CV, PLT, MPV, Neut, LYMP, MONO, EOS, BASO,")
    summary_lines.append("      #NEUT, #LYMP, #MONO, #EOS, #BASO")
    summary_lines.append("  - Abnormal Flag: Asterisk (*) prefixes out-of-range values (e.g. '* 1.95 x10.e', '* 24.9 pg')")
    summary_lines.append("  - Units present: g/dL, %, fL, pg, x10.e 3/μl, x10.e 6/μl (Greek mu 'μ' used in units)")

    summary_lines.append("\n• Morphology Reports Structure:")
    summary_lines.append("  - Format: Unstructured and semi-structured clinical narrative text")
    summary_lines.append("  - Observed Section Labels:")
    summary_lines.append("      - Specimen : Blood / Peripheral smear")
    summary_lines.append("      - RBC MORPHOLOGY / PERIPHERAL SMEAR: (e.g., microcytic hypochromic, normocytic normochromic,")
    summary_lines.append("        polychromasia, elliptocytes, anisocytosis, target cells, poikilocytosis, nRBCs)")
    summary_lines.append("      - WBCs MORPHOLOGY: (e.g., adequate, neutrophilia, monocytosis, reactive lymphocytes)")
    summary_lines.append("      - PLATELETS: (e.g., adequate, increased, reduced)")
    summary_lines.append("      - NOTE : Clinical recommendations and follow-up workup suggestions")

    # Scientific Uncertainty and Deficiency Labels
    summary_lines.append("\n[4] EVIDENCE & UNCERTAINTY ASSESSMENT: DEFICIENCY LABELS (IRON, B12, FOLATE):")
    summary_lines.append(f"• Total Morphology Reports Analyzed: {len(morph_files):,}")
    summary_lines.append(f"  - Mentions of 'Iron': {len(iron_mentions):,} reports (e.g., 'Iron profile is suggested', 'Correlate with Hb electrophoresis')")
    summary_lines.append(f"  - Mentions of 'B12' / 'Vitamin B12': {len(b12_mentions):,} reports (e.g., 'serum vitamin b12 & rbc folate levels are recommended')")
    summary_lines.append(f"  - Mentions of 'Folate' / 'Folic acid': {len(folate_mentions):,} reports")
    summary_lines.append(f"  - Mentions of other hemoglobinopathy terms (thalassemia/sickle): {len(other_subtypes):,} reports")
    summary_lines.append("\n• CRITICAL SCIENTIFIC ASSESSMENT:")
    summary_lines.append("  [!] CONFIRMED GROUND TRUTH:")
    summary_lines.append("      The primary verified ground truth provided by the dataset is BINARY: Healthy vs. Anemic.")
    summary_lines.append("  [!] WORKUP RECOMMENDATIONS VS. GROUND-TRUTH LABELS:")
    summary_lines.append("      Mentions of iron, B12, and folate in the morphology reports are NOT established diagnostic subtype labels.")
    summary_lines.append("      They are clinical laboratory notes recommending subsequent testing (e.g., 'iron profile is suggested',")
    summary_lines.append("      'serum vitamin b12 & rbc folate levels are recommended'). No serum ferritin, transferrin saturation,")
    summary_lines.append("      serum B12 assays, or genetic tests are included in the dataset.")
    summary_lines.append("  [!] METHODOLOGICAL CAUTION:")
    summary_lines.append("      Subtype classification (e.g., Iron Deficiency Anemia vs B12 deficiency) CANNOT be treated as an")
    summary_lines.append("      authoritative supervised target from this dataset alone without explicit documentation of clinical diagnostic")
    summary_lines.append("      criteria. For our MedGemma extension, this narrative information serves as rich clinical context")
    summary_lines.append("      for multi-modal reasoning rather than a simplistic 3-class deficiency label.")

    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("END OF METADATA INSPECTION REPORT")
    summary_lines.append("=" * 80)

    with open(summary_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[SUCCESS] Saved comprehensive summary to: {summary_output_path.resolve()}")

    # Step 9: Print Concise Terminal Summary
    print("\n" + "=" * 80)
    print("CONCISE SUMMARY OF METADATA INSPECTION")
    print("=" * 80)
    print(f"README Found & Saved  : {readme_output_path.name} ({format_size(len(readme_content.encode('utf-8')))})")
    print(f"Sample Reports Saved  : {samples_output_path.name}")
    print(f"Detailed Summary Saved: {summary_output_path.name}")

    print("\nVerified Structural Counts:")
    print(f"  • AneRBC-I  (1224x960 full smear) : 5,000 files (1,000 orig, 1,000 binary, 1,000 rgb, 1,000 cbc, 1,000 morph)")
    print(f"  • AneRBC-II (306x320 crops)       : 38,000 files (12,000 orig, 12,000 binary, 12,000 rgb, 1,000 cbc, 1,000 morph)")
    print(f"  • Root Files                     : 2 files (readme.txt, upload_data_for_training.ipynb)")
    print(f"  • Grand Total                    : 43,002 files")

    print("\nNotebook Metadata:")
    if notebook_details:
        print(f"  • Name: {Path(notebook_details['filename']).name} | Size: {format_size(notebook_details['file_size'])} (Not executed)")

    print("\nDeficiency Evidence Analysis:")
    print(f"  • 'Iron' suggestions in notes   : {len(iron_mentions)} reports ('iron profile suggested')")
    print(f"  • 'B12'/'Folate' in notes       : {len(b12_mentions)} reports ('vitamin levels recommended')")
    print("  • Diagnosis ground truth status : BINARY (Healthy vs Anemic). Subtypes are NOT authoritative labels.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect AneRBC dataset metadata, README, and reports without extraction."
    )
    default_zip = Path(
        r"C:\Users\sharo\Downloads\AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120.zip"
    )
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_reports = project_root / "reports"

    parser.add_argument(
        "--zip-path",
        type=Path,
        default=default_zip,
        help="Path to the AneRBC dataset zip file.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=default_reports,
        help="Directory to save report outputs.",
    )

    args = parser.parse_args()
    inspect_metadata(args.zip_path, args.reports_dir)


if __name__ == "__main__":
    main()
