"""
Archive Inspection Script for AneRBC Dataset
--------------------------------------------
Inspects the AneRBC dataset ZIP archive without extracting any files.
Gathers archive metadata, file counts, extensions, directory structure,
potential metadata files, and candidate data categories (images, masks, reports).
Saves findings to reports/archive_inventory.txt and prints a summary.
"""

import sys
import os
import zipfile
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def format_size(bytes_val: int) -> str:
    """Format bytes into a human-readable string (B, KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0 or unit == 'TB':
            return f"{bytes_val:,.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} B"


def inspect_archive(zip_path: Path, output_file: Path) -> None:
    """
    Inspect the ZIP archive without extracting contents.
    Analyzes metadata, structure, file types, and potential research assets.
    """
    print("=" * 80)
    print("ANERBC DATASET ARCHIVE INSPECTION")
    print("=" * 80)
    print(f"Target Archive : {zip_path}")
    print(f"Report Output  : {output_file}\n")

    # Step 1: Pre-validation checks
    if not zip_path.exists():
        print(f"[ERROR] Archive file not found at: {zip_path}")
        print("Please verify that the file exists and the path is correct.")
        sys.exit(1)

    if not zip_path.is_file():
        print(f"[ERROR] Specified path is a directory or special file, not a regular file: {zip_path}")
        sys.exit(1)

    try:
        # Check read permissions
        with open(zip_path, 'rb') as f:
            f.read(10)
    except PermissionError:
        print(f"[ERROR] Permission denied when trying to read archive: {zip_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to open archive for reading: {e}")
        sys.exit(1)

    if not zipfile.is_zipfile(zip_path):
        print(f"[ERROR] The target file is not a valid ZIP archive: {zip_path}")
        sys.exit(1)

    print("[INFO] Archive verified as a valid readable ZIP file. Analyzing metadata...")

    # Step 2: Read archive metadata
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            infolist = zf.infolist()
    except zipfile.BadZipFile as e:
        print(f"[ERROR] Corrupt or invalid ZIP archive: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error while opening ZIP archive: {e}")
        sys.exit(1)

    total_entries = len(infolist)
    file_entries = [info for info in infolist if not info.is_dir() and not info.filename.endswith('/')]
    dir_entries = [info for info in infolist if info.is_dir() or info.filename.endswith('/')]

    total_files = len(file_entries)
    total_dirs = len(dir_entries)

    total_uncompressed = sum(info.file_size for info in file_entries)
    total_compressed = sum(info.compress_size for info in file_entries)
    compression_ratio = (
        (1.0 - (total_compressed / total_uncompressed)) * 100
        if total_uncompressed > 0
        else 0.0
    )

    # Step 3: Extensions and Directory Breakdown
    extension_counter = Counter()
    top_level_folders = Counter()
    directory_file_counts = Counter()
    all_files_by_ext = defaultdict(list)

    # Useful metadata extensions
    METADATA_EXTS = {'.csv', '.xlsx', '.xls', '.tsv', '.txt', '.json', '.pdf', '.doc', '.docx', '.xml', '.yaml', '.yml'}
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    metadata_files = []
    image_files = []

    # Heuristic classifications (non-presumptive: reports matches found)
    possible_categories = {
        "original_images": [],
        "binary_masks": [],
        "rgb_masks": [],
        "cbc_reports": [],
        "morphology_reports": [],
        "doctor_decisions": [],
    }

    for info in file_entries:
        filename = info.filename
        p = Path(filename)
        ext = p.suffix.lower() if p.suffix else "[no_extension]"
        extension_counter[ext] += 1
        all_files_by_ext[ext].append(filename)

        # Directory accounting
        parts = p.parts
        if len(parts) > 1:
            top_level_folders[parts[0]] += 1
            parent_dir = str(p.parent).replace('\\', '/')
            directory_file_counts[parent_dir] += 1
        else:
            top_level_folders["[archive_root]"] += 1
            directory_file_counts["[archive_root]"] += 1

        # Check metadata
        if ext in METADATA_EXTS:
            metadata_files.append((filename, info.file_size, info.compress_size))

        # Check image files
        if ext in IMAGE_EXTS:
            image_files.append((filename, info.file_size))

        # Classify candidates based on path/name patterns
        lower_path = filename.lower()

        # Binary or RGB Masks
        if any(term in lower_path for term in ['mask', 'seg', 'binary']):
            if any(term in lower_path for term in ['rgb', 'color', 'colour']):
                possible_categories["rgb_masks"].append(filename)
            else:
                possible_categories["binary_masks"].append(filename)
        elif ext in IMAGE_EXTS:
            possible_categories["original_images"].append(filename)

        # CBC Reports
        if any(term in lower_path for term in ['cbc', 'complete_blood', 'blood_count', 'hemogram', 'haemogram']):
            possible_categories["cbc_reports"].append(filename)

        # Morphology Reports
        if any(term in lower_path for term in ['morph', 'smear', 'blood_film', 'film', 'ps_report', 'peripheral']):
            possible_categories["morphology_reports"].append(filename)

        # Doctor decisions / ground truth / diagnosis
        if any(term in lower_path for term in ['doctor', 'decision', 'diagnosis', 'expert', 'ground_truth', 'gt_', 'label', 'annota']):
            possible_categories["doctor_decisions"].append(filename)

    # Step 4: Prepare the comprehensive report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ANERBC DATASET ARCHIVE INVENTORY & INSPECTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Archive Path          : {zip_path}")
    report_lines.append(f"Total Archive Entries : {total_entries:,}")
    report_lines.append(f"Total Files           : {total_files:,}")
    report_lines.append(f"Total Folders (dirs)  : {total_dirs:,}")
    report_lines.append(f"Total Uncompressed    : {format_size(total_uncompressed)} ({total_uncompressed:,} bytes)")
    report_lines.append(f"Total Compressed      : {format_size(total_compressed)} ({total_compressed:,} bytes)")
    report_lines.append(f"Space Savings / Ratio : {compression_ratio:.2f}%")
    report_lines.append("-" * 80)

    # Extensions Breakdown
    report_lines.append("\n[1] FILE COUNTS BY EXTENSION:")
    report_lines.append(f"{'Extension':<20} {'Count':<10} {'Percentage':<10}")
    report_lines.append("-" * 40)
    for ext, count in extension_counter.most_common():
        pct = (count / total_files * 100) if total_files > 0 else 0
        report_lines.append(f"{ext:<20} {count:<10,} {pct:.2f}%")

    # Top-Level Folders
    report_lines.append("\n[2] TOP-LEVEL DIRECTORIES / ROOT STRUCTURE:")
    report_lines.append(f"{'Directory Name':<50} {'File Count':<10}")
    report_lines.append("-" * 62)
    for folder, count in top_level_folders.most_common():
        report_lines.append(f"{folder:<50} {count:<10,}")

    # Distinct Directory Paths (summary)
    report_lines.append(f"\nTotal Distinct Subdirectories with files: {len(directory_file_counts)}")
    report_lines.append(f"{'Directory Path':<65} {'Files':<10}")
    report_lines.append("-" * 77)
    for d_path, count in sorted(directory_file_counts.items()):
        report_lines.append(f"{d_path:<65} {count:<10,}")

    # Metadata Files Detected
    report_lines.append("\n[3] METADATA & DOCUMENTATION FILES:")
    report_lines.append(f"Total metadata/document files detected: {len(metadata_files)}")
    if metadata_files:
        report_lines.append(f"{'Filename':<70} {'Uncompressed':<15} {'Compressed':<15}")
        report_lines.append("-" * 100)
        for m_file, uncomp, comp in sorted(metadata_files):
            report_lines.append(f"{m_file:<70} {format_size(uncomp):<15} {format_size(comp):<15}")
    else:
        report_lines.append("  None found matching common metadata extensions (.csv, .xlsx, .txt, .json, .pdf, etc.).")

    # Candidate Categories Breakdown
    report_lines.append("\n[4] RESEARCH ASSET CANDIDATE DETECTION (HEURISTIC SEARCH):")
    for category, items in possible_categories.items():
        cat_name = category.replace('_', ' ').title()
        report_lines.append(f"\n  • Candidate {cat_name} (Total: {len(items):,}):")
        if items:
            sample_count = min(5, len(items))
            for item in items[:sample_count]:
                report_lines.append(f"      - {item}")
            if len(items) > sample_count:
                report_lines.append(f"      ... and {len(items) - sample_count:,} more")
        else:
            report_lines.append("      (No filenames matched this heuristic pattern)")

    # Representative Image Files
    report_lines.append("\n[5] REPRESENTATIVE IMAGE SAMPLES:")
    report_lines.append(f"Total Image Files: {len(image_files):,}")
    if image_files:
        for img_file, img_size in image_files[:10]:
            report_lines.append(f"  - {img_file} ({format_size(img_size)})")
        if len(image_files) > 10:
            report_lines.append(f"  ... and {len(image_files) - 10:,} more image files")
    else:
        report_lines.append("  No image files found in archive.")

    # Samples for Each Discovered Extension
    report_lines.append("\n[6] SAMPLE FILES FOR EACH DETECTED EXTENSION:")
    for ext, files in sorted(all_files_by_ext.items()):
        report_lines.append(f"\n  Extension: {ext} (Total: {len(files):,})")
        for f in files[:3]:
            report_lines.append(f"    - {f}")
        if len(files) > 3:
            report_lines.append(f"    ... {len(files) - 3} more")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF INVENTORY REPORT")
    report_lines.append("=" * 80)

    # Step 5: Save report to file
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(report_lines) + "\n")
        print(f"[SUCCESS] Detailed inventory saved to: {output_file.resolve()}\n")
    except Exception as e:
        print(f"[ERROR] Failed to write report to {output_file}: {e}")

    # Step 6: Print Concise Terminal Summary
    print("-" * 80)
    print("CONCISE SUMMARY OF ARCHIVE CONTENTS")
    print("-" * 80)
    print(f"Total Files       : {total_files:,}")
    print(f"Total Directories : {total_dirs:,}")
    print(f"Uncompressed Size : {format_size(total_uncompressed)}")
    print(f"Compressed Size   : {format_size(total_compressed)} ({compression_ratio:.1f}% space savings)")
    print("\nTop Extensions:")
    for ext, count in extension_counter.most_common(6):
        pct = (count / total_files * 100) if total_files > 0 else 0
        print(f"  • {ext:<15} : {count:>8,} ({pct:>5.1f}%)")

    print("\nTop-Level Folders:")
    for folder, count in top_level_folders.most_common():
        print(f"  • {folder:<45} : {count:>8,} files")

    print(f"\nMetadata Files Found : {len(metadata_files):,}")
    for m_file, uncomp, _ in metadata_files[:5]:
        print(f"  • {m_file} ({format_size(uncomp)})")
    if len(metadata_files) > 5:
        print(f"  ... and {len(metadata_files) - 5} additional metadata files")

    print("\nCandidate Research Assets Detected:")
    for category, items in possible_categories.items():
        cat_name = category.replace('_', ' ').title()
        print(f"  • {cat_name:<25}: {len(items):>8,} matching items")

    print("\n" + "=" * 80)
    print(f"Full inventory report is ready at: {output_file.resolve()}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect AneRBC dataset ZIP archive without extraction."
    )
    default_zip = Path(
        r"C:\Users\sharo\Downloads\AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120.zip"
    )
    # Default output relative to script's parent (project root)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_output = project_root / "reports" / "archive_inventory.txt"

    parser.add_argument(
        "--zip-path",
        type=Path,
        default=default_zip,
        help="Path to the AneRBC dataset zip file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to save the output inventory text report.",
    )

    args = parser.parse_args()
    inspect_archive(args.zip_path, args.output)


if __name__ == "__main__":
    main()
