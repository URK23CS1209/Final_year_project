
import argparse
import csv
import hashlib
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ZIP = Path(
    r"C:\Users\sharo\Downloads\AneRBC dataset a benchmark dataset "
    r"for computer-aided anemia diagnosis using RBC images. "
    r"httpsdoi.org10.1093databasebaae120.zip"
)

DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "dataset_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "image_hash_audit.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "image_hash_audit.txt"


def normalize_path(path):
    """Normalize ZIP paths and manifest paths for exact matching."""
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def sha256_zip_member(zip_file, member_name):
    """Hash a ZIP member incrementally without extracting it."""
    digest = hashlib.sha256()
    size = 0

    with zip_file.open(member_name, "r") as image_file:
        while True:
            chunk = image_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)

    return digest.hexdigest(), size


def run_audit(zip_path, manifest_path, output_path, summary_path):
    for path in (zip_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "image_path",
            "original_image_id",
            "dataset_tier",
            "directory_label",
        }
        missing_columns = required - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing_columns)}"
            )

        manifest_rows = list(reader)

    records = []
    hash_to_paths = defaultdict(list)
    archive_index = {}

    print(f"Manifest records: {len(manifest_rows):,}")
    print(f"ZIP archive: {zip_path}")
    print("Indexing ZIP entries...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if not info.is_dir():
                normalized = normalize_path(info.filename)
                archive_index.setdefault(normalized, info.filename)

        print(f"ZIP file entries: {len(archive_index):,}")
        print("Hashing manifest images. Please wait...")

        for index, row in enumerate(manifest_rows, start=1):
            manifest_image_path = normalize_path(row["image_path"])
            archive_name = archive_index.get(manifest_image_path)

            record = {
                "dataset_tier": row["dataset_tier"],
                "original_image_id": row["original_image_id"],
                "directory_label": row["directory_label"],
                "image_path": row["image_path"],
                "sha256": "",
                "size_bytes": "",
                "duplicate_group": "",
                "status": "",
            }

            if archive_name is None:
                record["status"] = "MISSING_IN_ARCHIVE"
                records.append(record)
                continue

            try:
                digest, size = sha256_zip_member(archive, archive_name)
                record["sha256"] = digest
                record["size_bytes"] = size
                record["status"] = "OK"
                hash_to_paths[digest].append(record)
            except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
                record["status"] = f"READ_ERROR: {type(exc).__name__}"

            records.append(record)

            if index % 500 == 0 or index == len(manifest_rows):
                print(f"Processed {index:,}/{len(manifest_rows):,} manifest rows")

    duplicate_groups = 0
    duplicate_extra_files = 0

    for digest, matching_records in hash_to_paths.items():
        if len(matching_records) > 1:
            duplicate_groups += 1
            duplicate_extra_files += len(matching_records) - 1
            group_name = f"SHA256-{digest[:12]}"

            for record in matching_records:
                record["duplicate_group"] = group_name

    columns = [
        "dataset_tier",
        "original_image_id",
        "directory_label",
        "image_path",
        "sha256",
        "size_bytes",
        "duplicate_group",
        "status",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    successful = sum(r["status"] == "OK" for r in records)
    missing = sum(r["status"] == "MISSING_IN_ARCHIVE" for r in records)
    errors = sum(r["status"].startswith("READ_ERROR") for r in records)
    unique_hashes = len(hash_to_paths)

    summary = [
        "ANERBC IMAGE SHA-256 AUDIT",
        "=" * 60,
        f"Archive: {zip_path}",
        f"Manifest: {manifest_path}",
        f"Manifest rows: {len(manifest_rows)}",
        f"Successfully hashed: {successful}",
        f"Missing from archive: {missing}",
        f"Read errors: {errors}",
        f"Unique SHA-256 hashes: {unique_hashes}",
        f"Duplicate hash groups: {duplicate_groups}",
        f"Extra records in duplicate groups: {duplicate_extra_files}",
        "",
        "A duplicate hash indicates byte-identical image content.",
        "It does not establish whether different images depict the same smear.",
        "Review duplicate groups before making any dataset changes.",
        "",
        f"Detailed CSV: {output_path}",
    ]

    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n" + "\n".join(summary))

    if missing or errors:
        print(
            "\nWARNING: Investigate missing entries or read errors "
            "before using this dataset."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Audit manifest image integrity without ZIP extraction."
    )
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    try:
        run_audit(
            args.zip_path,
            args.manifest,
            args.output,
            args.summary,
        )
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        csv.Error,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
