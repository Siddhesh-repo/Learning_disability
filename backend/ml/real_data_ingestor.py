"""
Real handwriting dataset ingestion utilities.

Phase 1 utility to ingest:
- Local dataset folder (e.g. ../../dyslexia/{yes,no})
- Optional Kaggle dataset download (e.g. u5awan/*)

The script copies images into backend/data/raw/handwriting and writes a
single manifest CSV in backend/data/processed for downstream training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from urllib.request import urlretrieve
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config import Config


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class RealDataIngestor:
    """Ingest real handwriting data into a normalized local structure."""

    POSITIVE_HINTS = ("yes", "dysgraphia", "risk", "positive")
    NEGATIVE_HINTS = ("no", "normal", "healthy", "control", "negative")
    POSITIVE_LABEL_ALIASES = {
        "yes", "y", "positive", "pos", "highpotential", "highrisk", "atrisk"
    }
    NEGATIVE_LABEL_ALIASES = {
        "no", "n", "negative", "neg", "lowpotential", "lowrisk", "notrisk"
    }

    def __init__(self) -> None:
        Config.init_dirs()
        self.raw_root = Config.DATA_DIR / "raw" / "handwriting"
        self.processed_root = Config.DATA_DIR / "processed"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.processed_root.mkdir(parents=True, exist_ok=True)

    def _iter_images(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

    def _infer_label(self, image_path: Path) -> str:
        def norm(value: str) -> str:
            return "".join(ch for ch in value.lower() if ch.isalnum())

        # Primary rule: inspect path components for explicit class aliases.
        # This supports directories such as "high potential" / "low potential".
        normalized_parts = [norm(part) for part in image_path.parts]
        for part in reversed(normalized_parts):
            if part in self.POSITIVE_LABEL_ALIASES:
                return "yes"
            if part in self.NEGATIVE_LABEL_ALIASES:
                return "no"

        # Fallback for datasets with less structured directory names.
        lowered = "/".join(part.lower() for part in image_path.parts)
        if any(hint in lowered for hint in self.POSITIVE_HINTS):
            return "yes"
        if any(hint in lowered for hint in self.NEGATIVE_HINTS):
            return "no"
        return "unknown"

    def _infer_split(self, image_path: Path) -> str:
        lowered = "/".join(part.lower() for part in image_path.parts)
        for split in ("train", "val", "valid", "validation", "test"):
            if f"/{split}/" in f"/{lowered}/":
                return "val" if split in ("val", "valid", "validation") else split
        return "unspecified"

    def _build_target_name(self, source_image: Path) -> str:
        digest = hashlib.md5(str(source_image).encode("utf-8")).hexdigest()[:10]
        return f"{source_image.stem}_{digest}{source_image.suffix.lower()}"

    def _copy_record(self, source_image: Path, source_name: str, label: str, split: str) -> dict:
        target_dir = self.raw_root / source_name / label
        target_dir.mkdir(parents=True, exist_ok=True)

        target_name = self._build_target_name(source_image)
        target_path = target_dir / target_name
        shutil.copy2(source_image, target_path)

        return {
            "source": source_name,
            "label": label,
            "split": split,
            "original_path": str(source_image),
            "stored_path": str(target_path),
        }

    def ingest_local(self, local_dyslexia_dir: Path) -> list[dict]:
        if not local_dyslexia_dir.exists():
            raise FileNotFoundError(f"Local dataset path not found: {local_dyslexia_dir}")

        records: list[dict] = []
        for image_path in self._iter_images(local_dyslexia_dir):
            label = self._infer_label(image_path)
            records.append(
                self._copy_record(
                    source_image=image_path,
                    source_name="local_dyslexia",
                    label=label,
                    split="unspecified",
                )
            )
        return records

    def download_kaggle_dataset(self, dataset_slug: str) -> Path:
        """Download and unzip a Kaggle dataset using kaggle CLI."""
        slug_safe = dataset_slug.replace("/", "_")
        destination = self.raw_root / "kaggle_downloads" / slug_safe
        destination.mkdir(parents=True, exist_ok=True)

        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(destination),
            "--unzip",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Kaggle download failed. Ensure kaggle CLI is installed and configured.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr: {proc.stderr.strip()}"
            )
        return destination

    def download_zip_dataset(self, zip_url: str, dataset_name: str = "u5awan") -> Path:
        """Download and unzip a dataset archive from a direct URL."""
        destination = self.raw_root / "downloads" / dataset_name
        destination.mkdir(parents=True, exist_ok=True)

        archive_path = destination / "dataset.zip"
        try:
            urlretrieve(zip_url, archive_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to download dataset from URL: {zip_url}\n{exc}") from exc

        # Kaggle download pages may return HTML login content if unauthenticated.
        try:
            header = archive_path.read_bytes()[:512].lower()
        except Exception:
            header = b""
        if b"<html" in header or b"<!doctype html" in header:
            raise RuntimeError(
                "Downloaded content is HTML, not a dataset zip. "
                "This usually means authentication is required for the source URL."
            )

        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(destination)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Downloaded file is not a valid zip archive: {archive_path}") from exc

        return destination

    def ingest_u5awan(self, u5awan_root: Path) -> list[dict]:
        if not u5awan_root.exists():
            raise FileNotFoundError(f"u5awan dataset path not found: {u5awan_root}")

        records: list[dict] = []
        for image_path in self._iter_images(u5awan_root):
            label = self._infer_label(image_path)
            split = self._infer_split(image_path)
            records.append(
                self._copy_record(
                    source_image=image_path,
                    source_name="u5awan",
                    label=label,
                    split=split,
                )
            )
        return records

    def save_manifest(self, records: list[dict], output_path: Path | None = None) -> Path:
        if not records:
            raise ValueError("No records to write. Provide at least one data source.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = output_path or (self.processed_root / f"handwriting_manifest_{timestamp}.csv")

        for idx, record in enumerate(records, start=1):
            record["sample_id"] = f"hw_{idx:06d}"

        fields = ["sample_id", "source", "label", "split", "original_path", "stored_path"]
        with manifest_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

        metadata = {
            "created_at": timestamp,
            "manifest": str(manifest_path),
            "total_samples": len(records),
            "sources": self._count_values(records, "source"),
            "labels": self._count_values(records, "label"),
            "splits": self._count_values(records, "split"),
        }
        metadata_path = self.processed_root / f"handwriting_manifest_{timestamp}_meta.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return manifest_path

    @staticmethod
    def _count_values(records: list[dict], key: str) -> dict:
        counts: dict[str, int] = {}
        for record in records:
            value = record.get(key, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest real handwriting datasets")
    parser.add_argument(
        "--local-dyslexia-dir",
        type=str,
        default=None,
        help="Path to local dataset folder containing yes/no image subfolders",
    )
    parser.add_argument(
        "--u5awan-path",
        type=str,
        default=None,
        help="Path to already downloaded u5awan dataset directory",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default=None,
        help="Optional Kaggle dataset slug to download (for example: owner/dataset)",
    )
    parser.add_argument(
        "--u5awan-zip-url",
        type=str,
        default=None,
        help="Optional direct URL to a zipped u5awan dataset archive",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=None,
        help="Optional explicit output path for the generated manifest CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = RealDataIngestor()
    records: list[dict] = []

    if args.local_dyslexia_dir:
        local_path = Path(args.local_dyslexia_dir).expanduser().resolve()
        local_records = ingestor.ingest_local(local_path)
        records.extend(local_records)
        print(f"Ingested local dyslexia samples: {len(local_records)}")

    if args.kaggle_dataset:
        downloaded_path = ingestor.download_kaggle_dataset(args.kaggle_dataset)
        print(f"Downloaded Kaggle dataset to: {downloaded_path}")
        u5awan_records = ingestor.ingest_u5awan(downloaded_path)
        records.extend(u5awan_records)
        print(f"Ingested Kaggle samples: {len(u5awan_records)}")

    if args.u5awan_zip_url:
        downloaded_path = ingestor.download_zip_dataset(args.u5awan_zip_url, dataset_name="u5awan")
        print(f"Downloaded u5awan zip dataset to: {downloaded_path}")
        u5awan_records = ingestor.ingest_u5awan(downloaded_path)
        records.extend(u5awan_records)
        print(f"Ingested u5awan zip samples: {len(u5awan_records)}")

    if args.u5awan_path:
        u5awan_path = Path(args.u5awan_path).expanduser().resolve()
        u5awan_records = ingestor.ingest_u5awan(u5awan_path)
        records.extend(u5awan_records)
        print(f"Ingested u5awan samples: {len(u5awan_records)}")

    if not args.kaggle_dataset and not args.u5awan_path and not args.u5awan_zip_url:
        print("u5awan dataset was not provided (no --kaggle-dataset, --u5awan-path, or --u5awan-zip-url).")

    output_path = Path(args.output_manifest).expanduser().resolve() if args.output_manifest else None
    manifest_path = ingestor.save_manifest(records=records, output_path=output_path)

    print("-" * 60)
    print(f"Manifest created: {manifest_path}")
    print(f"Total samples: {len(records)}")
    print("-" * 60)


if __name__ == "__main__":
    main()
