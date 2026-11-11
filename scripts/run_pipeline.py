#!/usr/bin/env python3
"""
Provenance ESG OCR worker (Chandra OCR 2 on vLLM), HPC edition

Batch OCR worker tuned for CINECA Leonardo Booster (4x A100 per node):
- Endpoints are static (from the SLURM-generated config, one vLLM server per GPU).
- Concurrency and timeouts tuned for dedicated A100 nodes (no spot preemption).
- Core-hour accounting instead of dollar cost.
- Stall detection enabled (30 min) so stuck jobs exit and SLURM retry can run.

Features: fixed batch size 4, JSON progress, 0-indexing, _full.json output only.

Usage:
    python scripts/run_pipeline.py config.json

See README.md for the config format and hpc/leonardo/ for the SLURM job.
"""

import os
import sys
import time
import json
import logging
import requests
import multiprocessing
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Set, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED

SCRIPT_DIR = Path(__file__).parent.resolve()
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

# Allow config path override via environment variable or command line argument
# Default to chandra_config.json, but can be overridden:
# - Environment: CHANDRA_CONFIG=/path/to/config.json
# - Command line: python chandra_main_v2.py /path/to/config.json
def get_config_path() -> Path:
    """Get config path from env var, command line arg, or default."""
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        return Path(sys.argv[1])
    env_path = os.environ.get("CHANDRA_CONFIG")
    if env_path:
        return Path(env_path)
    return SCRIPT_DIR / "chandra_config.json"

CONFIG_PATH = get_config_path()

# Fixed batch size - 4 pages per batch (chandra-2 4B model tolerates higher throughput)
BATCH_SIZE = 4

# vLLM settings
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "chandra"

# Processing settings
MAX_OUTPUT_TOKENS = 16000
IMAGE_DPI = 200
MIN_IMAGE_DIM = 2048
MAX_RETRIES_PER_BATCH = 3
# chandra-2 is a 4B model; A100s have plenty of headroom to run 12 concurrent
# requests per GPU against vLLM --max-num-seqs=64 without hitting queue stalls.
WORKERS_PER_GPU = 12

# Endpoint failure threshold (failures within 3 minutes to mark as dead)
ENDPOINT_FAILURE_THRESHOLD = 50

# Batch timeout (seconds) - batches exceeding this are marked as failed
BATCH_TIMEOUT_SECONDS = 1000

# Inference timeout (seconds) - hard kill inference if it hangs longer than this
# This actually terminates the stuck worker, freeing the GPU for other batches
INFERENCE_TIMEOUT_SECONDS = 1000

# Fresh restart interval (seconds) - script performs a fresh restart after this time
# Set to 0 to disable. On Leonardo endpoints are static; SLURM retry loop handles restarts.
FRESH_RESTART_INTERVAL = 0

# Stall detection (seconds) - exit if no PDF completes within this time
# Only activates after the first PDF completes. On Leonardo, stall likely means vLLM crashed.
# Env override (CHANDRA_STALL_TIMEOUT_SECONDS) lets a specific job loosen this WITHOUT
# changing the default for other pipelines — the huge-doc backlog tail (300-885pg docs)
# needs >1800s to complete its first document; the reparse slurm exports 3600 for that.
STALL_TIMEOUT_SECONDS = int(os.environ.get("CHANDRA_STALL_TIMEOUT_SECONDS", "1800"))  # 30 min default

# Fix #5 trigger: force-release endpoint_current_pdf assignments when the
# scheduler desyncs (pending work + zero futures + no recent submissions).
# 90s is long enough to not trip during normal inter-PDF transitions, but
# well under the 10-min steady-state watchdog threshold.
DESYNC_RELEASE_SECONDS = 90

# Render workers - process pool for PDF page rendering (CPU-bound)
# Leonardo node has 32 cores; 8 workers keep 4 GPUs well fed without overwhelming memory
RENDER_WORKERS = 8
RENDER_TIMEOUT_SECONDS = 120  # Max time for PDF page rendering

# Leonardo: 1 GPU-hour = 8 core-hours (each GPU = 1/4 node, node = 32 cores)
CORE_HOURS_PER_GPU_HOUR = 8

# Optional future tweaks (measure before changing):
# - BATCH_SIZE 3 or 4: fewer HTTP round-trips, but higher memory and timeout risk.
# - WORKERS_PER_GPU 5: try if GPU utilization is low; may increase queueing.
# - Skip per-batch health check in worker_http_inference: saves one GET per batch
#   on stable HPC; keep if you see frequent endpoint flakiness.
# - STATUS_INTERVAL 120: less log volume on very long runs.


# ============================================================================
# LOGGING SETUP
# ============================================================================

# Base logs directory: logs/ next to the working directory
# Operational logs go to logs/{target_dir_name}/ (set at runtime)
LOGS_BASE_DIR = Path(os.environ.get("OCR_LOGS_DIR", str(Path.cwd() / "logs")))

# Module-level target logs dir, set once in run_processing_cycle()
# Used by get_timeout_log_path, get_quarantine_path, get_shortlist_path
_target_logs_dir: Optional[Path] = None


def get_target_logs_dir() -> Optional[Path]:
    """Get the target-specific logs directory (logs/{target_dir_name}/)."""
    return _target_logs_dir


def set_target_logs_dir(pdf_dir_name: str) -> Path:
    """Set and create the target-specific logs directory. Returns the path."""
    global _target_logs_dir
    _target_logs_dir = LOGS_BASE_DIR / pdf_dir_name
    _target_logs_dir.mkdir(parents=True, exist_ok=True)
    return _target_logs_dir


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging with console handler. File handler added later via add_file_logging()."""
    logger = logging.getLogger("chandra_v2")
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs from root logger
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def add_file_logging(logs_dir: Path) -> None:
    """Add a file handler to the logger once we know the target dir."""
    logger = logging.getLogger("chandra_v2")
    # Avoid adding duplicate file handlers on fresh restarts
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    try:
        log_file = logs_dir / "chandra-workhorse.log"
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass


log = setup_logging()


# ============================================================================
# BLOCK-TYPE COUNTER (chandra-2 sanity tally across a run)
# ============================================================================

from threading import Lock as _BlockLock
_block_type_counts: Dict[str, int] = {}
_block_type_counts_lock = _BlockLock()


def _tally_block_types(chunks) -> None:
    """Tally block_type/label values seen in chunks, thread-safe."""
    if not chunks:
        return
    local: Dict[str, int] = {}
    for c in chunks:
        if not isinstance(c, dict):
            continue
        key = c.get("block_type") or c.get("label") or "unknown"
        if not isinstance(key, str):
            key = str(key)
        local[key] = local.get(key, 0) + 1
    if local:
        with _block_type_counts_lock:
            for k, v in local.items():
                _block_type_counts[k] = _block_type_counts.get(k, 0) + v


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BatchProgress:
    """Progress tracking for a single PDF."""
    pdf_name: str
    total_pages: int
    batch_size: int = BATCH_SIZE
    completed_batches: List[int] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, complete, error
    error: Optional[str] = None
    
    @property
    def total_batches(self) -> int:
        return (self.total_pages + self.batch_size - 1) // self.batch_size
    
    @property
    def is_complete(self) -> bool:
        return len(self.completed_batches) >= self.total_batches
    
    def get_pending_batches(self) -> List[int]:
        """Get list of batch numbers that still need processing."""
        all_batches = set(range(self.total_batches))
        completed = set(self.completed_batches)
        return sorted(all_batches - completed)


@dataclass
class BatchResult:
    """Result from processing a single batch."""
    batch_num: int
    page_start: int  # 0-indexed, inclusive
    page_end: int    # 0-indexed, exclusive
    pages: List[Dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None and len(self.pages) > 0


@dataclass
class EndpointInfo:
    """Endpoint tracking info."""
    name: str
    pod_id: str
    url: str
    failure_timestamps: List[float] = field(default_factory=list)  # Track failure times
    is_dead: bool = False


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def atomic_json_write(path: Path, data: Any, max_retries: int = 3) -> None:
    """Write JSON atomically using temp file + rename with retry logic."""
    import uuid
    for attempt in range(max_retries):
        # Use unique temp file to avoid race conditions
        temp_path = path.with_suffix(f'.tmp.{uuid.uuid4().hex[:8]}')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            temp_path.rename(path)
            return
        except (FileNotFoundError, OSError) as e:
            # Clean up temp file if it exists
            try:
                temp_path.unlink(missing_ok=True)
            except:
                pass
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Brief backoff
            else:
                log.warning(f"Failed to write {path} after {max_retries} attempts: {e}")


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None if not exists or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# ============================================================================
# TIMEOUT TRACKING
# ============================================================================

def get_timeout_log_path(output_dir: Path) -> Path:
    """Get path to timeout log file (in target logs dir if set, else output_dir)."""
    logs_dir = get_target_logs_dir()
    return (logs_dir or output_dir) / "_timeout_log.json"


def normalize_timeout_log_keys(timeout_log: Dict[str, int]) -> Tuple[Dict[str, int], int]:
    """
    Normalize timeout log keys to handle both formats:
    - 2-part: pdf_stem:batch_num (batch mode)
    - 3-part: pdf_stem:batch_num:sub_label (single-page mode)
    
    This ensures both scripts can read/write the same timeout log.
    Does NOT modify the log - just returns a normalized view for checking.
    
    Returns: (normalized_log, count_of_3part_keys_found)
    """
    normalized = {}
    three_part_count = 0
    
    for key, count in timeout_log.items():
        parts = key.split(':')
        if len(parts) == 3:
            # 3-part key from single-page script: pdf_stem:batch_num:sub_label
            # Normalize to 2-part for batch-level checking
            pdf_stem, batch_num, sub_label = parts
            key_2part = f"{pdf_stem}:{batch_num}"
            # Take max count if both a and b exist
            normalized[key_2part] = max(normalized.get(key_2part, 0), count)
            three_part_count += 1
        elif len(parts) == 2:
            # Already 2-part format
            normalized[key] = max(normalized.get(key, 0), count)
        else:
            # Unknown format, keep as-is
            normalized[key] = count
    
    return normalized, three_part_count


def load_timeout_log(output_dir: Path) -> Dict[str, int]:
    """
    Load timeout log from file.
    Returns dict: {"pdf_stem:batch_num": timeout_count} or {"pdf_stem:batch_num:sub_label": timeout_count}
    
    Compatible with both 2-part (batch) and 3-part (single-page) key formats.
    """
    log_path = get_timeout_log_path(output_dir)
    data = load_json(log_path)
    return data if data else {}


def save_timeout_log(output_dir: Path, timeout_log: Dict[str, int]) -> None:
    """Save timeout log to file."""
    log_path = get_timeout_log_path(output_dir)
    atomic_json_write(log_path, timeout_log)


def record_batch_timeout(output_dir: Path, timeout_log: Dict[str, int], pdf_stem: str, batch_num: int) -> int:
    """
    Record a batch timeout. Returns the new timeout count for this batch.
    
    - First timeout: count = 1 (permanently flagged, skip in future runs)
    """
    key = f"{pdf_stem}:{batch_num}"
    current_count = timeout_log.get(key, 0)
    new_count = current_count + 1
    timeout_log[key] = new_count
    save_timeout_log(output_dir, timeout_log)
    return new_count


def is_batch_permanently_flagged(timeout_log: Dict[str, int], pdf_stem: str, batch_num: int) -> bool:
    """Check if a batch has been flagged as permanently timed out (1+ timeouts).
    
    Checks both formats:
    - 2-part: pdf_stem:batch_num (from batch script)
    - 3-part: pdf_stem:batch_num:a or pdf_stem:batch_num:b (from single-page script)
    
    A batch is flagged if ANY of its pages/sub-labels have timed out.
    """
    # Check 2-part key (batch mode)
    key_2part = f"{pdf_stem}:{batch_num}"
    if timeout_log.get(key_2part, 0) >= 1:
        return True
    
    # Check 3-part keys (single-page mode) - either 'a' or 'b' sub-label
    key_3part_a = f"{pdf_stem}:{batch_num}:a"
    key_3part_b = f"{pdf_stem}:{batch_num}:b"
    if timeout_log.get(key_3part_a, 0) >= 1 or timeout_log.get(key_3part_b, 0) >= 1:
        return True
    
    return False


def check_and_quarantine_pdf(output_dir: Path, timeout_log: Dict[str, int], quarantine_log: Dict[str, Any], pdf_stem: str) -> bool:
    """
    Check if a PDF should be quarantined based on timeout count.
    Quarantines the PDF if it exceeds threshold.
    Returns True if the PDF was newly quarantined.
    """
    if is_file_quarantined(quarantine_log, pdf_stem):
        return False  # Already quarantined
    
    timeout_count = count_pdf_timeouts(timeout_log, pdf_stem)
    if timeout_count >= QUARANTINE_THRESHOLD:
        quarantine_file(output_dir, quarantine_log, pdf_stem, timeout_count)
        return True
    return False


# ============================================================================
# QUARANTINE SYSTEM - PDFs with too many timeouts
# ============================================================================

QUARANTINE_THRESHOLD = 10  # PDFs with 10+ timed-out batches are quarantined


def get_quarantine_path(output_dir: Path) -> Path:
    """Get path to quarantine log file (in target logs dir if set, else output_dir)."""
    logs_dir = get_target_logs_dir()
    return (logs_dir or output_dir) / "_quarantined_files.json"


def load_quarantine_log(output_dir: Path) -> Dict[str, Any]:
    """
    Load quarantine log from file.
    Returns dict: {"files": {"pdf_stem": {"timeout_count": N, "quarantined_at": timestamp}}}
    """
    log_path = get_quarantine_path(output_dir)
    data = load_json(log_path)
    return data if data else {"files": {}}


def save_quarantine_log(output_dir: Path, quarantine_log: Dict[str, Any]) -> None:
    """Save quarantine log to file."""
    log_path = get_quarantine_path(output_dir)
    atomic_json_write(log_path, quarantine_log)


def is_file_quarantined(quarantine_log: Dict[str, Any], pdf_stem: str) -> bool:
    """Check if a PDF file is quarantined."""
    return pdf_stem in quarantine_log.get("files", {})


def quarantine_file(output_dir: Path, quarantine_log: Dict[str, Any], pdf_stem: str, timeout_count: int) -> None:
    """Add a PDF to the quarantine list."""
    if "files" not in quarantine_log:
        quarantine_log["files"] = {}
    
    quarantine_log["files"][pdf_stem] = {
        "timeout_count": timeout_count,
        "quarantined_at": datetime.now().isoformat()
    }
    save_quarantine_log(output_dir, quarantine_log)
    log.warning(f"   🚫 QUARANTINED: {pdf_stem} ({timeout_count} timed-out batches)")


def count_pdf_timeouts(timeout_log: Dict[str, int], pdf_stem: str) -> int:
    """Count the number of timed-out batches for a specific PDF.
    
    Handles both formats and counts unique batches (not double-counting a and b sub-labels).
    """
    # Use a set to track unique batches (pdf:batch_num format)
    timed_out_batches = set()
    prefix = f"{pdf_stem}:"
    
    for key in timeout_log:
        if key.startswith(prefix):
            parts = key.split(':')
            if len(parts) == 2:
                # 2-part: pdf_stem:batch_num
                timed_out_batches.add(key)
            elif len(parts) == 3:
                # 3-part: pdf_stem:batch_num:sub_label -> normalize to 2-part
                batch_key = f"{parts[0]}:{parts[1]}"
                timed_out_batches.add(batch_key)
    
    return len(timed_out_batches)


def update_quarantine_from_timeout_log(output_dir: Path, timeout_log: Dict[str, int], quarantine_log: Dict[str, Any]) -> int:
    """
    Scan timeout log and quarantine any PDFs that exceed the threshold.
    Returns the number of newly quarantined files.
    
    Handles both 2-part (batch) and 3-part (single-page) key formats,
    counting unique batches per PDF.
    """
    # Group timeouts by PDF, counting unique batches
    pdf_timed_out_batches: Dict[str, set] = {}
    for key in timeout_log:
        parts = key.split(':')
        if len(parts) == 2:
            # 2-part: pdf_stem:batch_num
            pdf_stem = parts[0]
            batch_key = key
        elif len(parts) == 3:
            # 3-part: pdf_stem:batch_num:sub_label
            pdf_stem = parts[0]
            batch_key = f"{parts[0]}:{parts[1]}"  # Normalize to 2-part
        else:
            continue
        
        if pdf_stem not in pdf_timed_out_batches:
            pdf_timed_out_batches[pdf_stem] = set()
        pdf_timed_out_batches[pdf_stem].add(batch_key)
    
    # Convert to counts
    pdf_timeout_counts = {pdf: len(batches) for pdf, batches in pdf_timed_out_batches.items()}
    
    # Check for PDFs exceeding threshold
    newly_quarantined = 0
    for pdf_stem, count in pdf_timeout_counts.items():
        if count >= QUARANTINE_THRESHOLD and not is_file_quarantined(quarantine_log, pdf_stem):
            quarantine_file(output_dir, quarantine_log, pdf_stem, count)
            newly_quarantined += 1
    
    return newly_quarantined


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def get_progress_path(output_dir: Path, pdf_stem: str) -> Path:
    """Get path to progress JSON file."""
    return output_dir / pdf_stem / "_progress.json"


def load_progress(output_dir: Path, pdf_stem: str, total_pages: int) -> BatchProgress:
    """Load or create progress tracking for a PDF."""
    progress_path = get_progress_path(output_dir, pdf_stem)
    data = load_json(progress_path)
    
    if data:
        # Validate batch_size matches
        if data.get("batch_size") == BATCH_SIZE and data.get("total_pages") == total_pages:
            # Verify that completed batch files actually exist
            completed_batches = data.get("completed_batches", [])
            verified_batches = []
            missing_count = 0
            
            for batch_num in completed_batches:
                batch_path = get_batch_path(output_dir, pdf_stem, batch_num)
                if batch_path.exists():
                    verified_batches.append(batch_num)
                else:
                    missing_count += 1
            
            if missing_count > 0:
                log.warning(f"   ⚠️  {pdf_stem}: {missing_count} batch files missing, will reprocess")
            
            return BatchProgress(
                pdf_name=data["pdf_name"],
                total_pages=data["total_pages"],
                batch_size=data["batch_size"],
                completed_batches=verified_batches,
                status=data.get("status", "pending") if verified_batches == completed_batches else "pending",
                error=data.get("error")
            )
        else:
            # Batch size or page count changed - start fresh
            log.warning(f"   Config changed for {pdf_stem}, restarting")
    
    # Create new progress
    return BatchProgress(pdf_name=pdf_stem, total_pages=total_pages)


def save_progress(output_dir: Path, progress: BatchProgress) -> None:
    """Save progress to JSON file atomically."""
    progress_path = get_progress_path(output_dir, progress.pdf_name)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(progress_path, asdict(progress))


# ============================================================================
# BATCH FILE MANAGEMENT
# ============================================================================

def get_batch_dir(output_dir: Path, pdf_stem: str) -> Path:
    """Get path to batch results directory."""
    return output_dir / pdf_stem / "_batches"


def get_batch_path(output_dir: Path, pdf_stem: str, batch_num: int) -> Path:
    """Get path to a specific batch JSON file."""
    return get_batch_dir(output_dir, pdf_stem) / f"batch_{batch_num:03d}.json"


def save_batch_result(output_dir: Path, pdf_stem: str, result: BatchResult) -> None:
    """Save batch result to JSON file."""
    batch_dir = get_batch_dir(output_dir, pdf_stem)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = get_batch_path(output_dir, pdf_stem, result.batch_num)
    atomic_json_write(batch_path, asdict(result))


def load_batch_result(output_dir: Path, pdf_stem: str, batch_num: int) -> Optional[BatchResult]:
    """Load batch result from JSON file."""
    batch_path = get_batch_path(output_dir, pdf_stem, batch_num)
    data = load_json(batch_path)
    if data:
        return BatchResult(**data)
    return None


def cleanup_batch_files(output_dir: Path, pdf_stem: str) -> None:
    """Remove temporary batch files after successful merge."""
    batch_dir = get_batch_dir(output_dir, pdf_stem)
    if batch_dir.exists():
        import shutil
        shutil.rmtree(batch_dir)
    
    # Also remove progress file
    progress_path = get_progress_path(output_dir, pdf_stem)
    if progress_path.exists():
        progress_path.unlink()


# ============================================================================
# ENDPOINT MANAGEMENT
# ============================================================================

def get_endpoint_url(endpoint_value: str) -> str:
    """
    Get endpoint URL from endpoint value.
    Leonardo: endpoints are always host:port (e.g. lrdn0690:8000).
    """
    return f"http://{endpoint_value}/v1"


def check_endpoint_health(url: str, timeout: int = 5) -> bool:
    """Check if endpoint is healthy."""
    try:
        resp = requests.get(f"{url}/models", timeout=timeout, verify=False)
        return resp.status_code == 200
    except Exception:
        return False


def probe_endpoint_inference(url: str, timeout: int = 30) -> bool:
    """Active inference probe: send a minimal 1-token chat completion.

    /v1/models can return 200 while inference itself is hung (vLLM deadlock).
    This sends a real generation request and only returns True if it completes
    within `timeout` seconds.
    """
    try:
        resp = requests.post(
            f"{url}/chat/completions",
            json={
                "model": VLLM_MODEL_NAME,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=timeout,
            verify=False,
        )
        return resp.status_code == 200
    except Exception:
        return False


def load_endpoints_from_config(config: dict) -> Dict[str, EndpointInfo]:
    """Load endpoints from config dict."""
    endpoints = {}
    
    for name, pod_id in config.get("endpoints", {}).items():
        if pod_id:  # Skip empty endpoints
            url = get_endpoint_url(pod_id)
            endpoints[name] = EndpointInfo(name=name, pod_id=pod_id, url=url)
    
    return endpoints


def get_healthy_endpoints(endpoints: Dict[str, EndpointInfo]) -> List[EndpointInfo]:
    """Get list of healthy (non-dead) endpoints."""
    healthy = []
    for ep in endpoints.values():
        if ep.is_dead:
            continue
        if check_endpoint_health(ep.url):
            healthy.append(ep)
    return healthy


def mark_endpoint_failure(ep: EndpointInfo) -> None:
    """Mark an endpoint failure, potentially marking it as dead if 3 failures within 60s."""
    now = time.time()
    ep.failure_timestamps.append(now)
    
    # Keep only failures from last 3 minutes (180 seconds)
    cutoff = now - 180
    ep.failure_timestamps = [t for t in ep.failure_timestamps if t > cutoff]
    
    # Mark as dead if 15+ failures within 3 minutes
    if len(ep.failure_timestamps) >= ENDPOINT_FAILURE_THRESHOLD:
        ep.is_dead = True
        log.warning(f"   💀 {ep.name} marked as DEAD ({len(ep.failure_timestamps)} failures in 3min)")


def mark_endpoint_success(ep: EndpointInfo) -> None:
    """Clear failure timestamps on success."""
    ep.failure_timestamps.clear()


# ============================================================================
# PDF UTILITIES
# ============================================================================

def get_pdf_page_count(pdf_path: Path) -> Optional[int]:
    """Get page count of a PDF file."""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return None


def get_qualifying_files(pdf_dir: Path, prefixes: List[str], suffix: str, enabled: bool) -> List[Path]:
    """Get list of PDF files matching the filter criteria.
    
    Args:
        prefixes: List of prefixes to match (e.g., ["XAMS_", "XMIL_", "XETR_"])
    """
    all_pdfs = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    
    if not enabled:
        return all_pdfs
    
    qualifying = []
    for pdf in all_pdfs:
        stem = pdf.stem
        # Check if stem starts with any of the prefixes
        matches_prefix = any(stem.startswith(p) for p in prefixes) if prefixes else True
        if matches_prefix and stem.endswith(suffix):
            qualifying.append(pdf)
    
    return qualifying


def is_file_complete(output_dir: Path, pdf_stem: str) -> bool:
    """Check if a file has been fully processed (has _full.json)."""
    json_path = output_dir / pdf_stem / f"{pdf_stem}_full.json"
    return json_path.exists()


# ============================================================================
# SHORTLIST CACHING
# ============================================================================

def get_shortlist_path(output_dir: Path) -> Path:
    """Get path to shortlist cache file (in target logs dir if set, else output_dir)."""
    logs_dir = get_target_logs_dir()
    return (logs_dir or output_dir) / "_shortlist.json"


def load_shortlist(output_dir: Path) -> Optional[Dict]:
    """
    Load shortlist cache from file.
    Returns dict with 'files' list and 'created' timestamp, or None if not exists.
    """
    shortlist_path = get_shortlist_path(output_dir)
    return load_json(shortlist_path)


def save_shortlist(output_dir: Path, pdf_stems: List[str]) -> None:
    """Save shortlist of unprocessed files to cache."""
    shortlist_path = get_shortlist_path(output_dir)
    data = {
        "created": datetime.now().isoformat(),
        "count": len(pdf_stems),
        "files": pdf_stems
    }
    atomic_json_write(shortlist_path, data)
    log.info(f"   💾 Saved shortlist: {len(pdf_stems)} files to process")


def clear_shortlist(output_dir: Path) -> None:
    """Clear shortlist cache file."""
    shortlist_path = get_shortlist_path(output_dir)
    if shortlist_path.exists():
        shortlist_path.unlink()


def get_unprocessed_from_shortlist(output_dir: Path, pdf_dir: Path) -> Optional[List[Path]]:
    """
    Check shortlist cache and return only files that are still unprocessed.
    Returns None if shortlist doesn't exist or is invalid.
    
    This is MUCH faster than scanning all 50k+ files every time.
    Also filters out quarantined files early to avoid expensive operations on them.
    """
    shortlist = load_shortlist(output_dir)
    if not shortlist:
        return None
    
    cached_files = shortlist.get("files", [])
    if not cached_files:
        return None
    
    created = shortlist.get("created", "unknown")
    log.info(f"\n📋 Using cached shortlist (created: {created})")
    log.info(f"   Checking {len(cached_files)} files from cache...")
    
    # Load quarantine log to skip quarantined files early
    quarantine_log = load_quarantine_log(output_dir)
    quarantined_stems = set(quarantine_log.get("files", {}).keys())
    
    # Quick check: which files are now complete or quarantined?
    still_pending = []
    newly_complete = 0
    quarantined_skipped = 0
    
    for stem in cached_files:
        if stem in quarantined_stems:
            quarantined_skipped += 1
            continue
        if is_file_complete(output_dir, stem):
            newly_complete += 1
        else:
            # Verify PDF still exists
            pdf_path = pdf_dir / f"{stem}.pdf"
            if not pdf_path.exists():
                pdf_path = pdf_dir / f"{stem}.PDF"
            if pdf_path.exists():
                still_pending.append(pdf_path)
    
    log.info(f"   ✅ {newly_complete} files completed since last run")
    if quarantined_skipped > 0:
        log.info(f"   🚫 {quarantined_skipped} quarantined files skipped")
    log.info(f"   📋 {len(still_pending)} files still pending")
    
    # Update shortlist with remaining files (excluding quarantined)
    if still_pending:
        save_shortlist(output_dir, [p.stem for p in still_pending])
    else:
        clear_shortlist(output_dir)
    
    return still_pending


# ============================================================================
# BATCH PROCESSING WORKER
# ============================================================================

def worker_log(msg: str):
    """Log from worker to a debug file."""
    try:
        with open("/tmp/chandra_worker_debug.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            f.flush()
    except:
        pass



def downscale_image_b64(b64_str: str, scale: float = 0.5) -> Tuple[str, Tuple[int, int]]:
    """
    Downscale a base64-encoded PNG image by the given scale factor.
    Returns (new_b64, (new_width, new_height)).
    """
    import base64
    import io as io_module
    from PIL import Image
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io_module.BytesIO(img_bytes))
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io_module.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), (new_w, new_h)

def worker_render_batch(args: Tuple) -> Dict:
    """
    Process pool worker: render PDF pages to base64 images.
    
    Runs in a small process pool (RENDER_WORKERS) to keep memory usage low.
    Only handles CPU-bound PDF rendering, not GPU inference.
    
    Args tuple: (pdf_path, batch_num, page_start, page_end)
    Returns dict with images_b64, img_sizes, or error info.
    """
    try:
        pdf_path, batch_num, page_start, page_end = args
        worker_log(f"Rendering {Path(pdf_path).stem} batch {batch_num} pages {page_start}-{page_end-1}")
    except Exception as e:
        return {"status": "error", "batch_num": -1, "error": f"Args error: {e}"}
    
    # Set env vars BEFORE importing chandra - settings are loaded at import time
    os.environ["IMAGE_DPI"] = str(IMAGE_DPI)
    os.environ["MIN_IMAGE_DIM"] = str(MIN_IMAGE_DIM)
    os.environ["VLLM_API_KEY"] = VLLM_API_KEY
    os.environ["VLLM_MODEL_NAME"] = VLLM_MODEL_NAME
    
    import warnings
    warnings.filterwarnings("ignore")
    
    try:
        from chandra.input import load_file  # type: ignore[import-not-found]
        import base64
        import io as io_module
        
        page_range = f"{page_start}-{page_end - 1}"
        load_start = time.time()
        
        try:
            batch_images = load_file(str(pdf_path), {"page_range": page_range})
        except Exception as e:
            return {
                "status": "bad_file",
                "batch_num": batch_num,
                "error": f"Load failed: {str(e)}"
            }
        
        if not batch_images:
            return {
                "status": "error",
                "batch_num": batch_num,
                "error": f"No images in range {page_start}-{page_end-1}"
            }
        
        # Convert to base64 and record dimensions
        images_b64 = []
        img_sizes = []
        for img in batch_images:
            buf = io_module.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            images_b64.append(b64)
            img_sizes.append((img.width, img.height))
        
        worker_log(f"Rendered {Path(pdf_path).stem} batch {batch_num}: {len(images_b64)} pages in {time.time()-load_start:.1f}s")
        
        return {
            "status": "ok",
            "batch_num": batch_num,
            "page_start": page_start,
            "page_end": page_end,
            "images_b64": images_b64,
            "img_sizes": img_sizes,
        }
    
    except Exception as e:
        return {
            "status": "error",
            "batch_num": batch_num,
            "error": str(e)[:100]
        }


def worker_http_inference(args: Tuple) -> Dict:
    """
    Thread pool worker: send pre-rendered images to vLLM endpoint.
    
    Runs in a large thread pool - very low memory overhead (~8KB per thread
    vs ~150-250MB per process). This is where 99% of wall-clock time is spent
    (waiting for GPU inference over HTTP).
    
    Uses HTTP timeout instead of signal.alarm (signals don't work in threads).
    
    Args tuple: (images_b64, img_sizes, batch_num, page_start, page_end,
                  endpoint_name, endpoint_url)
    Returns dict with status, pages data, tokens, etc.
    """
    try:
        images_b64, img_sizes, batch_num, page_start, page_end, endpoint_name, endpoint_url = args
    except Exception as e:
        return {"status": "error", "batch_num": -1, "error": f"Args error: {e}"}
    
    # Suppress noisy logging
    import warnings
    import logging as worker_logging
    warnings.filterwarnings("ignore")
    worker_logging.getLogger("httpx").setLevel(worker_logging.ERROR)
    worker_logging.getLogger("openai").setLevel(worker_logging.ERROR)
    worker_logging.getLogger("httpcore").setLevel(worker_logging.ERROR)
    
    # Quick endpoint health check
    try:
        resp = requests.get(f"{endpoint_url}/models", timeout=3, verify=False)
        if resp.status_code != 200:
            return {
                "status": "endpoint_down",
                "batch_num": batch_num,
                "endpoint": endpoint_name,
                "error": "Endpoint failed pre-check"
            }
    except Exception:
        return {
            "status": "endpoint_down",
            "batch_num": batch_num,
            "endpoint": endpoint_name,
            "error": "Endpoint unreachable"
        }
    
    start_time = time.time()
    
    try:
        from openai import OpenAI
        from chandra.output import parse_html, parse_markdown, parse_chunks  # type: ignore[import-not-found]
        from chandra.prompts import PROMPT_MAPPING  # type: ignore[import-not-found]
        from PIL import Image
        
        # HTTP timeout replaces signal.alarm (which doesn't work in threads)
        client = OpenAI(
            api_key=VLLM_API_KEY,
            base_url=endpoint_url,
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
        prompt = PROMPT_MAPPING.get("ocr_layout", "Convert this image to markdown with layout information.")
        
        results = []
        for i, img_b64 in enumerate(images_b64):
            try:
                completion = client.chat.completions.create(
                    model=VLLM_MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }],
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0,
                )
                raw_output = completion.choices[0].message.content
                token_count = completion.usage.completion_tokens
                
                # Parse raw output using chandra's parsers
                html = parse_html(raw_output, include_headers_footers=True, include_images=True)
                markdown = parse_markdown(raw_output, include_headers_footers=True, include_images=True)
                
                # Create dummy image with correct dimensions for bbox scaling
                w, h = img_sizes[i]
                dummy_img = Image.new("RGB", (w, h))
                chunks = parse_chunks(raw_output, dummy_img)
                _tally_block_types(chunks)

                results.append({
                    "html": html,
                    "markdown": markdown,
                    "chunks": chunks,
                    "token_count": token_count,
                    "error": False,
                })
            except Exception as e:
                error_str = str(e).lower()
                error_msg = str(e)[:200]
                
                # Connection / infrastructure errors → endpoint is down
                if any(x in error_str for x in ["404", "connection", "refused", "reset", "closed", "eof"]):
                    return {
                        "status": "endpoint_down",
                        "batch_num": batch_num,
                        "endpoint": endpoint_name,
                        "error": f"Connection error: {error_msg}"
                    }
                
                # Timeout errors → inference hung
                if any(x in error_str for x in ["timeout", "timed out"]):
                    return {
                        "status": "inference_timeout",
                        "batch_num": batch_num,
                        "endpoint": endpoint_name,
                        "error": f"Request timeout: {error_msg}"
                    }
                
                # 400 Bad Request → check if "prompt too long", retry with downscaled image
                if "400" in error_str or "bad request" in error_str:
                    is_prompt_too_long = "maximum model length" in error_str or "longer than" in error_str
                    worker_log(f"400 Bad Request on {endpoint_name}: {error_msg}")
                    
                    if is_prompt_too_long:
                        # Downscale image by 50% and retry once
                        worker_log(f"   → Retrying page {page_start+i} with 50% downscale")
                        try:
                            scaled_b64, (sw, sh) = downscale_image_b64(img_b64, scale=0.5)
                            retry_completion = client.chat.completions.create(
                                model=VLLM_MODEL_NAME,
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{scaled_b64}"}},
                                        {"type": "text", "text": prompt}
                                    ]
                                }],
                                max_tokens=MAX_OUTPUT_TOKENS,
                                temperature=0,
                            )
                            raw_output = retry_completion.choices[0].message.content
                            token_count = retry_completion.usage.completion_tokens
                            html = parse_html(raw_output, include_headers_footers=True, include_images=True)
                            markdown = parse_markdown(raw_output, include_headers_footers=True, include_images=True)
                            # Use ORIGINAL dimensions for bbox scaling (downscale was only for token reduction)
                            w, h = img_sizes[i]
                            dummy_img = Image.new("RGB", (w, h))
                            chunks = parse_chunks(raw_output, dummy_img)
                            _tally_block_types(chunks)
                            results.append({
                                "html": html, "markdown": markdown, "chunks": chunks,
                                "token_count": token_count, "error": False,
                            })
                            worker_log(f"   ✓ Downscale retry succeeded for page {page_start+i} ({sw}x{sh}, {token_count} tokens)")
                            continue  # Success - move to next page in batch
                        except Exception as retry_e:
                            retry_msg = str(retry_e)[:150]
                            worker_log(f"   ✗ Downscale retry also failed for page {page_start+i}: {retry_msg}")
                    
                    return {
                        "status": "error",
                        "batch_num": batch_num,
                        "endpoint": endpoint_name,
                        "error": f"Bad request (400): {error_msg}"
                    }
                
                # Other errors → log and continue to next page in batch
                worker_log(f"API error on {endpoint_name} page {page_start+i}: {error_msg}")
                results.append({
                    "html": "",
                    "markdown": "",
                    "chunks": [],
                    "token_count": 0,
                    "error": True,
                })
        
        # Check for errors in results
        if any(r["error"] for r in results):
            return {
                "status": "error",
                "batch_num": batch_num,
                "endpoint": endpoint_name,
                "error": "Generation errors in batch"
            }
        
        # Build page data
        pages_data = []
        total_tokens = 0
        
        for i, result in enumerate(results):
            page_num = page_start + i
            page_data = {
                "page_num": page_num,
                "html": result["html"],
                "markdown": result["markdown"],
                "chunks": result["chunks"],
                "page_box": None,
                "token_count": result["token_count"],
            }
            pages_data.append(page_data)
            total_tokens += result["token_count"]
        
        elapsed = time.time() - start_time
        
        return {
            "status": "success",
            "batch_num": batch_num,
            "page_start": page_start,
            "page_end": page_end,
            "pages": pages_data,
            "tokens": total_tokens,
            "time": elapsed,
            "endpoint": endpoint_name,
        }
    
    except Exception as e:
        error_str = str(e).lower()
        if any(x in error_str for x in ["timeout", "timed out"]):
            return {
                "status": "inference_timeout",
                "batch_num": batch_num,
                "endpoint": endpoint_name,
                "error": str(e)[:100],
            }
        return {
            "status": "error",
            "batch_num": batch_num,
            "endpoint": endpoint_name,
            "error": str(e)[:100],
        }


# ============================================================================
# MERGE AND FINALIZE
# ============================================================================

def merge_batches_to_json(output_dir: Path, pdf_stem: str, progress: BatchProgress) -> Dict:
    """
    Merge all batch JSONs into final _full.json file.
    
    Returns dict with num_pages, total_tokens, success status.
    """
    file_output_dir = output_dir / pdf_stem
    full_json_path = file_output_dir / f"{pdf_stem}_full.json"
    
    # Collect all pages in order
    all_pages = []
    total_tokens = 0
    total_chunks = 0
    
    for batch_num in range(progress.total_batches):
        batch_result = load_batch_result(output_dir, pdf_stem, batch_num)
        if not batch_result or not batch_result.pages:
            return {
                "success": False,
                "error": f"Missing or empty batch {batch_num}"
            }
        
        for page_data in batch_result.pages:
            all_pages.append(page_data)
            total_tokens += page_data.get("token_count", 0)
            total_chunks += len(page_data.get("chunks", []))
    
    # Verify page count
    if len(all_pages) != progress.total_pages:
        return {
            "success": False,
            "error": f"Page count mismatch: expected {progress.total_pages}, got {len(all_pages)}"
        }
    
    # Verify pages are in order and sequential
    for i, page_data in enumerate(all_pages):
        if page_data["page_num"] != i:
            return {
                "success": False,
                "error": f"Page order error: expected page {i}, got {page_data['page_num']}"
            }
    
    # Build final JSON structure
    children = []
    page_metadata = []
    
    for page_data in all_pages:
        page_num = page_data["page_num"]
        
        # Build page node
        page_node = {
            "id": f"/page/{page_num}",
            "block_type": "Page",
            "page_num": page_num,
            "bbox": page_data.get("page_box"),
            "html": page_data.get("html", ""),
            "markdown": page_data.get("markdown", ""),
            "children": page_data.get("chunks", []),
            "token_count": page_data.get("token_count", 0)
        }
        children.append(page_node)
        
        # Build page metadata
        page_metadata.append({
            "page_num": page_num,
            "page_box": page_data.get("page_box"),
            "token_count": page_data.get("token_count", 0),
            "num_chunks": len(page_data.get("chunks", [])),
            "num_images": 0  # Images not saved separately
        })
    
    # Build final document
    document = {
        "block_type": "Document",
        "children": children,
        "metadata": {
            "file_name": pdf_stem,
            "num_pages": len(all_pages),
            "total_token_count": total_tokens,
            "total_chunks": total_chunks,
            "total_images": 0,
            "pages": page_metadata
        }
    }
    
    # Write atomically
    atomic_json_write(full_json_path, document)
    
    return {
        "success": True,
        "num_pages": len(all_pages),
        "total_tokens": total_tokens
    }


def finalize_pdf(output_dir: Path, pdf_stem: str, progress: BatchProgress) -> Dict:
    """
    Finalize a PDF after all batches complete.
    Merges batches and cleans up temp files.
    """
    log.info(f"   🔄 {pdf_stem}: merging {progress.total_batches} batches...")
    
    result = merge_batches_to_json(output_dir, pdf_stem, progress)
    
    if result["success"]:
        # Cleanup temp files
        cleanup_batch_files(output_dir, pdf_stem)
        log.info(f"   ✅ {pdf_stem}: merged {result['num_pages']} pages ({result['total_tokens']:,} tokens)")
    else:
        log.error(f"   ❌ {pdf_stem}: merge failed - {result.get('error')}")
    
    return result


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def process_pdfs(
    pdf_dir: Path,
    output_dir: Path,
    endpoints: Dict[str, EndpointInfo],
    pdfs_to_process: List[Path],
    workers_per_gpu: int = WORKERS_PER_GPU,
    fresh_restart_interval: int = FRESH_RESTART_INTERVAL
) -> Dict:
    """
    Main processing loop with batch-level parallelism.
    
    Returns summary dict with success_count, error_count, etc.
    """
    start_time = time.time()
    
    # Get healthy endpoints
    healthy_endpoints = get_healthy_endpoints(endpoints)
    if not healthy_endpoints:
        log.error("❌ No healthy endpoints available!")
        return {
            "success_count": 0,
            "error_count": len(pdfs_to_process),
            "total_pages": 0,
            "total_tokens": 0,
            "elapsed": time.time() - start_time,
            "restart_triggered": False
        }
    
    log.info(f"   📊 {len(healthy_endpoints)}/{len(endpoints)} endpoints online")
    
    total_concurrency = len(healthy_endpoints) * workers_per_gpu
    
    log.info(f"\n🚀 Starting processing")
    log.info(f"   GPUs online:      {len(healthy_endpoints)}")
    log.info(f"   Total workers:    {total_concurrency}")
    log.info(f"   Batch size:       {BATCH_SIZE} pages")
    
    # Counters
    success_count = 0
    error_count = 0
    total_tokens = 0
    total_pages = 0
    start_time = time.time()
    
    # Track endpoint active requests
    endpoint_active = {ep.name: 0 for ep in endpoints.values()}
    MAX_PER_ENDPOINT = workers_per_gpu  # Match config setting
    
    # Load timeout log and quarantine log
    timeout_log = load_timeout_log(output_dir)
    quarantine_log = load_quarantine_log(output_dir)
    
    # Update quarantine based on current timeout log
    newly_quarantined = update_quarantine_from_timeout_log(output_dir, timeout_log, quarantine_log)
    if newly_quarantined > 0:
        log.info(f"   🚫 {newly_quarantined} files newly quarantined")
    
    # Filter out quarantined files
    quarantined_count = 0
    filtered_pdfs = []
    for pdf_path in pdfs_to_process:
        if is_file_quarantined(quarantine_log, pdf_path.stem):
            quarantined_count += 1
        else:
            filtered_pdfs.append(pdf_path)
    
    if quarantined_count > 0:
        log.warning(f"   🚫 Skipped {quarantined_count} quarantined files ({QUARANTINE_THRESHOLD}+ timed-out batches)")
    
    # PDF-to-endpoint affinity: each endpoint processes one PDF at a time
    # This prevents a problematic PDF from blocking ALL endpoints
    endpoint_current_pdf: Dict[str, Optional[Path]] = {ep.name: None for ep in endpoints.values()}
    pdf_assigned_endpoint: Dict[Path, str] = {}  # Which endpoint is handling which PDF
    
    skipped_flagged = 0
    
    # Build work queue: organize batches BY PDF for endpoint affinity
    pdf_work = {}
    pdf_pending_batches: Dict[Path, List[Dict]] = {}  # Per-PDF batch queues
    unassigned_pdfs: List[Path] = []  # PDFs waiting to be assigned to an endpoint
    
    for pdf_path in filtered_pdfs:
        page_count = get_pdf_page_count(pdf_path)
        if not page_count:
            log.warning(f"   ⚠️  {pdf_path.name}: Cannot read page count, skipping")
            error_count += 1
            continue
        
        progress = load_progress(output_dir, pdf_path.stem, page_count)
        progress.status = "in_progress"
        save_progress(output_dir, progress)
        
        pdf_work[pdf_path] = progress
        pdf_pending_batches[pdf_path] = []
        
        # Add pending batches (skip permanently flagged ones)
        for batch_num in progress.get_pending_batches():
            if is_batch_permanently_flagged(timeout_log, pdf_path.stem, batch_num):
                skipped_flagged += 1
                continue
            page_start = batch_num * BATCH_SIZE
            page_end = min(page_start + BATCH_SIZE, page_count)
            pdf_pending_batches[pdf_path].append({
                "pdf": pdf_path,
                "batch_num": batch_num,
                "page_start": page_start,
                "page_end": page_end,
                "attempts": 0
            })
        
        # Only add to unassigned if it has pending batches
        if pdf_pending_batches[pdf_path]:
            unassigned_pdfs.append(pdf_path)
    
    total_pending = sum(len(batches) for batches in pdf_pending_batches.values())
    # Baseline: batches already completed before this run (from previous runs)
    baseline_completed = sum(len(p.completed_batches) for p in pdf_work.values())
    if skipped_flagged > 0:
        log.warning(f"   ⚠️  Skipped {skipped_flagged} batches flagged as permanently timed out")
    log.info(f"   Total batches:    {total_pending}")
    log.info(f"   PDF affinity:     1 PDF per endpoint (prevents cascade blocking)")
    
    if not unassigned_pdfs and not any(pdf_pending_batches.values()):
        log.info("✅ All batches complete!")
        # Just finalize any incomplete PDFs
        for pdf_path, progress in pdf_work.items():
            if progress.is_complete and not is_file_complete(output_dir, pdf_path.stem):
                result = finalize_pdf(output_dir, pdf_path.stem, progress)
                if result["success"]:
                    success_count += 1
                    total_pages += result["num_pages"]
                    total_tokens += result["total_tokens"]
                else:
                    error_count += 1
        return {
            "success_count": success_count,
            "error_count": error_count,
            "total_pages": total_pages,
            "total_tokens": total_tokens,
            "elapsed": time.time() - start_time,
            "restart_triggered": False
        }
    
    def assign_pdf_to_endpoint(ep: EndpointInfo) -> Optional[Path]:
        """Assign an unassigned PDF to this endpoint."""
        if not unassigned_pdfs:
            return None
        pdf_path = unassigned_pdfs.pop(0)
        endpoint_current_pdf[ep.name] = pdf_path
        pdf_assigned_endpoint[pdf_path] = ep.name
        return pdf_path
    
    def release_pdf_from_endpoint(ep_name: str, pdf_path: Path):
        """Release a PDF from an endpoint (after completion or quarantine)."""
        if endpoint_current_pdf.get(ep_name) == pdf_path:
            endpoint_current_pdf[ep_name] = None
        if pdf_path in pdf_assigned_endpoint:
            del pdf_assigned_endpoint[pdf_path]
    
    def pick_endpoint_for_batch(batch_info: Dict) -> Optional[EndpointInfo]:
        """Pick an endpoint for a batch, respecting PDF affinity."""
        pdf_path = batch_info["pdf"]
        
        # If PDF already assigned, use that endpoint
        if pdf_path in pdf_assigned_endpoint:
            ep_name = pdf_assigned_endpoint[pdf_path]
            ep = endpoints.get(ep_name)
            if ep and not ep.is_dead and endpoint_active.get(ep_name, 0) < MAX_PER_ENDPOINT:
                return ep
            return None  # Assigned endpoint is full or dead
        
        # PDF not assigned - find an endpoint with no current PDF
        for ep in healthy_endpoints:
            if ep.is_dead:
                continue
            if endpoint_current_pdf.get(ep.name) is None:
                if endpoint_active.get(ep.name, 0) < MAX_PER_ENDPOINT:
                    # Assign this PDF to this endpoint
                    endpoint_current_pdf[ep.name] = pdf_path
                    pdf_assigned_endpoint[pdf_path] = ep.name
                    return ep
        
        return None  # No available endpoint
    
    def get_next_batch_to_submit() -> Optional[Dict]:
        """Get the next batch to submit, prioritizing PDFs on endpoints with capacity."""
        # Check all endpoints and either get a batch from their current PDF or assign a new PDF
        for ep_name in list(endpoint_current_pdf.keys()):
            ep = endpoints.get(ep_name)
            if not ep or ep.is_dead:
                continue
            if endpoint_active.get(ep_name, 0) >= MAX_PER_ENDPOINT:
                continue  # This endpoint is full
            
            pdf_path = endpoint_current_pdf.get(ep_name)
            
            if pdf_path and pdf_path in pdf_pending_batches:
                batches = pdf_pending_batches[pdf_path]
                if batches:
                    return batches.pop(0)
                else:
                    # No more batches for this PDF - release and assign new
                    release_pdf_from_endpoint(ep_name, pdf_path)
                    pdf_path = None  # Fall through to assign new PDF
            
            # No PDF assigned or just released - try to assign a new one
            if pdf_path is None and unassigned_pdfs:
                new_pdf = assign_pdf_to_endpoint(ep)
                if new_pdf and pdf_pending_batches.get(new_pdf):
                    log.info(f"   📎 {ep_name} assigned: {new_pdf.stem}")
                    return pdf_pending_batches[new_pdf].pop(0)
        
        return None
    
    def has_pending_work() -> bool:
        """Check if there's any pending work."""
        if unassigned_pdfs:
            return True
        for batches in pdf_pending_batches.values():
            if batches:
                return True
        return False
    
    # Hybrid executor: small process pool for PDF rendering, thread pool for HTTP inference
    # This reduces memory from ~150-250MB per process to ~8KB per thread for HTTP calls
    # With 30 GPUs × 4 workers: 120 threads (~1MB) vs 120 processes (~18-30GB)
    render_pool_size = min(RENDER_WORKERS, os.cpu_count() or 4)
    log.info(f"   Render workers:   {render_pool_size} (process pool)")
    log.info(f"   HTTP workers:     {total_concurrency} (thread pool)")
    
    with ProcessPoolExecutor(max_workers=render_pool_size) as render_executor, \
         ThreadPoolExecutor(max_workers=total_concurrency) as http_executor:
        # Unified futures dict tracks both render and HTTP phases
        # Each entry: {future: {"type": "render"|"http", "batch_info": ..., "ep": ..., "submit_time": ...}}
        futures = {}
        
        def submit_render(batch_info: Dict) -> bool:
            """Submit a batch for rendering, respecting PDF-endpoint affinity."""
            ep = pick_endpoint_for_batch(batch_info)
            if not ep:
                return False
            
            endpoint_active[ep.name] = endpoint_active.get(ep.name, 0) + 1
            
            args = (
                batch_info["pdf"],
                batch_info["batch_num"],
                batch_info["page_start"],
                batch_info["page_end"],
            )
            future = render_executor.submit(worker_render_batch, args)
            futures[future] = {"type": "render", "batch_info": batch_info, "ep": ep, "submit_time": time.time()}
            return True
        
        # Initial fill: assign PDFs to endpoints
        initial_assigned = 0
        for ep in healthy_endpoints:
            if unassigned_pdfs:
                pdf_path = assign_pdf_to_endpoint(ep)
                if pdf_path:
                    initial_assigned += 1
                    log.info(f"   📎 {ep.name} assigned: {pdf_path.stem}")
        
        # Submit initial renders - fill each endpoint up to MAX_PER_ENDPOINT
        consecutive_failures = 0
        while len(futures) < total_concurrency:
            batch_info = get_next_batch_to_submit()
            if not batch_info:
                break
            if not submit_render(batch_info):
                # Put it back and try another
                pdf_pending_batches[batch_info["pdf"]].insert(0, batch_info)
                consecutive_failures += 1
                if consecutive_failures > len(healthy_endpoints) * 2:
                    break
                continue
            consecutive_failures = 0
        
        log.info(f"   Submitted {len(futures)} initial batches across {initial_assigned} endpoints")
        
        # Main loop
        last_status_time = time.time()
        last_status_completed = 0  # Track completed batches at last status update
        STATUS_INTERVAL = 60  # Print status every 60 seconds
        restart_triggered = False  # Track if we need a fresh restart

        # Stall detection tracking.
        # Fix #2: arm the stall clock at job start rather than after the first PDF
        # completes. Otherwise, if every worker hangs on its first PDF the
        # detector never arms (observed as a latent bug, not the root cause of
        # the Apr-2026 stall but fixed here for safety).
        last_pdf_completion_time = time.time()
        stall_exit = False

        # Fix #3: active endpoint inference probes (not just /v1/models).
        # vLLM can serve models OK while /chat/completions hangs.
        ENDPOINT_PROBE_INTERVAL = 60
        ENDPOINT_PROBE_TIMEOUT = 30
        ENDPOINT_PROBE_FAILS_TO_KILL = 2
        last_endpoint_probe_time = time.time()
        endpoint_probe_failures: Dict[str, int] = {ep_name: 0 for ep_name in endpoints}

        # Fix #4: per-PDF submit timestamps, independent of the `futures` dict.
        # If all HTTP futures for a PDF disappear (e.g. executor silently drops
        # them) the old per-future BATCH_TIMEOUT_SECONDS check never fires
        # because there's nothing to match against. Track the last submit time
        # of any batch for each assigned PDF so we can detect "lost" work.
        pdf_last_submit_time: Dict[Path, float] = {}

        # Fix #5: release-on-desync tracking. If work is pending, no futures
        # are in flight, and no new submissions have happened for
        # DESYNC_RELEASE_SECONDS, force-release every endpoint's PDF assignment
        # so the scheduler can recover from an `endpoint_current_pdf` desync.
        # Tracked independently of STATUS_INTERVAL so tuning the log cadence
        # doesn't shift the recovery window.
        last_submit_time = time.time()

        def _wrap_submit_render(batch_info):
            """submit_render + bookkeeping for Fix #4 and Fix #5."""
            nonlocal last_submit_time
            ok = submit_render(batch_info)
            if ok:
                pdf_last_submit_time[batch_info["pdf"]] = time.time()
                last_submit_time = time.time()
            return ok

        while futures or has_pending_work():
            # Fix #1: stall check BEFORE any `continue` in the submit-loop
            # branches below. The original placement (post-`wait`) was skipped
            # by the `alive but no work submitted - retrying` path, which is
            # exactly the state the Apr-2026 stall got stuck in.
            if STALL_TIMEOUT_SECONDS > 0:
                stall_duration = time.time() - last_pdf_completion_time
                if stall_duration > STALL_TIMEOUT_SECONDS:
                    log.warning(f"⚠️ STALL DETECTED: No PDF completed in {stall_duration:.0f}s - exiting gracefully")
                    stall_exit = True
                    for future in futures:
                        future.cancel()
                    break

            # Fix #3: periodic active inference probe of every endpoint.
            now = time.time()
            if now - last_endpoint_probe_time >= ENDPOINT_PROBE_INTERVAL:
                last_endpoint_probe_time = now
                for ep_name, ep in endpoints.items():
                    if ep.is_dead:
                        continue
                    if probe_endpoint_inference(ep.url, timeout=ENDPOINT_PROBE_TIMEOUT):
                        endpoint_probe_failures[ep_name] = 0
                    else:
                        endpoint_probe_failures[ep_name] += 1
                        log.warning(f"   🔎 {ep_name} inference probe failed ({endpoint_probe_failures[ep_name]}/{ENDPOINT_PROBE_FAILS_TO_KILL})")
                        if endpoint_probe_failures[ep_name] >= ENDPOINT_PROBE_FAILS_TO_KILL:
                            log.warning(f"   💀 {ep_name} marked DEAD by inference probe (ghost-alive)")
                            ep.is_dead = True
                            # Cancel outstanding futures for this endpoint and
                            # release its PDF so other endpoints can pick it up.
                            for fut, info in list(futures.items()):
                                if info["ep"].name == ep_name:
                                    fut.cancel()
                                    futures.pop(fut, None)
                                    endpoint_active[ep_name] = max(0, endpoint_active.get(ep_name, 0) - 1)
                            assigned_pdf = endpoint_current_pdf.get(ep_name)
                            if assigned_pdf is not None:
                                # Requeue the PDF; batches already in the pending
                                # list stay, just unassign.
                                release_pdf_from_endpoint(ep_name, assigned_pdf)
                                if assigned_pdf not in unassigned_pdfs and pdf_pending_batches.get(assigned_pdf):
                                    unassigned_pdfs.append(assigned_pdf)

            # Fix #4: per-PDF stale-submit check. If a PDF's last submit is
            # older than BATCH_TIMEOUT_SECONDS and has no in-flight future,
            # the work is effectively lost. Release the endpoint and requeue.
            now = time.time()
            for pdf_path, last_submit in list(pdf_last_submit_time.items()):
                if now - last_submit <= BATCH_TIMEOUT_SECONDS:
                    continue
                has_inflight = any(info["batch_info"]["pdf"] == pdf_path for info in futures.values())
                if has_inflight:
                    continue
                # Stale assignment with no work in flight — release + requeue.
                ep_name = pdf_assigned_endpoint.get(pdf_path)
                log.warning(f"   ⏰ LOST WORK: {pdf_path.stem} last submit {now - last_submit:.0f}s ago, no futures — releasing {ep_name}")
                if ep_name:
                    release_pdf_from_endpoint(ep_name, pdf_path)
                if pdf_path not in unassigned_pdfs and pdf_pending_batches.get(pdf_path):
                    unassigned_pdfs.append(pdf_path)
                pdf_last_submit_time.pop(pdf_path, None)

            # Fix #5: release-on-desync. The observed "4 pending, 4 alive,
            # 0 unassigned" deadlock: `endpoint_current_pdf` holds PDFs nobody
            # can submit batches for. Force-release after DESYNC_RELEASE_SECONDS
            # of pending work with zero futures and zero new submissions.
            if (has_pending_work()
                    and len(futures) == 0
                    and time.time() - last_submit_time > DESYNC_RELEASE_SECONDS):
                log.warning(f"   🔧 SCHEDULER DESYNC: pending work, no futures, no submissions in {time.time() - last_submit_time:.0f}s — force-releasing all endpoint assignments")
                for ep_name in list(endpoint_current_pdf.keys()):
                    pdf_path = endpoint_current_pdf.get(ep_name)
                    if pdf_path is not None:
                        release_pdf_from_endpoint(ep_name, pdf_path)
                        if pdf_path not in unassigned_pdfs and pdf_pending_batches.get(pdf_path):
                            unassigned_pdfs.append(pdf_path)
                pdf_last_submit_time.clear()
                # Reset the trigger so we don't spam this path every tick.
                last_submit_time = time.time()

            if not futures:
                if has_pending_work():
                    # Debug: show state
                    pending_count = sum(len(batches) for batches in pdf_pending_batches.values())
                    active_eps = [ep.name for ep in endpoints.values() if not ep.is_dead]
                    log.info(f"⏳ Waiting for endpoints... ({pending_count} pending, {len(active_eps)} alive, {len(unassigned_pdfs)} unassigned PDFs)")
                    
                    time.sleep(5)
                    
                    # Try to resurrect dead endpoints by re-checking health
                    for ep in endpoints.values():
                        if ep.is_dead:
                            if check_endpoint_health(ep.url):
                                ep.is_dead = False
                                ep.failure_timestamps.clear()
                                log.info(f"   🔄 {ep.name} resurrected (health check passed)")
                    
                    healthy_endpoints = get_healthy_endpoints(endpoints)
                    if healthy_endpoints:
                        submitted_count = 0
                        consecutive_failures = 0
                        while len(futures) < total_concurrency:
                            batch_info = get_next_batch_to_submit()
                            if not batch_info:
                                break
                            if not _wrap_submit_render(batch_info):
                                pdf_pending_batches[batch_info["pdf"]].insert(0, batch_info)
                                consecutive_failures += 1
                                if consecutive_failures > len(healthy_endpoints) * 2:
                                    break
                                continue
                            submitted_count += 1
                            consecutive_failures = 0
                        if submitted_count > 0:
                            log.info(f"   📤 Submitted {submitted_count} batches")
                    
                    if not futures and not has_pending_work():
                        log.info("✅ All work completed!")
                        break
                    elif not futures:
                        # Still have work but can't submit - check why
                        alive_eps = sum(1 for ep in endpoints.values() if not ep.is_dead)
                        if alive_eps == 0:
                            log.error("❌ No endpoints available! All endpoints are dead.")
                            break
                        else:
                            log.warning(f"⚠️ {alive_eps} endpoints alive but no work submitted - retrying...")
                            continue
                else:
                    break
            
            # Wait for any future (render or HTTP) to complete.
            # Stall detection now lives at the top of the outer loop (Fix #1).
            done, _ = wait(futures.keys(), timeout=5, return_when=FIRST_COMPLETED)

            # Periodic status update
            if time.time() - last_status_time > STATUS_INTERVAL:
                elapsed = time.time() - start_time
                # Count batches completed THIS run (subtract baseline from previous runs)
                total_completed_batches = sum(len(p.completed_batches) for p in pdf_work.values()) - baseline_completed
                batches_this_interval = total_completed_batches - last_status_completed
                batches_per_min = batches_this_interval  # Since interval is 60s
                # Count active (non-dead) endpoints
                active_eps = sum(1 for ep in endpoints.values() if not ep.is_dead)
                pending_count = sum(len(batches) for batches in pdf_pending_batches.values())
                active_pdfs = [p.stem for p in endpoint_current_pdf.values() if p]
                # Count renders vs HTTP in-flight
                render_count = sum(1 for f in futures.values() if f["type"] == "render")
                http_count = sum(1 for f in futures.values() if f["type"] == "http")
                # Leonardo: core-hours per file (1 GPU-hour = 8 core-hours)
                avg_pages = sum(p.total_pages for p in pdf_work.values()) / len(pdf_work) if pdf_work else 80
                avg_batches_per_file = avg_pages / BATCH_SIZE
                if batches_per_min > 0:
                    core_hours_per_min = active_eps * (CORE_HOURS_PER_GPU_HOUR / 60)
                    files_per_min = batches_per_min / avg_batches_per_file
                    core_hours_per_file = core_hours_per_min / files_per_min
                    log.info(f"\033[1;36m   📊 Status: {http_count}+{render_count} in-flight, {pending_count} pending, {total_completed_batches} done ({batches_per_min}/min) | {len(active_pdfs)} PDFs on {active_eps} endpoints | {core_hours_per_file:.2f} core-h/file ({elapsed:.0f}s)\033[0m")
                else:
                    log.info(f"\033[1;36m   📊 Status: {http_count}+{render_count} in-flight, {pending_count} pending, {total_completed_batches} done ({batches_per_min}/min) | {len(active_pdfs)} PDFs on {active_eps} endpoints ({elapsed:.0f}s)\033[0m")
                last_status_completed = total_completed_batches
                last_status_time = time.time()
            
            # Check if fresh restart interval reached
            if fresh_restart_interval > 0 and (time.time() - start_time) >= fresh_restart_interval and not restart_triggered:
                log.info(f"🔄 Fresh restart interval reached ({fresh_restart_interval}s) - waiting 60s for in-flight batches...")
                restart_triggered = True
                # Clear all pending work
                pdf_pending_batches.clear()
                unassigned_pdfs.clear()
                # Sleep 60s to let running batches complete naturally
                time.sleep(60)
                # Force shutdown executors
                log.info(f"🔄 Initiating fresh restart now...")
                render_executor.shutdown(wait=False, cancel_futures=True)
                http_executor.shutdown(wait=False, cancel_futures=True)
                futures.clear()
                break  # Exit the processing loop
            
            # Check for timed-out futures
            current_time = time.time()
            timed_out = []
            for future, info in list(futures.items()):
                if info["type"] == "http" and current_time - info["submit_time"] > BATCH_TIMEOUT_SECONDS:
                    timed_out.append(future)
                elif info["type"] == "render" and current_time - info["submit_time"] > RENDER_TIMEOUT_SECONDS:
                    timed_out.append(future)
            
            # Handle timed-out futures
            for future in timed_out:
                info = futures.pop(future)
                used_ep = info["ep"]
                batch_info = info["batch_info"]
                endpoint_active[used_ep.name] = max(0, endpoint_active.get(used_ep.name, 0) - 1)
                
                pdf_path = batch_info["pdf"]
                batch_num = batch_info["batch_num"]
                elapsed_batch = current_time - info["submit_time"]
                
                future.cancel()
                
                if info["type"] == "render":
                    log.warning(f"   ⏰ RENDER TIMEOUT: {pdf_path.stem} batch {batch_num} ({elapsed_batch:.0f}s) - retrying")
                    batch_info["attempts"] += 1
                    if batch_info["attempts"] < MAX_RETRIES_PER_BATCH:
                        if pdf_path in pdf_pending_batches:
                            pdf_pending_batches[pdf_path].append(batch_info)
                else:
                    # HTTP timeout - flag batch permanently
                    record_batch_timeout(output_dir, timeout_log, pdf_path.stem, batch_num)
                    log.warning(f"   ⏰ TIMEOUT: {pdf_path.stem} batch {batch_num} on {used_ep.name} ({elapsed_batch:.0f}s) - FLAGGED (will skip on re-run)")
                    
                    if check_and_quarantine_pdf(output_dir, timeout_log, quarantine_log, pdf_path.stem):
                        if pdf_path in pdf_pending_batches:
                            pdf_pending_batches[pdf_path].clear()
                        release_pdf_from_endpoint(used_ep.name, pdf_path)
            
            if not done:
                continue  # Timeout, no futures completed yet
            
            for future in done:
                if future not in futures:
                    continue  # Already handled as timeout
                info = futures.pop(future)
                used_ep = info["ep"]
                batch_info = info["batch_info"]
                future_type = info["type"]
                
                pdf_path = batch_info["pdf"]
                batch_num = batch_info["batch_num"]
                
                # ============================================================
                # RENDER COMPLETION → submit HTTP inference
                # ============================================================
                if future_type == "render":
                    try:
                        render_result = future.result()
                    except Exception as e:
                        log.error(f"   💥 Render crashed: {pdf_path.name} batch {batch_num}: {e}")
                        endpoint_active[used_ep.name] = max(0, endpoint_active.get(used_ep.name, 0) - 1)
                        batch_info["attempts"] += 1
                        if batch_info["attempts"] < MAX_RETRIES_PER_BATCH:
                            if pdf_path in pdf_pending_batches:
                                pdf_pending_batches[pdf_path].append(batch_info)
                        else:
                            error_count += 1
                        continue
                    
                    if render_result["status"] == "ok":
                        # Chain: submit HTTP inference with rendered images
                        http_args = (
                            render_result["images_b64"],
                            render_result["img_sizes"],
                            batch_info["batch_num"],
                            batch_info["page_start"],
                            batch_info["page_end"],
                            used_ep.name,
                            used_ep.url,
                        )
                        http_future = http_executor.submit(worker_http_inference, http_args)
                        futures[http_future] = {"type": "http", "batch_info": batch_info, "ep": used_ep, "submit_time": time.time()}
                    elif render_result["status"] == "bad_file":
                        endpoint_active[used_ep.name] = max(0, endpoint_active.get(used_ep.name, 0) - 1)
                        log.warning(f"   ⚠️  {pdf_path.name}: Bad file - {render_result.get('error')}")
                        progress = pdf_work[pdf_path]
                        progress.status = "error"
                        progress.error = render_result.get("error")
                        save_progress(output_dir, progress)
                        if pdf_path in pdf_pending_batches:
                            pdf_pending_batches[pdf_path].clear()
                        release_pdf_from_endpoint(used_ep.name, pdf_path)
                        error_count += 1
                    else:
                        log.warning(f"   ⚠️  Render error: {pdf_path.name} batch {batch_num}: {render_result.get('error', 'unknown')}")
                        endpoint_active[used_ep.name] = max(0, endpoint_active.get(used_ep.name, 0) - 1)
                        batch_info["attempts"] += 1
                        if batch_info["attempts"] < MAX_RETRIES_PER_BATCH:
                            if pdf_path in pdf_pending_batches:
                                pdf_pending_batches[pdf_path].append(batch_info)
                        else:
                            error_count += 1
                    continue  # Don't fall through to HTTP handling
                
                # ============================================================
                # HTTP COMPLETION → handle inference result
                # ============================================================
                endpoint_active[used_ep.name] = max(0, endpoint_active.get(used_ep.name, 0) - 1)
                progress = pdf_work[pdf_path]
                
                try:
                    result = future.result()
                    
                    if result["status"] == "success":
                        mark_endpoint_success(used_ep)
                        
                        # Save batch result
                        batch_result = BatchResult(
                            batch_num=batch_num,
                            page_start=result["page_start"],
                            page_end=result["page_end"],
                            pages=result["pages"],
                            tokens=result["tokens"]
                        )
                        save_batch_result(output_dir, pdf_path.stem, batch_result)
                        
                        # Update progress
                        progress.completed_batches.append(batch_num)
                        save_progress(output_dir, progress)
                        
                        ep_display = used_ep.name.replace("leonardo-", "") if used_ep.name.startswith("leonardo-") else used_ep.name
                        print(f"   ✓ {pdf_path.stem[:25]}: batch {batch_num+1}/{progress.total_batches} ({result['tokens']:,} tok, {result.get('time', 0):.1f}s) → {ep_display}", flush=True)
                        
                        # Check if PDF is complete
                        if progress.is_complete:
                            fin_result = finalize_pdf(output_dir, pdf_path.stem, progress)
                            if fin_result["success"]:
                                success_count += 1
                                total_pages += fin_result["num_pages"]
                                total_tokens += fin_result["total_tokens"]
                                log.info(f"\033[92m✅ [{success_count}/{len(pdfs_to_process)}] {pdf_path.name} complete\033[0m")
                                # Update stall detection tracking (Fix #2: no first-PDF gate)
                                last_pdf_completion_time = time.time()
                                pdf_last_submit_time.pop(pdf_path, None)
                                # Release endpoint for next PDF
                                release_pdf_from_endpoint(used_ep.name, pdf_path)
                            else:
                                error_count += 1
                                release_pdf_from_endpoint(used_ep.name, pdf_path)
                    
                    elif result["status"] == "endpoint_down":
                        mark_endpoint_failure(used_ep)
                        batch_info["attempts"] += 1
                        
                        if batch_info["attempts"] < MAX_RETRIES_PER_BATCH:
                            # Re-add to PDF's queue
                            if pdf_path in pdf_pending_batches:
                                pdf_pending_batches[pdf_path].append(batch_info)
                        else:
                            log.error(f"   ❌ {pdf_path.name} batch {batch_num} failed after {batch_info['attempts']} attempts")
                            progress.status = "error"
                            progress.error = result.get("error")
                            save_progress(output_dir, progress)
                            error_count += 1
                    
                    elif result["status"] == "inference_timeout":
                        # HTTP timeout from OpenAI client - flag batch and move on
                        record_batch_timeout(output_dir, timeout_log, pdf_path.stem, batch_num)
                        log.warning(f"   ⏰ INFERENCE TIMEOUT: {pdf_path.stem} batch {batch_num} on {used_ep.name} - batch flagged")
                        
                        # Check if PDF should be quarantined mid-run
                        if check_and_quarantine_pdf(output_dir, timeout_log, quarantine_log, pdf_path.stem):
                            if pdf_path in pdf_pending_batches:
                                pdf_pending_batches[pdf_path].clear()
                            release_pdf_from_endpoint(used_ep.name, pdf_path)
                    
                    elif result["status"] == "bad_file":
                        log.warning(f"   ⚠️  {pdf_path.name}: Bad file - {result.get('error')}")
                        progress.status = "error"
                        progress.error = result.get("error")
                        save_progress(output_dir, progress)
                        if pdf_path in pdf_pending_batches:
                            pdf_pending_batches[pdf_path].clear()
                        release_pdf_from_endpoint(used_ep.name, pdf_path)
                        error_count += 1
                    
                    else:
                        log.warning(f"   ⚠️  HTTP error: {pdf_path.name} batch {batch_num}: status={result.get('status')}, {result.get('error', '')[:100]}")
                        batch_info["attempts"] += 1
                        if batch_info["attempts"] < MAX_RETRIES_PER_BATCH:
                            if pdf_path in pdf_pending_batches:
                                pdf_pending_batches[pdf_path].append(batch_info)
                        else:
                            error_count += 1
                
                except Exception as e:
                    log.error(f"   💥 Batch crashed: {pdf_path.name} batch {batch_num}: {e}")
                    batch_info["attempts"] += 1
                    if batch_info["attempts"] < MAX_RETRIES_PER_BATCH:
                        if pdf_path in pdf_pending_batches:
                            pdf_pending_batches[pdf_path].append(batch_info)
                    else:
                        error_count += 1
                
                # Check if this PDF is done and release the endpoint
                if pdf_path in pdf_pending_batches and not pdf_pending_batches[pdf_path]:
                    ep_name = pdf_assigned_endpoint.get(pdf_path)
                    if ep_name:
                        release_pdf_from_endpoint(ep_name, pdf_path)
            
            # Refill pipeline with new renders
            consecutive_failures = 0
            while len(futures) < total_concurrency:
                batch_info = get_next_batch_to_submit()
                if not batch_info:
                    break
                # Skip if PDF already errored
                if pdf_work[batch_info["pdf"]].status == "error":
                    continue
                if not _wrap_submit_render(batch_info):
                    pdf_pending_batches[batch_info["pdf"]].insert(0, batch_info)
                    consecutive_failures += 1
                    if consecutive_failures > len(healthy_endpoints) * 2:
                        break
                    continue
                consecutive_failures = 0

    elapsed = time.time() - start_time

    # chandra-2 block-type tally across this run (one aggregated line)
    with _block_type_counts_lock:
        if _block_type_counts:
            tally = ", ".join(f"{k}={v}" for k, v in sorted(_block_type_counts.items(), key=lambda x: -x[1]))
            log.info(f"🧾 Block-type counts: {tally}")

    # Log summary on stall exit
    if stall_exit:
        incomplete_pdfs = [p.stem for p, prog in pdf_work.items() if not prog.is_complete]
        log.warning(f"⚠️ Stall exit summary: {success_count}/{len(pdfs_to_process)} PDFs completed")
        if incomplete_pdfs:
            log.warning(f"   Incomplete PDFs ({len(incomplete_pdfs)}): {', '.join(incomplete_pdfs[:5])}{'...' if len(incomplete_pdfs) > 5 else ''}")
            log.info(f"   💡 Partial batches saved - can be resumed with chandra-api/ocr_batch_missing.py")
    
    return {
        "success_count": success_count,
        "error_count": error_count,
        "total_pages": total_pages,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "restart_triggered": restart_triggered,
        "stall_exit": stall_exit
    }


# ============================================================================
# MAIN
# ============================================================================

def run_processing_cycle() -> Dict:
    """
    Run a single processing cycle. Returns result dict with success_count, etc.
    This is extracted from main() to allow periodic fresh restarts.
    """
    # Load config
    log.info("📋 Loading configuration...")
    
    if not CONFIG_PATH.exists():
        log.error(f"❌ Config not found: {CONFIG_PATH}")
        log.info("   Please create chandra_config.json with endpoints")
        return {"success_count": 0, "error_count": 0, "total_pages": 0, "total_tokens": 0, "elapsed": 0, "exit": True}
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Endpoints are static on Leonardo (SLURM script writes config before launch)
    
    # Get PDF path from environment variable (preferred) or config (fallback)
    pdf_path = os.environ.get("PDF_INPUT") or config.get("pdf_path")
    if not pdf_path:
        log.error("❌ PDF_INPUT environment variable not set and pdf_path not configured in config")
        return {"success_count": 0, "error_count": 0, "total_pages": 0, "total_tokens": 0, "elapsed": 0, "exit": True}
    
    pdf_dir = Path(pdf_path)
    if not pdf_dir.exists():
        log.error(f"❌ PDF directory not found: {pdf_dir}")
        return {"success_count": 0, "error_count": 0, "total_pages": 0, "total_tokens": 0, "elapsed": 0, "exit": True}
    
    # Setup output directory
    output_dir = Path(os.environ.get("OCR_DIR", str(Path.cwd() / "output")))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup target-specific logs directory: logs/{target_dir_name}/
    target_logs = set_target_logs_dir(pdf_dir.name)
    add_file_logging(target_logs)
    
    log.info(f"📁 PDF source:  {pdf_dir}")
    log.info(f"📁 Output:      {output_dir}")
    log.info(f"📁 Logs:        {target_logs}")
    
    # File filtering
    filter_cfg = config.get("file_filter", {})
    # Support both single prefix (string) and multiple prefixes (list)
    prefix_cfg = filter_cfg.get("prefix", "")
    if isinstance(prefix_cfg, str):
        prefixes = [prefix_cfg] if prefix_cfg else []
    else:
        prefixes = prefix_cfg  # Already a list
    suffix = filter_cfg.get("suffix", "")
    filter_enabled = filter_cfg.get("enabled", False)
    
    if filter_enabled:
        if len(prefixes) == 1:
            log.info(f"🔍 Filter: {prefixes[0]}*{suffix}.pdf")
        else:
            log.info(f"🔍 Filter: ({', '.join(prefixes)})*{suffix}.pdf")
    
    # Try to use cached shortlist first (much faster for large directories)
    to_process = get_unprocessed_from_shortlist(output_dir, pdf_dir)
    
    if to_process is not None:
        # Using cached shortlist - skip full scan
        already_complete = []  # Not computed when using cache
        qualifying_pdfs = to_process  # For display purposes
    else:
        # No cache - do full scan
        log.info("📂 Scanning for PDF files...")
        qualifying_pdfs = get_qualifying_files(pdf_dir, prefixes, suffix, filter_enabled)
        
        log.info(f"   Found {len(qualifying_pdfs)} qualifying PDFs")
        
        if not qualifying_pdfs:
            log.error("❌ No qualifying PDF files found!")
            return {"success_count": 0, "error_count": 0, "total_pages": 0, "total_tokens": 0, "elapsed": 0, "exit": True}
        
        # Check for already complete files
        to_process = []
        already_complete = []
        
        for pdf in qualifying_pdfs:
            if is_file_complete(output_dir, pdf.stem):
                already_complete.append(pdf)
            else:
                to_process.append(pdf)
        
        # Save shortlist for next run
        if to_process:
            save_shortlist(output_dir, [p.stem for p in to_process])
    
    # Early quarantine filtering - skip quarantined files before expensive page counting
    # (The shortlist path already filters these, but this catches the full-scan path
    # and any files quarantined between shortlist save and now)
    quarantine_log_early = load_quarantine_log(output_dir)
    quarantined_stems = set(quarantine_log_early.get("files", {}).keys())
    if quarantined_stems:
        pre_filter = len(to_process)
        to_process = [p for p in to_process if p.stem not in quarantined_stems]
        quarantined_skipped = pre_filter - len(to_process)
        if quarantined_skipped > 0:
            log.info(f"   🚫 Early filter: skipped {quarantined_skipped} quarantined files")
            # Update shortlist without quarantined files
            if to_process:
                save_shortlist(output_dir, [p.stem for p in to_process])
    
    print("\n" + "=" * 70)
    print("  PROCESSING SUMMARY")
    print("=" * 70)
    if already_complete:
        print(f"  📊 Total qualifying PDFs:    {len(qualifying_pdfs)}")
        print(f"  ✅ Already complete:         {len(already_complete)}")
    if quarantined_stems:
        quarantined_in_list = len(quarantined_stems)
        print(f"  🚫 Quarantined:             {quarantined_in_list}")
    print(f"  📋 Remaining to process:     {len(to_process)}")
    print("=" * 70)
    
    if not to_process:
        log.info("✅ All files already processed!")
        return {"success_count": 0, "error_count": 0, "total_pages": 0, "total_tokens": 0, "elapsed": 0, "all_done": True}
    
    # Estimate pages
    total_pages_estimate = 0
    for pdf in to_process:
        count = get_pdf_page_count(pdf)
        if count:
            total_pages_estimate += count
    
    log.info(f"   Estimated pages: {total_pages_estimate}")
    
    # Confirm
    print(f"\n   Processing {len(to_process)} files (~{total_pages_estimate} pages)\n")
    
    # Load endpoints (fresh each cycle - clears dead state)
    log.info("🔍 Loading endpoints...")
    endpoints = load_endpoints_from_config(config)
    
    if not endpoints:
        log.error("❌ No endpoints configured!")
        return {"success_count": 0, "error_count": 0, "total_pages": 0, "total_tokens": 0, "elapsed": 0, "exit": True}
    
    log.info(f"   Found {len(endpoints)} endpoints in config")
    
    # Get workers_per_gpu and fresh_restart_interval from config
    proc_cfg = config.get("processing", {})
    workers_per_gpu = proc_cfg.get("workers_per_gpu", WORKERS_PER_GPU)
    fresh_restart_interval = proc_cfg.get("fresh_restart_interval", FRESH_RESTART_INTERVAL)
    log.info(f"   Workers per GPU: {workers_per_gpu}")
    if fresh_restart_interval > 0:
        log.info(f"   Fresh restart:   every {fresh_restart_interval}s ({fresh_restart_interval//60} min)")
    
    # Check endpoint health
    log.info("🔍 Checking endpoint health...")
    for name, ep in endpoints.items():
        if check_endpoint_health(ep.url):
            log.info(f"   ✅ {name} is ready")
        else:
            log.info(f"   ❌ {name} is offline")
            ep.is_dead = True
    
    # Process
    result = process_pdfs(pdf_dir, output_dir, endpoints, to_process, workers_per_gpu, fresh_restart_interval)
    
    return result


def main():
    # Ensure spawn method for multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    print("\n" + "=" * 70)
    print("  CHANDRA OCR v2 - Simplified Processing")
    print("=" * 70 + "\n")
    
    # Track cumulative stats across restart cycles
    cumulative_success = 0
    cumulative_errors = 0
    cumulative_pages = 0
    cumulative_tokens = 0
    overall_start = time.time()
    cycle_count = 0
    
    while True:
        cycle_count += 1
        cycle_start = time.time()
        
        if cycle_count > 1:
            print("\n" + "=" * 70)
            print(f"  🔄 FRESH RESTART - Cycle {cycle_count}")
            print("=" * 70 + "\n")
        
        # Run processing cycle
        result = run_processing_cycle()
        
        # Check for exit conditions
        if result.get("exit"):
            sys.exit(1)
        
        if result.get("all_done"):
            break
        
        # Accumulate stats
        cumulative_success += result.get("success_count", 0)
        cumulative_errors += result.get("error_count", 0)
        cumulative_pages += result.get("total_pages", 0)
        cumulative_tokens += result.get("total_tokens", 0)
        
        cycle_elapsed = time.time() - cycle_start
        
        # Print cycle summary
        print("\n" + "-" * 70)
        print(f"  📊 Cycle {cycle_count} complete: {result['success_count']} files, {result['total_pages']} pages in {cycle_elapsed:.0f}s")
        print("-" * 70)
        
        # Check if fresh restart was triggered by the processing loop
        if result.get("restart_triggered"):
            log.info(f"🔄 Fresh restart triggered - reloading endpoints and batch queue...")
            continue  # Loop back for fresh restart
        else:
            # Normal completion - exit the loop
            break
    
    # Final summary
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print("  PROCESSING COMPLETE")
    print("=" * 70)
    if cycle_count > 1:
        print(f"  🔄 Total cycles:     {cycle_count}")
    print(f"  ⏱️  Total time:      {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"  ✅ Success:         {cumulative_success} files ({cumulative_pages} pages)")
    print(f"  ❌ Errors:          {cumulative_errors}")
    print(f"  🔢 Total Tokens:    {cumulative_tokens:,}")
    print("=" * 70)
    
    if cumulative_success > 0 and overall_elapsed > 0:
        pages_per_min = (cumulative_pages / overall_elapsed) * 60
        print(f"\n📊 Performance: {pages_per_min:.1f} pages/min")


if __name__ == "__main__":
    main()
