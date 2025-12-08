#!/usr/bin/env python3
"""
Provenance ESG Processing Pipeline
===================================
Batch processes PDF files (sustainability/ESG reports) using Chandra OCR.
Outputs structured Markdown, HTML, and JSON with page metadata.

Architecture:
- Uses ProcessPoolExecutor for file-level parallelism (pypdfium2 isolation)
- Supports multiple vLLM endpoints with automatic failover
- Robust error handling with retry logic

Configuration:
- Set VLLM_ENDPOINTS env var or modify VLLM_ENDPOINTS dict below
- Adjust BATCH_SIZE, FILES_PER_GPU for throughput tuning

Usage:
    python scripts/run_pipeline.py

Environment Variables:
    VLLM_API_BASE    - vLLM server URL (default: http://localhost:8000/v1)
    VLLM_API_KEY     - API key (default: EMPTY)
    VLLM_MODEL_NAME  - Model name (default: chandra)
    IMAGE_DPI        - PDF rendering DPI (default: 200)
    MIN_IMAGE_DIM    - Minimum image dimension (default: 1536)
"""

import os
import sys
import time
import requests
import random
import datetime
import re
import json
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "input"      # Place PDFs here
OUTPUT_DIR = PROJECT_DIR / "output"    # Results go here

# vLLM Server settings
# Format: {"endpoint_name": "pod_id_or_host"}
# For RunPod: pod_id (e.g., "abc123xyz") -> https://{pod_id}-8000.proxy.runpod.net/v1
# For local: "localhost:8000" -> http://localhost:8000/v1
VLLM_ENDPOINTS = json.loads(os.environ.get("VLLM_ENDPOINTS", "{}")) or {
    "local": "localhost:8000",
}

VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "chandra")

# Processing options
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "28"))           # Pages per batch
FILES_PER_GPU = int(os.environ.get("FILES_PER_GPU", "4"))      # Concurrent files per GPU
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))          # Retries per file
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))

# Image quality
IMAGE_DPI = int(os.environ.get("IMAGE_DPI", "200"))
MIN_IMAGE_DIM = int(os.environ.get("MIN_IMAGE_DIM", "1536"))

# Output options
SAVE_HTML = True
SAVE_IMAGES = True
INCLUDE_IMAGES = True
IMAGES_AS_TEXT = True
INCLUDE_HEADERS_FOOTERS = True
PAGINATE_OUTPUT = True
PAGE_RANGE = os.environ.get("PAGE_RANGE", None)  # e.g., "0-10" for first 11 pages

# Skip already processed files
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "false").lower() == "true"

# ============================================================================
# UTILS
# ============================================================================

def log(msg: str) -> None:
    """Print timestamped log message."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def get_endpoint_url(endpoint_id: str) -> str:
    """Convert endpoint ID to full URL."""
    if ":" in endpoint_id:
        # Local or direct host:port
        return f"http://{endpoint_id}/v1"
    else:
        # RunPod format
        return f"https://{endpoint_id}-8000.proxy.runpod.net/v1"


def check_endpoint_health(name: str, endpoint_id: str) -> bool:
    """Check if endpoint is responding."""
    try:
        url = get_endpoint_url(endpoint_id)
        resp = requests.get(f"{url}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def check_endpoint_health_by_url(url: str) -> bool:
    """Check health using full URL."""
    try:
        resp = requests.get(f"{url}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_pdf_page_count(pdf_path: Path) -> int | None:
    """Get page count of a PDF file."""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return None


# ============================================================================
# WORKER PROCESS
# ============================================================================

def worker_process_file(args: tuple) -> dict:
    """
    Worker function running in a separate process.
    Self-contained to ensure clean state for pypdfium2.
    
    Args:
        args: Tuple of (pdf_path, endpoint_name, endpoint_url)
    
    Returns:
        Dict with status and metadata
    """
    pdf_path, endpoint_name, endpoint_url = args
    
    # Quick endpoint health check BEFORE loading PDF
    if not check_endpoint_health_by_url(endpoint_url):
        return {
            "status": "endpoint_down",
            "file": pdf_path.name,
            "error": "Endpoint failed pre-check",
            "endpoint": endpoint_name
        }
    
    # File validation
    try:
        if pdf_path.stat().st_size < 100:
            return {"status": "bad_file", "file": pdf_path.name, "error": "File too small (<100 bytes)"}
        
        with open(pdf_path, 'rb') as f:
            header = f.read(1024)
            if b'%PDF-' not in header:
                return {"status": "bad_file", "file": pdf_path.name, "error": "Missing PDF header"}
    except Exception as e:
        return {"status": "bad_file", "file": pdf_path.name, "error": str(e)}

    # Setup environment
    os.environ["VLLM_API_BASE"] = endpoint_url
    os.environ["VLLM_API_KEY"] = VLLM_API_KEY
    os.environ["VLLM_MODEL_NAME"] = VLLM_MODEL_NAME
    os.environ["IMAGE_DPI"] = str(IMAGE_DPI)
    os.environ["MIN_IMAGE_DIM"] = str(MIN_IMAGE_DIM)
    
    # Import locally to avoid shared state
    sys.path.insert(0, str(PROJECT_DIR))
    from chandra.input import load_file
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem
    from chandra.scripts.cli import save_merged_output
    
    start_time = time.time()
    total_tokens = 0
    
    try:
        # Load PDF
        config = {}
        if PAGE_RANGE:
            if isinstance(PAGE_RANGE, str):
                config = {"page_range": PAGE_RANGE}
            else:
                return {"status": "bad_file", "file": pdf_path.name, 
                        "error": f"PAGE_RANGE must be None or string, got {type(PAGE_RANGE).__name__}"}
        
        try:
            images = load_file(str(pdf_path), config)
        except Exception as e:
            return {"status": "bad_file", "file": pdf_path.name, "error": f"Load failed: {str(e)}"}

        if not images:
            return {"status": "bad_file", "file": pdf_path.name, "error": "No images extracted"}

        num_pages = len(images)
        file_stem = pdf_path.stem[:30]
        
        # Initialize model
        model = InferenceManager(method="vllm")
        all_results = []
        batch_times = []
        
        # Inference loop with progress logging
        for batch_start in range(0, len(images), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(images))
            batch_images = images[batch_start:batch_end]
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (num_pages + BATCH_SIZE - 1) // BATCH_SIZE
            
            batch = [
                BatchInputItem(image=img, prompt_type="ocr_layout")
                for img in batch_images
            ]
            
            generate_kwargs = {
                "include_images": INCLUDE_IMAGES,
                "images_as_text": IMAGES_AS_TEXT,
                "include_headers_footers": INCLUDE_HEADERS_FOOTERS,
                "max_retries": 0  # Don't retry internally
            }
            if MAX_OUTPUT_TOKENS:
                generate_kwargs["max_output_tokens"] = MAX_OUTPUT_TOKENS
            
            batch_start_time = time.time()
            try:
                results = model.generate(batch, **generate_kwargs)
            except Exception as e:
                error_str = str(e)
                if "404" in error_str or "Connection" in error_str or "timeout" in error_str.lower():
                    return {
                        "status": "endpoint_down",
                        "file": pdf_path.name,
                        "error": error_str[:100],
                        "endpoint": endpoint_name
                    }
                raise
            
            batch_elapsed = time.time() - batch_start_time
            batch_times.append(batch_elapsed)
            
            # Check for vLLM errors
            if any(r.error for r in results):
                return {
                    "status": "endpoint_down",
                    "file": pdf_path.name,
                    "error": "vLLM generation failed",
                    "endpoint": endpoint_name
                }
            
            all_results.extend(results)
            batch_tokens = sum(r.token_count for r in results)
            total_tokens += batch_tokens
            
            # Progress logging
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = total_batches - batch_num
            eta_seconds = remaining_batches * avg_batch_time
            eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds >= 60 else f"{eta_seconds:.0f}s"
            
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}]    {file_stem}: pages {batch_start+1}-{batch_end}/{num_pages} "
                  f"({batch_num}/{total_batches}) {batch_elapsed:.1f}s, ETA {eta_str} -> {endpoint_name}")
        
        # Validate results
        if not all_results or all(not r.markdown for r in all_results):
            return {
                "status": "error",
                "file": pdf_path.name,
                "error": "No valid output generated",
                "endpoint": endpoint_name
            }
        
        # Save output
        save_merged_output(
            OUTPUT_DIR,
            pdf_path.name,
            all_results,
            save_images=SAVE_IMAGES,
            save_html=SAVE_HTML,
            paginate_output=PAGINATE_OUTPUT,
        )

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "file": pdf_path.name,
            "tokens": total_tokens,
            "pages": num_pages,
            "time": elapsed,
            "endpoint": endpoint_name
        }
        
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "Connection" in error_str or "timeout" in error_str.lower():
            return {
                "status": "endpoint_down",
                "file": pdf_path.name,
                "error": error_str[:100],
                "endpoint": endpoint_name
            }
        return {
            "status": "error",
            "file": pdf_path.name,
            "error": error_str[:100],
            "endpoint": endpoint_name
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for the pipeline."""
    # Ensure spawn method for safety
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Check input directory
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        log(f"Created input directory: {INPUT_DIR}")
        log("Place PDF files in this directory and run again.")
        sys.exit(0)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load PDFs
    log("Scanning for PDF files...")
    all_pdfs = sorted(INPUT_DIR.glob("*.pdf")) + sorted(INPUT_DIR.glob("*.PDF"))
    log(f"Found {len(all_pdfs)} PDFs in {INPUT_DIR}")
    
    pending_files = []
    skipped_existing = 0
    total_pages_estimate = 0
    
    for pdf in all_pdfs:
        # Check if already processed
        if SKIP_EXISTING:
            out_folder = OUTPUT_DIR / pdf.stem
            out_file = out_folder / f"{pdf.stem}.md"
            if out_file.exists():
                skipped_existing += 1
                continue
        
        page_count = get_pdf_page_count(pdf)
        total_pages_estimate += page_count if page_count else 0
        pending_files.append({"path": pdf, "attempts": 0, "excluded": set(), "pages": page_count})
    
    log(f"Queued: {len(pending_files)} files (~{total_pages_estimate} pages), skipped {skipped_existing} existing")
    
    if not pending_files:
        log("No files to process.")
        sys.exit(0)

    # Check endpoints
    log("Checking endpoint health...")
    healthy_endpoints = []
    endpoint_pod_ids = {}
    
    for name, endpoint_id in VLLM_ENDPOINTS.items():
        if check_endpoint_health(name, endpoint_id):
            url = get_endpoint_url(endpoint_id)
            healthy_endpoints.append((name, url))
            endpoint_pod_ids[name] = endpoint_id
            log(f"   [OK] {name} is ready")
        else:
            log(f"   [--] {name} is offline")
    
    if not healthy_endpoints:
        log("ERROR: No healthy endpoints available!")
        sys.exit(1)

    # Process
    total_concurrency = len(healthy_endpoints) * FILES_PER_GPU
    log(f"Starting: {len(pending_files)} files, {len(healthy_endpoints)} GPUs, {total_concurrency} workers")
    
    success_count = 0
    error_count = 0
    bad_file_count = 0
    total_tokens = 0
    total_pages_processed = 0
    completed_files = 0
    start_time = time.time()
    
    worker_stats = {name: {"files": 0, "time": 0.0, "pages": 0} for name, _ in healthy_endpoints}
    dead_endpoints = set()
    
    def pick_endpoint(excluded):
        all_excluded = excluded | dead_endpoints
        candidates = [ep for ep in healthy_endpoints if ep[0] not in all_excluded]
        if not candidates:
            candidates = [ep for ep in healthy_endpoints if ep[0] not in dead_endpoints]
        if not candidates:
            return None
        return random.choice(candidates)
    
    def verify_endpoint(name):
        if name in endpoint_pod_ids:
            return check_endpoint_health(name, endpoint_pod_ids[name])
        return False

    with ProcessPoolExecutor(max_workers=total_concurrency) as executor:
        futures = {}
        
        # Initial fill
        while len(futures) < total_concurrency and pending_files:
            info = pending_files.pop(0)
            ep = pick_endpoint(info["excluded"])
            if ep is None:
                log("ERROR: No healthy endpoints remaining!")
                pending_files.insert(0, info)
                break
            ep_name, ep_url = ep
            args = (info["path"], ep_name, ep_url)
            futures[executor.submit(worker_process_file, args)] = (info, ep_name)
            
        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            
            for future in done:
                info, used_ep = futures.pop(future)
                pdf_name = info["path"].name
                
                try:
                    res = future.result()
                    
                    if res["status"] == "success":
                        completed_files += 1
                        pages = res.get("pages", 0)
                        total_pages_processed += pages
                        log(f"[{completed_files}] OK: {pdf_name} ({pages} pages, {res['time']:.1f}s) -> {res['endpoint']}")
                        success_count += 1
                        total_tokens += res["tokens"]
                        if res['endpoint'] in worker_stats:
                            worker_stats[res['endpoint']]["files"] += 1
                            worker_stats[res['endpoint']]["time"] += res['time']
                            worker_stats[res['endpoint']]["pages"] += pages
                        
                    elif res["status"] == "bad_file":
                        log(f"WARN: {pdf_name}: Invalid PDF ({res['error']}) - Skipping")
                        bad_file_count += 1
                        completed_files += 1
                        
                    elif res["status"] == "endpoint_down":
                        log(f"ENDPOINT: {pdf_name} error on {used_ep}: {res['error'][:30]}...")
                        
                        if not verify_endpoint(used_ep):
                            if used_ep not in dead_endpoints:
                                log(f"   {used_ep} is DOWN - removing from pool")
                                dead_endpoints.add(used_ep)
                        
                        info["excluded"].add(used_ep)
                        remaining = [ep for ep in healthy_endpoints if ep[0] not in dead_endpoints]
                        if remaining:
                            log(f"   Re-routing {pdf_name} ({len(remaining)} endpoints left)")
                            pending_files.insert(0, info)
                        else:
                            log(f"   No endpoints left for {pdf_name}")
                            error_count += 1
                            completed_files += 1
                        
                    elif res["status"] == "skipped":
                        completed_files += 1
                        
                    else:
                        log(f"ERROR: {pdf_name} failed on {used_ep}: {res.get('error', 'Unknown')[:50]}")
                        info["attempts"] += 1
                        info["excluded"].add(used_ep)
                        
                        if info["attempts"] < MAX_RETRIES:
                            log(f"   Re-queuing {pdf_name} (attempt {info['attempts']+1}/{MAX_RETRIES})")
                            pending_files.append(info)
                        else:
                            log(f"   {pdf_name} FAILED permanently")
                            error_count += 1
                            completed_files += 1
                            
                except Exception as e:
                    log(f"CRASH: Process crashed on {pdf_name}: {e}")
                    error_count += 1
                    completed_files += 1
            
            # Check for all endpoints down
            remaining_endpoints = [ep for ep in healthy_endpoints if ep[0] not in dead_endpoints]
            if not remaining_endpoints and pending_files:
                log("ERROR: All endpoints down! Stopping.")
                break
            
            # Refill workers
            current_concurrency = len(remaining_endpoints) * FILES_PER_GPU
            while len(futures) < current_concurrency and pending_files:
                info = pending_files.pop(0)
                ep = pick_endpoint(info["excluded"])
                if ep is None:
                    pending_files.insert(0, info)
                    break
                ep_name, ep_url = ep
                args = (info["path"], ep_name, ep_url)
                futures[executor.submit(worker_process_file, args)] = (info, ep_name)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"  FINISHED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Success:   {success_count} files ({total_pages_processed} pages)")
    print(f"  Bad Files: {bad_file_count}")
    print(f"  Errors:    {error_count}")
    print(f"  Tokens:    {total_tokens:,}")
    if dead_endpoints:
        print(f"  Dead Endpoints: {', '.join(sorted(dead_endpoints))}")
    print("\n  WORKER STATISTICS:")
    print(f"  {'Endpoint':<15} | {'Files':<6} | {'Pages':<7} | {'Time (s)':<10} | {'Sec/Page':<10}")
    print("  " + "-"*58)
    for name, stats in sorted(worker_stats.items()):
        status = " (dead)" if name in dead_endpoints else ""
        if stats["files"] > 0:
            sec_per_page = stats['time'] / stats['pages'] if stats['pages'] > 0 else 0
            print(f"  {name:<15} | {stats['files']:<6} | {stats['pages']:<7} | {stats['time']:<10.1f} | {sec_per_page:<10.2f}{status}")
        else:
            print(f"  {name:<15} | 0      | 0       | 0.0        | -{status}")
    print("="*60)
    
    if pending_files:
        log(f"WARNING: {len(pending_files)} files could not be processed")


if __name__ == "__main__":
    main()

