import os
import re
import base64
import io
import time
import subprocess
import threading
import requests

import runpod
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb guard for large PDFs
from pdf2image import convert_from_bytes
from html import escape

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/baidu/Unlimited-OCR"
MAX_PAGES = 100
MAX_NEW_TOKENS = 8192
VLLM_PORT = 8000
VLLM_URL = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
VLLM_HEALTH_URL = f"http://localhost:{VLLM_PORT}/health"

vllm_process = None

def log(msg):
    print(f"[BOOT] {msg}", flush=True)

# ===============================
# OUTPUT POST-PROCESSING
# ===============================
def clean_unlimited_ocr_output(raw_text: str) -> str:
    """
    Clean Unlimited-OCR raw output by:
    1. Extracting text content from <|ref|>…</|ref|> grounding tags
    2. Removing <|det|>…</|det|> coordinate bounding boxes
    3. Preserving all markdown structure (tables, headings, lists, etc.)
    """
    if not raw_text:
        return ""

    text = raw_text

    # Extract content from <|ref|>...</|ref|> tags (keep the text inside)
    text = re.sub(r'<\|ref\|>(.*?)</\|ref\|>', r'\1', text, flags=re.DOTALL)

    # Remove <|det|>...</|det|> coordinate boxes entirely
    text = re.sub(r'<\|det\|>.*?</\|det\|>', '', text, flags=re.DOTALL)

    # Remove any remaining special tokens from the model
    text = re.sub(r'<\|[^|]*\|>', '', text)

    # Clean up excessive whitespace but preserve markdown structure
    # Collapse multiple blank lines into at most two (keeps paragraph separation)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove trailing whitespace per line
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    return text.strip()

# ===============================
# HTML CONVERSION
# ===============================
def convert_ocr_to_html(text: str, preserve_layout: bool = False) -> str:
    """
    Convert OCR tagged output into clean HTML.
    
    Input format:
        title [x1, y1, x2, y2]Content
        text  [x1, y1, x2, y2]Content
        table [x1, y1, x2, y2]<table>...</table>
        image [x1, y1, x2, y2]
    
    Args:
        text: Raw OCR text with bounding box annotations.
        preserve_layout: If True, uses absolute CSS positioning to recreate
                         the original document layout. If False, outputs a
                         clean reading-order HTML flow.
    
    Returns:
        HTML string wrapped in <div class="ocr-page">.
    """
    if not text or text.strip() in ("[Empty or unreadable page]", ""):
        return '<div class="ocr-page empty">[Empty or unreadable page]</div>'

    lines = text.strip().split('\n')
    elements = []
    max_x, max_y = 0, 0

    # Pattern: tag [x1, y1, x2, y2]content
    line_re = re.compile(
        r'^(title|text|table|image)\s+\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\](.*)$'
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = line_re.match(line)
        if not m:
            # Fallback: plain text line that didn't match pattern
            elements.append(f'<p class="ocr-text">{escape(line)}</p>')
            continue

        tag_type, x1, y1, x2, y2, content = m.groups()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

        # Build style for absolute positioning
        bbox_attr = f'[{x1},{y1},{x2},{y2}]'
        if preserve_layout:
            base_style = f'position:absolute;left:{x1}px;top:{y1}px;width:{x2 - x1}px;'
        else:
            base_style = ''

        if tag_type == 'title':
            inner = escape(content)
            if preserve_layout:
                style = base_style + f'height:{y2 - y1}px;margin:0;'
                el = f'<h2 class="ocr-title" style="{style}" data-bbox="{bbox_attr}">{inner}</h2>'
            else:
                el = f'<h2 class="ocr-title" data-bbox="{bbox_attr}">{inner}</h2>'

        elif tag_type == 'text':
            inner = escape(content)
            if preserve_layout:
                style = base_style + f'height:{y2 - y1}px;margin:0;'
                el = f'<p class="ocr-text" style="{style}" data-bbox="{bbox_attr}">{inner}</p>'
            else:
                el = f'<p class="ocr-text" data-bbox="{bbox_attr}">{inner}</p>'

        elif tag_type == 'table':
            # The model wraps table text in <table>...</table> but usually
            # without proper <tr><td>. We extract the inner text and build a
            # simple table structure. You can customize this parsing logic.
            table_match = re.search(r'<table>(.*?)</table>', content, re.DOTALL)
            if table_match:
                raw = table_match.group(1).strip()
            else:
                raw = content.strip()

            # Heuristic: split by newlines for rows, then by colon for key-value
            rows_html = ""
            # First try to split by obvious line breaks or commas
            row_candidates = [r.strip() for r in re.split(r'[\n,](?=\w+[:])', raw) if r.strip()]
            if len(row_candidates) <= 1:
                # If no key-value pattern found, just split by newlines
                row_candidates = [r.strip() for r in raw.split('\n') if r.strip()]
            if len(row_candidates) <= 1:
                # Last resort: split by commas
                row_candidates = [r.strip() for r in raw.split(',') if r.strip()]

            for row in row_candidates:
                if ':' in row and row.count(':') == 1 and len(row) < 200:
                    key, val = row.split(':', 1)
                    rows_html += f'<tr><td class="ocr-td-key"><strong>{escape(key.strip())}</strong></td><td class="ocr-td-val">{escape(val.strip())}</td></tr>'
                else:
                    rows_html += f'<tr><td colspan="2">{escape(row)}</td></tr>'

            if not rows_html:
                rows_html = f'<tr><td>{escape(raw)}</td></tr>'

            inner = f'<table class="ocr-table">{rows_html}</table>'

            if preserve_layout:
                style = base_style
                el = f'<div class="ocr-table-wrapper" style="{style}" data-bbox="{bbox_attr}">{inner}</div>'
            else:
                el = f'<div class="ocr-table-wrapper" data-bbox="{bbox_attr}">{inner}</div>'

        elif tag_type == 'image':
            if preserve_layout:
                style = base_style + f'height:{y2 - y1}px;background:#f0f0f0;border:1px dashed #999;display:flex;align-items:center;justify-content:center;'
                el = f'<div class="ocr-image" style="{style}" data-bbox="{bbox_attr}">[Image]</div>'
            else:
                el = f'<div class="ocr-image" data-bbox="{bbox_attr}">[Image]</div>'

        elements.append(el)

    # Wrap page
    if preserve_layout and max_x > 0 and max_y > 0:
        container_style = (
            f'position:relative;width:{max_x}px;height:{max_y}px;'
            f'border:1px solid #ddd;background:#fff;margin-bottom:20px;'
        )
        html = f'<div class="ocr-page" style="{container_style}">\n' + '\n'.join(elements) + '\n</div>'
    else:
        html = '<div class="ocr-page">\n' + '\n'.join(elements) + '\n</div>'

    return html


def build_full_html(pages_html: list, title: str = "OCR Document") -> str:
    """
    Wrap per-page HTML fragments into a complete HTML document with basic CSS.
    """
    pages_joined = '\n'.join(
        f'<div class="page-container" data-page="{i+1}">{html}</div>'
        for i, html in enumerate(pages_html)
    )

    css = """
    <style>
        body { font-family: Georgia, serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 20px; color: #333; background: #fafafa; }
        .page-container { background: #fff; border: 1px solid #e0e0e0; padding: 40px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .ocr-page { position: relative; }
        .ocr-title { color: #1a1a1a; font-size: 1.4em; margin-top: 1.2em; margin-bottom: 0.4em; border-bottom: 1px solid #eee; padding-bottom: 0.2em; }
        .ocr-text { margin: 0.6em 0; text-align: justify; }
        .ocr-table-wrapper { margin: 1em 0; overflow-x: auto; }
        .ocr-table { border-collapse: collapse; width: 100%; font-size: 0.95em; }
        .ocr-table td { border: 1px solid #ccc; padding: 6px 10px; vertical-align: top; }
        .ocr-td-key { white-space: nowrap; background: #f7f7f7; width: 30%; }
        .ocr-image { color: #666; font-style: italic; font-size: 0.9em; margin: 0.5em 0; }
        .empty { color: #999; font-style: italic; text-align: center; padding: 40px; }
    </style>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{escape(title)}</title>
    {css}
</head>
<body>
{pages_joined}
</body>
</html>"""

# ===============================
# HALLUCINATION DETECTION
# ===============================
def is_hallucinated_output(text: str) -> bool:
    """
    Detect genuinely hallucinated or empty output.
    This is intentionally conservative — we preserve tables and structured
    content (pipe characters, dashes, etc.) since the model is designed
    to produce structured markdown output including tables.
    """
    if not text or len(text.strip()) < 10:
        return True

    text_lower = text.lower()

    # Only flag truly empty / generic placeholder responses
    hallucination_indicators = [
        "this page is blank", "no text found", "empty page",
        "the image appears to be", "there is no visible text",
        "the document appears to be blank", "i cannot see any text",
    ]
    for indicator in hallucination_indicators:
        if indicator in text_lower:
            return True

    # Detect excessive repetition (same line repeated many times)
    lines = text.strip().split('\n')
    if len(lines) > 20:
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        unique_lines = set(non_empty_lines)
        if len(non_empty_lines) > 0 and len(unique_lines) < 3:
            return True

    # Check if content is essentially empty (only special chars / whitespace)
    if sum(c.isalnum() for c in text) < 10:
        return True

    return False

# ===============================
# IMAGE DECODING
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    target_width = 1600
    scale = target_width / img.width
    img = img.resize((target_width, int(img.height * scale)), Image.BICUBIC)
    return img

def decode_pdf(b64):
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes, dpi=300, fmt="png", thread_count=4, use_pdftocairo=True,
        size=(1400, None),  # cap width to keep pixel count under bomb limit
    )
    # Resize oversized pages to keep memory under control
    resized = []
    for img in images[:MAX_PAGES]:
        img = img.convert("RGB")
        if img.width > 1600:
            scale = 1600 / img.width
            img = img.resize((1600, int(img.height * scale)), Image.BICUBIC)
        resized.append(img)
    return resized

def image_to_base64_url(img: Image.Image) -> str:
    """Convert a PIL image to a base64 data URL for the vLLM API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ===============================
# vLLM SERVER MANAGEMENT
# ===============================
def stream_output(pipe, label):
    """Stream subprocess output to stdout."""
    for line in iter(pipe.readline, ''):
        print(f"[vLLM {label}] {line}", end='', flush=True)

def start_vllm_server():
    """Start the vLLM server as a background process."""
    global vllm_process

    log("Starting vLLM server...")
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--port", str(VLLM_PORT),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", "32768",
        "--max-num-seqs", "8",
        "--gpu-memory-utilization", "0.90",
        "--logits_processors",
        "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
    ]

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Stream vLLM output in a background thread
    t = threading.Thread(target=stream_output, args=(vllm_process.stdout, "OUT"), daemon=True)
    t.start()

    # Wait for the server to be ready
    max_wait = 300  # 5 minutes
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(VLLM_HEALTH_URL, timeout=2)
            if r.status_code == 200:
                log(f"vLLM server ready in {time.time() - start:.1f}s")
                return True
        except requests.ConnectionError:
            pass

        # Check if process died
        if vllm_process.poll() is not None:
            log(f"vLLM server exited with code {vllm_process.returncode}")
            return False

        time.sleep(2)

    log("vLLM server timed out!")
    return False

# ===============================
# OCR VIA vLLM
# ===============================
OCR_PROMPT_TEXT = "<image>document parsing."

def ocr_page(image: Image.Image) -> str:
    """Send a single page to the vLLM server for OCR."""
    img_url = image_to_base64_url(image)

    payload = {
        "model": MODEL_PATH,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": OCR_PROMPT_TEXT
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img_url}
                    },
                ]
            }
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "extra_body": {
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": 35, "window_size": 128},
        },
    }

    resp = requests.post(VLLM_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    raw_text = data["choices"][0]["message"]["content"].strip()

    # Clean grounding tokens, preserve all markdown structure (tables etc.)
    md = clean_unlimited_ocr_output(raw_text)

    return md

def ocr_batch(images: list) -> list:
    """Process multiple pages sequentially via the vLLM server."""
    results = []
    for img in images:
        text = ocr_page(img)
        results.append(text)
    return results

# ===============================
# HANDLER
# ===============================
def handler(event):
    try:
        input_data = event.get("input", {})
        
        if "image" in input_data:
            pages = [decode_image(input_data["image"])]
        elif "file" in input_data:
            pages = decode_pdf(input_data["file"])
        else:
            return {"status": "error", "message": "Missing image or file"}

        # Options
        output_format = input_data.get("output_format", "json").lower()  # "json" or "html"
        preserve_layout = input_data.get("preserve_layout", False)       # True = absolute CSS positioning
        full_html = input_data.get("full_html", True)                    # True = wrap in <html><body>

        total_pages = len(pages)
        log(f"Processing {total_pages} pages via vLLM...")
        start_time = time.time()

        batch_results = ocr_batch(pages)

        extracted_pages = []
        pages_html = []

        for j, text in enumerate(batch_results):
            page_num = j + 1

            if text.upper().startswith("EMPTY_PAGE"):
                text = "[Empty or unreadable page]"
            elif is_hallucinated_output(text):
                log(f"Warning: Page {page_num} appears to be hallucinated")
                text = "[Empty or unreadable page]"

            extracted_pages.append({"page": page_num, "text": text})

            # Generate HTML for this page
            if output_format == "html":
                page_html = convert_ocr_to_html(text, preserve_layout=preserve_layout)
                pages_html.append(page_html)

        elapsed = time.time() - start_time
        log(f"Completed {total_pages} pages in {elapsed:.1f}s ({elapsed/total_pages:.1f}s/page)")

        # Build response
        if output_format == "html":
            if full_html:
                html_output = build_full_html(pages_html, title="OCR Result")
                return {
                    "status": "success",
                    "total_pages": len(extracted_pages),
                    "format": "html",
                    "html": html_output
                }
            else:
                # Return per-page HTML fragments
                return {
                    "status": "success",
                    "total_pages": len(extracted_pages),
                    "format": "html",
                    "pages": [{"page": i+1, "html": html} for i, html in enumerate(pages_html)]
                }
        else:
            # Default JSON with raw text
            return {
                "status": "success",
                "total_pages": len(extracted_pages),
                "pages": extracted_pages
            }

    except Exception as e:
        log(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}

# ===============================
# PRELOAD: START vLLM SERVER
# ===============================
log("Booting vLLM server...")
if not start_vllm_server():
    log("FATAL: vLLM server failed to start!")
    raise RuntimeError("vLLM server failed to start")

log("Running dummy warmup...")
try:
    dummy_image = Image.new('RGB', (1600, 1200), color='white')
    _ = ocr_page(dummy_image)
    log("Warmup complete!")
except Exception as e:
    log(f"Warmup error (non-fatal): {e}")

runpod.serverless.start({"handler": handler})