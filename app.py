import os
import re
import sys
import time
import json
import logging
import tempfile
import subprocess
import threading
from queue import Queue

from flask import Flask, request, jsonify

import requests

# =========================
# CONFIG & LOGGING
# =========================

# Test config

logging.basicConfig(
    format='[%(asctime)s][%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

GOOGLE_FORM_SECRET = os.environ.get("GOOGLE_FORM_SECRET", "your-secret")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", AIPIPE_TOKEN)  # Use OpenAI key if set, else fallback to AIPIPE

MAX_CHAIN_SECONDS = 180

app = Flask(__name__)
task_queue = Queue(maxsize=16) 


# =========================
# DIRECT LLM SOLVER (Render page, pass to LLM, get answer)
# =========================

def solve_quiz_with_llm(quiz_url, email, secret, error_context=None):
    """
    Render quiz page with Selenium, download data files, and ask LLM to solve directly.
    Returns: {'submit_url': '...', 'answer': ...}
    error_context: Optional error reason from previous incorrect attempt
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    
    driver = None
    try:
        # Step 1: Render page with Selenium
        logger.info(f"Rendering quiz page with Selenium: {quiz_url}")
        options = Options()
        options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        
        driver = webdriver.Chrome(options=options)
        driver.get(quiz_url)
        
        # Wait for page load and JS execution
        WebDriverWait(driver, 20).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(4)  # Extra time for JS/base64 decoding
        
        # Get rendered HTML (base64 content is now decoded)
        rendered_html = driver.page_source
        soup = BeautifulSoup(rendered_html, 'html.parser')
        
        # Step 2: Extract data file URLs (including audio)
        data_files = []
        audio_files = []
        
        # Debug: log all links found
        all_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(quiz_url, href)
            all_links.append(f"{a.get_text(strip=True)}: {full_url}")
            
            if any(ext in href.lower() for ext in ['.pdf', '.csv', '.json', '.txt']):
                data_files.append(full_url)
            elif any(ext in href.lower() for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.opus']):
                audio_files.append(full_url)
        
        logger.info(f"All links found: {all_links}")
        
        # Also check for audio tags
        audio_tags_found = soup.find_all('audio')
        logger.info(f"Audio tags found: {len(audio_tags_found)}")
        for audio_tag in audio_tags_found:
            logger.info(f"Audio tag: {audio_tag}")
            src = audio_tag.get('src')
            if src:
                audio_files.append(urljoin(quiz_url, src))
        
        logger.info(f"Found {len(data_files)} data files and {len(audio_files)} audio files to download")
        
        # Step 3: Download and process data files
        file_contents = {}
        MAX_PREVIEW_SIZE = 2 * 1024 * 1024  # 2MB limit for preview

        for file_url in data_files:
            try:
                # Stream download to check size without loading full file
                content_accumulated = b""
                truncated = False
                
                with requests.get(file_url, stream=True, timeout=30) as r:
                    if r.status_code == 200:
                        for chunk in r.iter_content(chunk_size=8192):
                            content_accumulated += chunk
                            if len(content_accumulated) > MAX_PREVIEW_SIZE:
                                truncated = True
                                break
                        
                        if truncated:
                            file_contents[file_url] = f"[File too large (> {MAX_PREVIEW_SIZE} bytes) - Use Option 3 to process locally]"
                            continue
                        
                        # Process small files
                        if file_url.lower().endswith('.pdf'):
                            # Extract PDF text
                            from PyPDF2 import PdfReader
                            import io
                            try:
                                reader = PdfReader(io.BytesIO(content_accumulated))
                                text = ""
                                for i, page in enumerate(reader.pages):
                                    text += f"\n--- Page {i+1} ---\n"
                                    text += page.extract_text() or ""
                                file_contents[file_url] = text
                            except Exception as pdf_err:
                                file_contents[file_url] = f"[PDF parsing failed: {pdf_err}]"
                        else:
                            # Assume text/csv
                            file_contents[file_url] = content_accumulated.decode('utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Failed to download {file_url}: {e}")
        
        # Step 3.5: Download and transcribe audio files
        audio_transcriptions = {}
        for audio_url in audio_files:
            try:
                logger.info(f"Downloading audio file: {audio_url}")
                audio_resp = requests.get(audio_url, timeout=60)
                if audio_resp.status_code == 200:
                    # Transcribe using hybrid approach (API first, local fallback)
                    logger.info(f"Transcribing audio file...")
                    
                    # Determine file extension from URL
                    import os
                    from urllib.parse import urlparse
                    parsed_url = urlparse(audio_url)
                    file_ext = os.path.splitext(parsed_url.path)[1] or '.mp3'
                    
                    # Save audio to temp file with correct extension
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_audio:
                        tmp_audio.write(audio_resp.content)
                        tmp_audio_path = tmp_audio.name
                    
                    transcription = None
                    
                    # Use gpt-4o-audio-preview via AIPipe (supports audio input in JSON)
                    try:
                        # Use faster-whisper for local transcription
                        logger.info("Transcribing audio locally with faster-whisper...")
                        
                        from faster_whisper import WhisperModel
                        
                        # Initialize model (downloaded on first run)
                        # Use 'base' or 'small' for speed. 'int8' is faster on CPU.
                        model_size = "base"
                        model = WhisperModel(model_size, device="cpu", compute_type="int8")
                        
                        segments, info = model.transcribe(tmp_audio_path, beam_size=5)
                        
                        logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")
                        
                        transcription = ""
                        for segment in segments:
                            transcription += segment.text + " "
                        
                        transcription = transcription.strip()
                        logger.info(f"Audio transcribed locally")
                        logger.info(f"Transcription: {transcription[:200]}...")
                        
                        if transcription:
                            audio_transcriptions[audio_url] = transcription
                        
                    except Exception as transcription_error:
                        logger.error(f"Audio transcription failed: {transcription_error}")
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(tmp_audio_path):
                        os.unlink(tmp_audio_path)
                        
            except Exception as e:
                logger.error(f"Failed to process audio {audio_url}: {e}")
        
        # Step 4: Prepare context for LLM
        page_text = soup.get_text(separator='\n', strip=True)
        
        # Extract and parse hyperlinks with their content
        hyperlinks = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            full_url = urljoin(quiz_url, href)
            
            # Try to fetch and preview the link content
            link_preview = ""
            try:
                logger.info(f"Fetching hyperlink: {full_url}")
                # Quick HEAD request to check content type
                head_resp = requests.head(full_url, timeout=5, allow_redirects=True)
                content_type = head_resp.headers.get('Content-Type', '').lower()
                logger.info(f"Content-Type: {content_type}")
                
                if 'text/html' in content_type:
                    # It's an HTML page - fetch and preview
                    resp = requests.get(full_url, timeout=10)
                    if resp.status_code == 200:
                        link_soup = BeautifulSoup(resp.text, 'html.parser')
                        link_text = link_soup.get_text(separator=' ', strip=True)
                        # Show first 300 chars of content
                        preview_text = link_text[:300] if len(link_text) > 300 else link_text
                        logger.info(f"Preview text length: {len(preview_text)}, content: {preview_text[:100]}")
                        link_preview = f"\n    Content preview: {preview_text}"
                elif 'application/json' in content_type:
                    link_preview = " [JSON endpoint]"
                elif 'pdf' in content_type:
                    link_preview = " [PDF file]"
                elif 'csv' in content_type or 'text/csv' in content_type:
                    link_preview = " [CSV file]"
                else:
                    link_preview = f" [{content_type}]"
            except Exception as e:
                # If we can't fetch, just note it
                logger.warning(f"Failed to fetch hyperlink {full_url}: {e}")
                link_preview = " [could not preview]"
            
            hyperlinks.append(f"  - {text}: {full_url}{link_preview}")
        
        hyperlinks_section = ""
        if hyperlinks:
            hyperlinks_section = "\n\nHYPERLINKS FOUND ON PAGE:\n" + "\n".join(hyperlinks)
        
        # Build file contents summary
        files_summary = ""
        for url, content in file_contents.items():
            files_summary += f"\n\n=== FILE: {url} ===\n{content[:5000]}\n"  # Limit to 5000 chars per file
        
        # Add audio transcriptions
        if audio_transcriptions:
            files_summary += "\n\n=== AUDIO TRANSCRIPTIONS ===\n"
            for url, transcription in audio_transcriptions.items():
                files_summary += f"\n--- Audio from {url} ---\n{transcription}\n"
        
        # Step 5: Ask LLM to solve (hybrid mode)
        error_section = ""
        if error_context:
            error_section = f"""
**PREVIOUS ATTEMPT WAS INCORRECT**:
Your previous answer was wrong. The server responded: "{error_context}"
Please reconsider your approach and provide a corrected answer.

"""
        
        prompt = f"""{error_section}You are solving a quiz. Follow this priority order:

**PRIORITY 1 - TRY DIRECT SOLVING FIRST**:
If the question is trivial or can be answered with the information on the page, solve it directly and return JSON.

**PRIORITY 2 - SCRAPING** (if you need additional data from a webpage):
If you need data from another webpage, use scraping mode.

**PRIORITY 3 - WRITE PYTHON CODE** (ONLY for external data processing):
Use this ONLY when you need to:
- Process external files (CSV, PDF, Audio, images)
- Download and analyze data files
- Read instructions from external sources
- Perform complex calculations on downloaded data

DO NOT use code for simple tasks that can be solved directly.

QUIZ PAGE:
{page_text[:5000]}
{hyperlinks_section}

DATA PREVIEWS (Content & Transcriptions):
{files_summary}

DATA FILES (URLs):
{json.dumps(data_files, indent=2)}

AUDIO FILES (URLs):
{json.dumps(audio_files, indent=2)}

INSTRUCTIONS:

**OPTION 1: Direct Solving** (Use this for simple/trivial questions)
Return JSON:
{{
  "mode": "direct",
  "submit_url": "https://...",
  "answer": <computed answer>,
  "reasoning": "brief explanation"
}}

**OPTION 2: Scraping Needed**
Return JSON:
{{
  "mode": "scrape",
  "scrape_url": "https://...",
  "submit_url": "https://...",
  "reasoning": "need to scrape data from this URL first"
}}

**OPTION 3: Write Python Code** (ONLY for external data processing)
Write a Python script that downloads and processes external data.
The script can either:
   - Compute the answer and return: `{{"submit_url": "...", "answer": ...}}`
   - Submit directly and print the server response (which may contain a next URL to continue)

**IMPORTANT NOTES**:
- Try OPTION 1 first for simple questions
- Use OPTION 3 ONLY when there are actual CSV, PDF, Audio files or external data to process
- If your code submits directly, the system will automatically continue to the next URL if provided in the response

**EXPECTED TASKS**:
- **Scraping**: Extract data from websites (use `selenium` or `requests`).
- **API Sourcing**: Fetch data from APIs (use `requests`).
- **Cleansing**: Clean text, data, or PDF content.
- **Processing**: Transform data, transcribe audio (`faster_whisper`), or process images (resize, crop, compress via `PIL`).
- **Analysis**: Filter, sort, aggregate, reshape, or apply statistical/ML models (`pandas`, `scipy`, `sklearn`).
- **Visualization**: Generate charts/narratives (note: visualization libraries like matplotlib are NOT currently allowed, output text descriptions instead).

**PDF PROCESSING GUIDELINES**:
- Use `PyPDF2` for extracting text from PDFs
- Parse extracted text with string manipulation or pandas to extract table data

CRITICAL: 
- **PYTHON VERSION**: Ensure all code is compatible with Python 3.12
- Use OPTION 3 for ANY CSV, PDF, or Audio processing.
- If writing code, output ONLY the code block (```python ... ```).
- **IMPORTANT**: Always return the data output from the code block as JSON to stdout.
- **IMPORTANT**: Convert all numpy/pandas types to native Python types (e.g., `int(result)`, `float(result)`) before printing JSON. `numpy.int64` is NOT JSON serializable.
- **ANSWER FORMAT REQUIREMENTS**:
  - The answer may be: **boolean**, **number**, **string**, **base64 URI** of a file, or a **JSON object** combining these
  - For file attachments: Encode as base64 data URI (e.g., `data:image/png;base64,iVBORw0KG...`)
  - **PAYLOAD SIZE LIMIT**: Ensure the final JSON payload is under 1MB
  - For visualizations: If generating charts/images, convert to base64 and include in answer
  - Example formats:
    - Boolean: `{{"answer": true}}`
    - Number: `{{"answer": 42}}`
    - String: `{{"answer": "hello"}}`
    - File: `{{"answer": "data:image/png;base64,iVBORw0KGgo..."}}`
    - JSON object: `{{"answer": {{"count": 5, "image": "data:image/png;base64,..."}}}}`
- **ALLOWED LIBRARIES**: You may ONLY use standard Python libraries (os, json, re, math, etc.) and the following pre-installed packages:
  - `requests`
  - `pandas`
  - `beautifulsoup4` (bs4)
  - `PyPDF2`
  - `selenium`
  - `faster_whisper`
  - `flask`
  - `fastapi`
  - `uvicorn`
  - `scipy`
  - `sklearn`
  - `numpy`
  - `PIL` (Pillow)
  DO NOT import any other third-party libraries unless they are dependencies of the above.
"""
        
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "openai/gpt-4o-mini",  # Use better model for direct solving
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # Log the prompt for debugging
        logger.info("=" * 80)
        logger.info("PROMPT BEING SENT TO LLM:")
        logger.info("=" * 80)
        logger.info(prompt)
        logger.info("=" * 80)
        
        logger.info("Asking LLM to solve quiz (hybrid mode)...")
        llm_resp = requests.post(AIPIPE_URL, headers=headers, json=payload, timeout=90)
        llm_resp.raise_for_status()
        llm_data = llm_resp.json()
        
        content = llm_data["choices"][0]["message"]["content"].strip()
        
        # Check for Python code block
        if "```python" in content or "import " in content:
            logger.info("LLM generated Python code. Executing...")
            
            # Retry loop for script execution errors
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                logger.info(f"Script execution attempt {attempt}/{max_retries}")
                
                # Extract code
                code = content
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0].strip()
                
                # Execute script
                script_result = execute_hybrid_script(code, email, secret, quiz_url)
                
                # Check for execution error
                if script_result.get('error'):
                    error_msg = script_result['error']
                    logger.warning(f"Script execution failed (attempt {attempt}): {error_msg}")
                    
                    if attempt < max_retries:
                        # Re-prompt LLM with error to fix the code
                        logger.info("Re-prompting LLM to fix the error...")
                        error_prompt = f"""
The Python script you generated encountered an error:

ERROR:
{error_msg}

FAILING CODE:
```python
{code}
```

ORIGINAL QUIZ:
{page_text[:3000]}

Please fix the code to handle this error. Common issues:
- HTML selectors might be wrong (check the actual page structure)
- Data might be in a different format
- API endpoints might have changed

Generate the FIXED Python code (output ONLY the code block):
"""
                        
                        retry_payload = {
                            "model": "openai/gpt-4o-mini",
                            "messages": [{"role": "user", "content": error_prompt}],
                        }
                        
                        try:
                            retry_resp = requests.post(AIPIPE_URL, headers=headers, json=retry_payload, timeout=90)
                            retry_resp.raise_for_status()
                            retry_data = retry_resp.json()
                            content = retry_data["choices"][0]["message"]["content"].strip()
                            logger.info("LLM provided fixed code, retrying...")
                            continue  # Retry with new code
                        except Exception as e:
                            logger.error(f"Failed to get fixed code from LLM: {e}")
                            break
                    else:
                        # Max retries reached
                        logger.error(f"Script failed after {max_retries} attempts")
                        return {
                            'submit_url': None,
                            'answer': None,
                            'reasoning': f"Script execution failed after {max_retries} attempts: {error_msg}"
                        }
                else:
                    # Success! Process the result
                    if script_result.get('submit_url') and 'answer' in script_result:
                        return {
                            'submit_url': script_result.get('submit_url'),
                            'answer': script_result.get('answer'),
                            'reasoning': f'Solved via generated Python script (attempt {attempt})'
                        }
                    else:
                        # Assume script submitted directly
                        logger.info(f"Script output does not contain submit_url/answer. Assuming direct submission. Output: {script_result}")
                        return {
                            'submit_url': None,
                            'answer': None,
                            'reasoning': f"Script executed successfully (attempt {attempt}). Output: {script_result}",
                            'direct_submission_response': script_result
                        }

        # Remove markdown code fences if present (for JSON)
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        result = json.loads(content)
        mode = result.get('mode', 'direct')
        
        # Check if LLM needs to scrape additional data
        if mode == 'scrape':
            scrape_url = result.get('scrape_url')
            logger.info(f"LLM identified scraping needed: {scrape_url}")
            
            if scrape_url:
                # Fetch the scraping page with Selenium (in case it needs JS rendering)
                scrape_driver = None
                try:
                    logger.info(f"Fetching scraping URL with Selenium: {scrape_url}")
                    
                    scrape_options = Options()
                    scrape_options.add_argument('--headless=new')
                    scrape_options.add_argument('--disable-gpu')
                    scrape_options.add_argument('--no-sandbox')
                    
                    scrape_driver = webdriver.Chrome(options=scrape_options)
                    scrape_driver.get(scrape_url)
                    
                    # Wait for page load
                    WebDriverWait(scrape_driver, 10).until(
                        lambda d: d.execute_script("return document.readyState") == "complete"
                    )
                    time.sleep(2)  # Extra time for JS
                    
                    # Get rendered content
                    scrape_html = scrape_driver.page_source
                    scrape_soup = BeautifulSoup(scrape_html, 'html.parser')
                    scraped_text = scrape_soup.get_text(separator='\n', strip=True)
                    
                    logger.info(f"Scraped content length: {len(scraped_text)} chars")
                    logger.info(f"Scraped content preview: {scraped_text[:200]}")
                    
                    if len(scraped_text) == 0:
                        logger.warning("Scraped content is empty!")
                    
                    # Re-prompt LLM with the scraped content
                    reprompt = f"""
You previously identified that you need to scrape data from: {scrape_url}

Here is the scraped content:

SCRAPED PAGE:
{scraped_text[:5000]}

ORIGINAL QUIZ PAGE:
{page_text[:5000]}

Now compute the answer based on the scraped data.

CRITICAL OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no code fences, no explanations
2. Do NOT include comments (no // or /* */)
3. Use this exact format:

{{"submit_url":"{result.get('submit_url', '')}", "answer":"your_computed_answer", "reasoning":"brief explanation"}}

Replace "your_computed_answer" with the actual answer you computed from the scraped data.
Output ONLY the JSON object, nothing else.
"""
                    
                    logger.info("Re-prompting LLM with scraped content...")
                    reprompt_payload = {
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": reprompt}],
                    }
                    
                    llm_resp2 = requests.post(AIPIPE_URL, headers=headers, json=reprompt_payload, timeout=90)
                    llm_resp2.raise_for_status()
                    llm_data2 = llm_resp2.json()
                    
                    content2 = llm_data2["choices"][0]["message"]["content"].strip()
                    
                    logger.info("=" * 80)
                    logger.info("LLM RESPONSE TO RE-PROMPT:")
                    logger.info("=" * 80)
                    logger.info(content2)
                    logger.info("=" * 80)
                    
                    # Extract JSON from response (minimal cleanup)
                    # Remove markdown code fences if present
                    if "```json" in content2:
                        content2 = content2.split("```json")[1].split("```")[0].strip()
                    elif content2.startswith("```"):
                        lines2 = content2.split("\n")
                        content2 = "\n".join(lines2[1:-1]).strip()
                    
                    # Find JSON object if there's extra text
                    start_idx = content2.find('{')
                    end_idx = content2.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = content2[start_idx:end_idx+1]
                    else:
                        json_str = content2
                    
                    logger.info(f"Extracted JSON: {json_str[:300]}")
                    
                    try:
                        result = json.loads(json_str)
                        logger.info("LLM solved with scraped data")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM response as JSON: {e}")
                        logger.error(f"Full response was: {content2[:500]}")
                        return {"submit_url": None, "answer": None, "reasoning": f"JSON parse error: {e}"}
                    
                except Exception as e:
                    logger.error(f"Failed to scrape or re-prompt: {e}")
                    return {"submit_url": None, "answer": None, "reasoning": f"Scraping failed: {e}"}
                finally:
                    if scrape_driver:
                        try:
                            scrape_driver.quit()
                        except:
                            pass
        else:
            logger.info("LLM solved directly")
        
        # Handle relative submit URLs (e.g., "/submit" -> "https://domain.com/submit")
        submit_url = result.get('submit_url')
        if submit_url and not submit_url.startswith('http'):
            from urllib.parse import urljoin
            submit_url = urljoin(quiz_url, submit_url)
            result['submit_url'] = submit_url
            logger.info(f"Converted relative URL to absolute: {submit_url}")
        
        logger.info(f"Solution: submit_url={result.get('submit_url')}, answer={result.get('answer')}, mode={mode}")
        
        return {
            'submit_url': result.get('submit_url'),
            'answer': result.get('answer'),
            'reasoning': result.get('reasoning', '')
        }
        
    except Exception as e:
        logger.error(f"Failed to solve quiz with LLM: {e}")
        import traceback
        traceback.print_exc()
        return {
            'submit_url': None,
            'answer': None,
            'reasoning': f'Error: {e}'
        }
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


# =========================
# HYBRID SCRIPT EXECUTION
# =========================

def execute_hybrid_script(script_code, email, secret, quiz_url, timeout_seconds=60):
    """
    Execute LLM-generated script with injected variables.
    The script has access to: email, secret, quiz_url, AIPIPE_TOKEN
    """
    try:
        # Inject variables at the top of the script
        injected_script = f"""
import os
import sys
import json

# Injected variables
os.environ['AIPIPE_TOKEN'] = {json.dumps(AIPIPE_TOKEN)}
email = {json.dumps(email)}
secret = {json.dumps(secret)}
quiz_url = {json.dumps(quiz_url)}

# LLM-generated code starts here
{script_code}
"""
        
        logger.info("Executing hybrid script...")
        logger.info(f"Script length: {len(injected_script)} chars")
        
        # Save script for debugging
        with open("generated_solver_temp.py", "w") as f:
            f.write(injected_script)
        
        # Execute script and capture output
        result = subprocess.run(
            [sys.executable, "-c", injected_script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"Script execution failed: {error_msg}")
            return {"error": error_msg, "submit_url": None, "answer": None}
        
        # Parse JSON output from script
        output = result.stdout.strip()
        logger.info(f"Script output: {output[:500]}")
        
        # Try to parse the output directly as JSON first
        try:
            result_json = json.loads(output)
            logger.info("Successfully parsed output as JSON")
            return result_json
        except json.JSONDecodeError:
            logger.info("Output is not pure JSON, looking for JSON pattern...")
        
        # Find JSON in output (might have other print statements)
        # Look for the last JSON object in the output
        json_match = None
        json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output))
        if json_matches:
            # Try the last match first (most likely the final result)
            json_match = json_matches[-1]
            matched_str = json_match.group(0)
            
            # Try JSON first
            try:
                result_json = json.loads(matched_str)
                logger.info("Successfully parsed JSON from pattern match")
                return result_json
            except json.JSONDecodeError:
                # Fallback: try ast.literal_eval for Python dict format
                import ast
                try:
                    result_json = ast.literal_eval(matched_str)
                    if isinstance(result_json, dict):
                        logger.info("Successfully parsed output using ast.literal_eval")
                        return result_json
                except Exception as e:
                    logger.warning(f"ast.literal_eval also failed: {e}")
        
        logger.error(f"No valid JSON found in script output: {output[:200]}")
        return {"submit_url": None, "answer": None}
            
    except subprocess.TimeoutExpired:
        logger.error(f"Script execution timed out after {timeout_seconds}s")
        return {"submit_url": None, "answer": None}
    except Exception as e:
        logger.error(f"Script execution error: {e}")
        import traceback
        traceback.print_exc()
        return {"submit_url": None, "answer": None}


# =========================
# LLM: GENERATE SOLVER SCRIPT (DEPRECATED - keeping for reference)
# =========================

def generate_solver_script(email, secret, quiz_url, question=None, submit_url=None, attachments=None, max_retries=3):
    """
    DEPRECATED: This function generates a Python script for solving quizzes.
    Now using solve_quiz_with_llm() for direct solving instead.
    """
    
    prompt = f"""
Write a Python 3 script to solve this quiz question.

GIVEN INFORMATION (already extracted):
email = {json.dumps(email)}
secret = {json.dumps(secret)}
quiz_url = {json.dumps(quiz_url)}
submit_url = {json.dumps(submit_url or "UNKNOWN")}

QUESTION:
{question or "No question extracted - check page content"}

DATA FILES TO PROCESS:
{attachments_str or "None"}

YOUR TASK:
1. Download and process the data files listed above
   - PDFs: Use PyPDF2.PdfReader to extract text from each page
   - CSVs: Use pandas.read_csv() to load data
   - Store data for analysis

2. Read the QUESTION carefully and compute the correct answer
   - Pay attention to specific instructions (e.g., "3rd number" = index [2])
   - For column operations: df['column_name'].sum() or .max()
   - For PDF pages: Remember 0-indexing (page 2 = index 1)
   - Filter BEFORE aggregating if needed

3. Output JSON to stdout:
   {{"submit_url": submit_url, "payload": {{"email": email, "secret": secret, "url": quiz_url, "answer": ANSWER}}}}

IMPORTANT:
- If submit_url is "UNKNOWN", try to find it by checking the quiz page or use a fallback
- Return int if answer should be an integer (use int() if needed)
- Print ONLY the JSON output to stdout, log debug info to stderr
- No markdown, no code fences, just valid Python code

Required imports:
import sys, json, io, requests
from PyPDF2 import PdfReader
import pandas as pd

Output ONLY valid Python code.
"""
    
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-5-nano",
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Calling AIPipe to generate solver script (attempt {attempt})")
            resp = requests.post(AIPIPE_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            code = data["choices"][0]["message"]["content"]
            for marker in ("``````", "`"):
                code = code.replace(marker, "")
            code = code.strip()
            logger.info(f"Generated solver script, length={len(code)} chars")
            return code
        except Exception as e:
            logger.error(f"AIPipe error: {e}")
            if attempt == max_retries:
                raise
            time.sleep(3)


# =========================
# RUN GENERATED SCRIPT (MODIFIED: NO DELETION)
# =========================

def run_solver_script(code, timeout_seconds):
    """
    Run generated code as a script using a fixed local file.
    The file is NOT deleted after execution for debugging.
    Expect final stdout line to be JSON: {"submit_url": "...", "payload": {...}}
    """
    # Use a fixed file path in the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "generated_solver_temp.py")
    
    try:
        # Write the code to the fixed local file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Saved generated solver script to {script_path}")
    except Exception as e:
        logger.error(f"Failed to save generated script to {script_path}: {e}")
        return None

    logger.info(f"Executing solver script {script_path} with timeout={timeout_seconds}s")
    start = time.time()
    try:
        proc = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.time() - start
        logger.info(f"Solver exited code={proc.returncode} in {elapsed:.1f}s")
        if proc.stderr:
            logger.debug(f"Solver STDERR:\n{proc.stderr}")

        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            logger.warning("Solver produced no stdout lines")
            return None

        last = lines[-1]
        try:
            result = json.loads(last)
            if "submit_url" in result and "payload" in result:
                return result
            logger.warning("Solver JSON missing required keys")
            return None
        except json.JSONDecodeError:
            logger.warning("Last stdout line is not valid JSON")
            return None
    except subprocess.TimeoutExpired:
        logger.error("Solver script timed out")
        return None
    finally:
        # --- MODIFICATION: File deletion is REMOVED ---
        # The script file at script_path will remain for debugging.
        pass


# =========================
# QUIZ CHAIN LOGIC (Executed by worker)
# =========================

def solve_quiz_chain(email, secret, start_url):
    """
    Background chain logic to solve quizzes sequentially using direct LLM solving.
    """
    logger.info(f"Starting quiz chain for {email} at {start_url}")
    start_time = time.time()
    current_url = start_url
    results = []

    while current_url and (time.time() - start_time) < MAX_CHAIN_SECONDS:
        logger.info(f"Solving quiz at {current_url}")
        chain_elapsed = time.time() - start_time
        time_left = MAX_CHAIN_SECONDS - chain_elapsed
        if time_left <= 15:
            logger.warning("Not enough time remaining for another full solve")
            break

        try:
            # Use direct LLM solving (renders page, downloads files, computes answer)
            logger.info("Solving quiz with LLM...")
            solution = solve_quiz_with_llm(current_url, email, secret)
            
            if not solution or (not solution.get('submit_url') and not solution.get('direct_submission_response')):
                logger.error("LLM failed to find submit URL and no direct submission occurred")
                results.append({"url": current_url, "error": "no_submit_url_found"})
                break
            
            submit_url = solution['submit_url']
            answer = solution['answer']
            
            logger.info(f"LLM solution: submit_url={submit_url}, answer={answer}")
            
        except Exception as e:
            logger.error(f"Quiz solving failed: {e}")
            results.append({"url": current_url, "error": f"solving_failed: {e}"})
            break


        # Step 3: Submit answer and retry on incorrect responses
        max_answer_retries = 3
        for retry_attempt in range(1, max_answer_retries + 1):
            logger.info(f"Answer submission attempt {retry_attempt}/{max_answer_retries}")
            
            submit_url = solution.get('submit_url')
            answer = solution.get('answer')
            direct_response = solution.get('direct_submission_response')
            
            if direct_response:
                logger.info("Using direct submission response from script.")
                data = direct_response
            elif submit_url and answer is not None:
                logger.info(f"Submitting answer to {submit_url}: {answer}")
                try:
                    payload = {
                        "email": email,
                        "secret": secret,
                        "url": current_url,
                        "answer": answer
                    }
                    resp = requests.post(submit_url, json=payload, timeout=15)
                    resp.raise_for_status()
                    
                    # Check if response is empty or invalid
                    if not resp.text or not resp.text.strip():
                        logger.error("Submission returned empty response - cannot continue")
                        results.append({"url": current_url, "error": "empty_response_from_server"})
                        # Break from both retry loop and outer while loop
                        current_url = None
                        break
                    
                    try:
                        data = resp.json()
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Submission returned invalid JSON: {json_err}")
                        results.append({"url": current_url, "error": f"invalid_json_response: {json_err}"})
                        # Break from both retry loop and outer while loop
                        current_url = None
                        break
                except requests.exceptions.Timeout:
                    logger.warning(f"Submission timed out after 15 seconds (attempt {retry_attempt}/{max_answer_retries})")
                    if retry_attempt < max_answer_retries:
                        logger.info("Retrying after timeout...")
                        continue  # Retry with same solution
                    else:
                        logger.error(f"Submission timed out after {max_answer_retries} attempts")
                        results.append({"url": current_url, "error": "submission_timeout"})
                        break
                except Exception as e:
                    logger.error(f"Submission failed: {e}")
                    results.append({"url": current_url, "error": f"submission_failed: {e}"})
                    break
            else:
                logger.error("LLM failed to find submit URL or answer, and no direct submission detected.")
                results.append({"url": current_url, "error": "no_submit_url_found"})
                break

            correct = data.get("correct")
            next_url = data.get("url")
            reason = data.get("reason", "")
            
            # Normalize correct to boolean (handle both string and boolean)
            if isinstance(correct, str):
                correct = correct.lower() in ("true", "1", "yes")
            
            # Check if answer is correct
            if correct is True:
                logger.info(f"Answer is correct on attempt {retry_attempt}")
                results.append({
                    "url": current_url,
                    "answer": answer,
                    "correct": correct,
                    "reason": reason,
                    "reasoning": solution.get('reasoning', ''),
                    "attempts": retry_attempt
                })
                
                # Check if next_url is different from current_url
                if next_url and next_url != current_url:
                    current_url = next_url
                else:
                    if next_url == current_url:
                        logger.info("Quiz complete - next URL is same as current URL.")
                    else:
                        logger.info("Quiz chain complete (no next URL).")
                    current_url = None  # Signal outer loop to stop
                break  # Exit retry loop
            elif correct is False:
                logger.warning(f"Answer is incorrect on attempt {retry_attempt}. Reason: {reason}")
                
                if retry_attempt < max_answer_retries:
                    # Re-prompt LLM with error feedback
                    logger.info("Re-prompting LLM with error feedback...")
                    try:
                        retry_solution = solve_quiz_with_llm(current_url, email, secret, error_context=reason)
                        if retry_solution:
                            solution = retry_solution
                            continue  # Retry with new solution
                    except Exception as e:
                        logger.error(f"Failed to get retry solution: {e}")
                        break
                else:
                    # Max retries reached
                    logger.error(f"Answer still incorrect after {max_answer_retries} attempts")
                    results.append({
                        "url": current_url,
                        "answer": answer,
                        "correct": False,
                        "reason": reason,
                        "reasoning": solution.get('reasoning', ''),
                        "attempts": max_answer_retries
                    })
                    
                    # Continue to next URL if available
                    if next_url:
                        logger.info(f"Moving to next quiz despite incorrect answer: {next_url}")
                        current_url = next_url
                    else:
                        logger.info("No next URL available. Quiz chain complete.")
                    break  # Exit retry loop
            else:
                # correct is None - response doesn't have 'correct' field
                logger.error(f"Response missing 'correct' field. Data: {data}")
                results.append({
                    "url": current_url,
                    "error": "invalid_response_format",
                    "data": data
                })
                break  # Exit both loops
        else:
            # Loop completed without break (shouldn't happen, but handle it)
            logger.info("Quiz chain complete or dead end.")
            break
    
    logger.info(f"Quiz chain completed for {email}. Processed {len(results)} quiz question(s). Final results: {results}")


# =========================
# BACKGROUND WORKERS
# =========================

def worker():
    """Background worker that processes tasks from queue."""
    while True:
        req = task_queue.get()
        try:
            logger.info(f"Background worker picked task for email={req.get('email')}")
            solve_quiz_chain(
                email=req["email"],
                secret=req["secret"],
                start_url=req["url"],
            )
        except Exception as e:
            logger.error(f"Background worker error: {e}", exc_info=True)
        finally:
            task_queue.task_done()

# Start background workers (e.g., 2 workers)
for _ in range(2):
    t = threading.Thread(target=worker, daemon=True)
    t.start()


# =========================
# FLASK ENDPOINTS (ASYNC RESPONSE)
# =========================

@app.route("/quiz", methods=["POST"])
def handle_quiz():
    """
    Accept quiz request and queue it for background processing.
    Returns immediately with acknowledgment (200).
    """
    logger.info("API /quiz called.")
    try:
        req = request.get_json()
        if not req:
            return jsonify(error="Invalid JSON"), 400
    except Exception:
        return jsonify(error="Invalid JSON"), 400

    if req.get("secret") != GOOGLE_FORM_SECRET:
        logger.warning("Invalid secret provided.")
        return jsonify(error="Invalid secret"), 403

    required = ("email", "url", "secret")
    if any(k not in req for k in required):
        return jsonify(error="Missing required fields"), 400

    try:
        # Put task onto the queue immediately
        task_queue.put(req, block=False)
        logger.info(f"Request acknowledged and queued for email={req['email']}.")
        
        # Immediate response: 200 Acknowledged
        return jsonify(
            status="acknowledged",
            message="Quiz solving started in background worker."
        ), 200
    except Exception:
        logger.error("Task queue is full; cannot accept more requests right now.")
        return jsonify(error="Server busy"), 503


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "AIPipe LLM-driven Quiz Solver (async)",
        "status": "running"
    }), 200


if __name__ == "__main__":
    logger.info("Starting Flask server on port 7860...")
    app.run(host="0.0.0.0", port=7860)
