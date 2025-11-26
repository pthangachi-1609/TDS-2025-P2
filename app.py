from github.GithubException import GithubException
import os
import time
import threading
import logging
from flask import Flask, request, jsonify
from github import Github, UnknownObjectException
import requests
import json
from google import genai
import hashlib
from queue import Queue


# Set up logging
logging.basicConfig(
    format='[%(asctime)s][%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

GOOGLE_FORM_SECRET = os.getenv("GOOGLE_FORM_SECRET")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USER = os.getenv("GITHUB_USER")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
# PIPE = "GEMINI"
PIPE = "OPENAI"

app = Flask(__name__)
gh = Github(GITHUB_TOKEN)


task_queue = Queue(maxsize=16)

client = genai.Client()
chat = client.chats.create(model="gemini-2.5-flash")


def worker():
    while True:
        req = task_queue.get()
        try:
            # Your existing function (already handles retries)
            process_request(req)
        except Exception as e:
            logger.error(f"Background worker error: {e}")
        finally:
            task_queue.task_done()


for _ in range(2):  # Tune this number for your quota/environment
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()


def upsert_github_file(repo, path, content, commit_msg, branch="main"):
    """
    Upserts a file to a specific path in a GitHub repository on a given branch.

    Args:
        repo: PyGithub Repository object.
        path (str): The full path to the file in the repo.
        content (str): The content to write to the file.
        commit_msg (str): The commit message.
        branch (str): The name of the branch to commit to. Defaults to "main".
    """
    try:
        # Get the file to see if it exists on the specified branch
        file = repo.get_contents(path, ref=branch)

        # If it exists, update it
        result = repo.update_file(
            path=file.path,
            message=commit_msg,
            content=content,
            sha=file.sha,
            branch=branch
        )
        print(f"Updated '{path}' on branch '{branch}'.")
        return result

    except GithubException as e:
        if e.status == 404:
            # If the file does not exist, create it
            result = repo.create_file(
                path=path,
                message=commit_msg,
                content=content,
                branch=branch
            )
            print(f"Created '{path}' on branch '{branch}'.")
            return result

        else:
            # Re-raise other exceptions
            print(f"Encountered an unexpected error: {e}")
            return None


def get_repo_name_from_task(task):
    """Creates a stable and predictable repository name from a task ID."""
    # Hash ONLY the stable task_id to get a unique fingerprint
    sha1 = hashlib.sha1(task.encode('utf-8')).hexdigest()
    short_hash = sha1[:8]

    return f"{task}-{short_hash}"


def llm_generate_file(prompt):
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if PIPE == "GEMINI":
                logger.info(f"Calling LLM for file generation with {PIPE}")
                response = chat.send_message(prompt)
                output = response.text
                return output
            else:
                logger.info(
                    f"Calling LLM for file generation... [Attempt {attempt}]")
                headers = {
                    "Authorization": f"Bearer {AIPIPE_TOKEN}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "openai/gpt-5-nano",
                    "messages": [{"role": "user", "content": prompt}]
                }

                resp = requests.post(
                    AIPIPE_URL, headers=headers, json=payload, timeout=120
                )
                resp.raise_for_status()  # Raise exception for bad status codes
                logger.info(
                    f"Received response from LLM. Status: {resp.status_code}")
                logger.debug(f"Raw response: {resp.text[:200]}...")

                response_json = resp.json()
                output = response_json["choices"][0]["message"]["content"]
                return output

        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            if 'resp' in locals():
                logger.error(f"Response text: {resp.text[:500]}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")

        if attempt < max_attempts:
            logger.warning(
                f"LLM API failed, retrying in 5 seconds... (Attempt {attempt+1} of {max_attempts})")
            time.sleep(5)  # Wait before retrying
        else:
            logger.error("LLM API failed after maximum retries.")
            raise Exception("LLM API failed after 3 attempts.")


def llm_generate_file2(prompt):
    if PIPE == "GEMINI":
        logger.info(f"Calling LLM for file generation with {PIPE}")
        response = chat.send_message(prompt)
        output = response.text
    else:
        logger.info("Calling LLM for file generation...")
        headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}",
                   "Content-Type": "application/json"}
        payload = {"model": "openai/gpt-5-nano",
                   "messages": [{"role": "user", "content": prompt}]}

        try:
            resp = requests.post(AIPIPE_URL, headers=headers,
                                 json=payload, timeout=120)
            resp.raise_for_status()  # Raise exception for bad status codes
            logger.info(
                f"Received response from LLM. Status: {resp.status_code}")

            # Log the raw response for debugging
            logger.debug(f"Raw response: {resp.text[:200]}...")

            response_json = resp.json()
            output = response_json["choices"][0]["message"]["content"]

        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response text: {resp.text[:500]}")
            raise Exception(
                f"Invalid JSON response from LLM API: {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response structure: {e}")
            logger.error(f"Response JSON: {response_json}")
            raise Exception(f"Unexpected API response format: missing {e}")

    return output


def generate_code(brief, app_code, attachments=None, round_num=1, checks=None, output_dir="output"):
    """
    Constructs a prompt for generating a Flask app that can run as a server
    OR export static files for GitHub Pages deployment.
    """
    attachments = attachments or []
    checks = checks or []

    checks_section = "\n".join(
        f"- {chk}" for chk in checks) or "- None specified."
    att_preview = "\n".join(
        f"- {att['name']} (Preview: {att['url'][:64]}...)"
        for att in attachments
    ) or "- None"

    sample_data_json = (
        '{\n'
        '  "attachments": [\n'
        '    { "name": "sample.png", "url": "data:image/png;base64,iVBORw..." }\n'
        '  ]\n'
        '}'
    )

    # prompt = (
    #     "TASK: Build an app (app.py) based on the brief and functional checks below.\n"
    #     "By default, generate a minimal and generic Python app.\n"
    #     "If ANY requirement, check, or brief context specifically mentions/requires 'Flask', the app MUST use Flask and follow all Flask usage conventions (setup, routing, server, context, etc.).\n"
    #     "App must visually and functionally demonstrate every condition in the brief/checks in its static export (if required).\n"

    #     "--BRIEF--\n"
    #     f"{brief}\n\n"

    #     "--- FUNCTIONAL REQUIREMENTS ---\n"
    #     f"{checks_section}\n"
    #     "Exclude any requirements/checks related to README or repository setup.\n"
    #     "Ensure all imports required for the app are included.\n\n"

    #     "--- ATTACHMENT HANDLING ---\n"
    #     "At runtime, app must load attachments from repo root data.json, which has this format:\n"
    #     f"{sample_data_json}\n"
    #     f"Attachments to process:\n{att_preview}\n\n"

    #     "--- DUAL MODE OPERATION ---\n"
    #     "Your app must support two modes:\n"
    #     "1. Development: Run 'python app.py' to start a Flask server on any standard port, serving pages dynamically.\n"
    #     f"2. Static Export: Run 'python app.py --export' to generate all pages as static HTML files in '{output_dir}/', including assets and decoded files.\n"
    #     "    - Export all required assets (CSS, JS, images, decoded attachments) so links work offline.\n"
    #     "    - Establish an application context to run the function if flask is the backend\n"
    #     "    - No user interaction is required; script completes after export.\n\n"

    #     "--- SPECIFIC INSTRUCTIONS ---\n"
    #     "1. Parse data.json to extract and decode all Data URIs.\n"
    #     "2. Use attachments as required by BRIEF context.\n"
    #     "3. Do NOT hardcode example data; always load from data.json.\n"
    #     "4. In export mode, save all decoded binary/static files (images, etc.) to output_dir.\n"
    #     "5. Ensure all relative links and references work in the exported static site.\n"
    #     "6. Export CSS either inline or as separate file to output_dir, as needed for proper rendering.\n"
    #     "7. All exported files must reside in output_dir and be referenced correctly from HTML pages.\n"
    #     "8. The exported site must be fully self-contained and ready for GitHub Pages.\n\n"
    #     "9. Ensure rendering github pages strictly follows FUNCTIONAL REQUIREMENTS around pages, layout, content, and features.\n"
    #     "10. When writing any HTML, CSS, or JavaScript strings that use .format() in Python, always use double curly braces ({{ }}) for all curly braces that do not represent format placeholders. This is required for correct string formatting and to avoid runtime KeyError or ValueError.\n"
    #     "11. **Important** : Ensure rendering github pages strictly follows FUNCTIONAL REQUIREMENTS around pages, layout, content, and features. Also, no redirects or embedded links to access app from github pages are allowed.\n"

    #     "--- OUTPUT REQUIREMENT ---\n"
    #     "Output ONLY the code for app.py. Do NOT use markdown, code fences, sample input/output, or external explanations."
    # )
    # if round_num == 1:
    #     prompt = (
    #         "TASK: Build app.py based on the BRIEF and FUNCTIONAL REQUIREMENTS below.\n"
    #         "By default, produce a minimal, generic Python app.\n"
    #         "If ANY requirement, check, or brief context explicitly mentions or requires 'Flask', you MUST use Flask and rigorously follow Flask conventions (setup, routing, context, dynamic server, etc.).\n"
    #         "The exported static site must visually and functionally demonstrate EVERY condition in the brief and functional checks—no hidden logic. All outputs must be directly observable.\n\n"

    #         "--BRIEF--\n"
    #         f"{brief}\n\n"

    #         "--- FUNCTIONAL REQUIREMENTS (ALL MUST BE REFLECTED IN OUTPUT) ---\n"
    #         f"{checks_section}\n"
    #         "Exclude any README or repository setup requirements.\n"
    #         "Include all required imports for the app’s code.\n\n"

    #         "--- ATTACHMENT HANDLING ---\n"
    #         "At runtime, load attachments from data.json in the repo root, using this format:\n"
    #         f"{sample_data_json}\n"
    #         f"Attachments to process:\n{att_preview}\n\n"

    #         "--- MODES OF OPERATION ---\n"
    #         "Support BOTH modes:\n"
    #         "1. Development: 'python app.py' starts a Flask (if required) server on any standard port, serving pages dynamically.\n"
    #         f"2. Static Export: 'python app.py --export' generates all site pages, assets, decoded attachment files, CSS/JS/images, etc., into '{output_dir}/' so links work offline. No user interaction—script exits after export.\n"
    #         "    - If Flask is used, correctly handle application context during export.\n\n"

    #         "--- SPECIFIC INSTRUCTIONS ---\n"
    #         "1. Parse data.json to extract and decode all Data URIs. Do NOT hardcode attachments or sample data.\n"
    #         "2. Use attachments exactly as required by the BRIEF and functional checks.\n"
    #         "3. In export mode, save ALL binary/static files to output_dir. Ensure HTML, images, scripts, and CSS are properly linked for the static site.\n"
    #         "4. Export CSS either inline or as separate file to output_dir as needed for correct rendering.\n"
    #         "5. All exported files (HTML, CSS, attachments) must reside in output_dir and be referenced correctly in the HTML.\n"
    #         "6. Exported site must be fully self-contained, directly validating every functional requirement—no redirects, no external links required for complete inspection.\n"
    #         "7. All code must use standard Python 3 indentation (4 spaces per level, no tabs), be free of indentation or syntax errors, and be ready to run without modification.\n\n"
    #         "8. **Important:** Static GitHub Pages output must strictly follow FUNCTIONAL REQUIREMENTS for layout, content, pages, and features.\n"
    #         "9. When generating any Python string containing HTML, CSS, or JS for use with .format(), always escape literal curly braces with double braces ({{ }}) except for variable placeholders. This prevents KeyError and ensures correct formatting.\n"
    #         "10. Never include links or redirects that require navigation outside the exported static site to access app features in GitHub Pages.\n"
    #         "11. All code must use standard Python 3 indentation (4 spaces per level, no tabs), be free of indentation or syntax errors, and be ready to run without modification.\n\n"

    #         "--- OUTPUT REQUIREMENT ---\n"
    #         "Output ONLY the complete code for app.py. No markdown, code fences, sample input/output, comments outside code, or explanations."
    #     )
    # else:
    #     prompt = (
    #         "TASK: Update ONLY the feature logic in app.py below according to the revised brief and functional checks for round 2.\n"
    #         "Keep all other code, structure, attachments/data.json handling, export/static/dual-mode routines, and file naming exactly as-is.\n"
    #         "You may ONLY change or add code necessary to fulfill the new logic/routes/output as described.\n"
    #         "Do NOT modify headers, comments, initialization, dual-mode switches, export routines, or static file patterns except as required by the new brief and checks.\n"
    #         "If using Flask, keep all existing routing/app setup and modify only route logic as needed.\n"
    #         "Preserve all import statements, file IO, and static export patterns unless feature logic demands otherwise.\n"

    #         "--ROUND 2 BRIEF--\n"
    #         f"{brief}\n\n"

    #         "--- UPDATED FUNCTIONAL REQUIREMENTS ---\n"
    #         f"{checks_section}\n"
    #         "All new requirements must be reflected in app logic, view output, or site content as relevant.\n"
    #         "Preserve dependencies, attachment decoding, and dual-mode support from the previous version.\n\n"
    #         "All code must use standard Python 3 indentation (4 spaces per level, no tabs), be free of indentation or syntax errors, and be ready to run without modification.\n\n"

    #         "--PREVIOUS APP.PY CODE (START)--\n"
    #         f"{app_code}\n"
    #         "--PREVIOUS APP.PY CODE (END)--\n\n"

    #         "--- OUTPUT REQUIREMENT ---\n"
    #         "Output ONLY the new, full code for app.py (with the core logic updated for round 2 as above, everything else untouched). Do NOT include markdown, code fences, or explanations."
    #     )

    if round_num == 1:
        prompt = (
            "TASK: Build app.py based on the BRIEF and FUNCTIONAL REQUIREMENTS below.\n"
            "By default, create a minimal, generic Python app.\n"
            "If ANY requirement, check, or brief context explicitly mentions or requires 'Flask', rigorously use Flask with correct conventions (setup, routing, context, dynamic server, etc.).\n"
            "Exported static site must visually and functionally demonstrate EVERY condition in the brief and functional checks—no hidden logic, everything directly observable for inspection.\n"
            "All code must be syntactically and indentation-correct Python 3 ready-to-run code (4 spaces per level, no tabs).\n\n"
            "--BRIEF--\n"
            f"{brief}\n\n"
            "--- FUNCTIONAL REQUIREMENTS (ALL MUST BE REFLECTED IN OUTPUT) ---\n"
            f"{checks_section}\n"
            "Exclude README/repo setup requirements.\n"
            "Include all imports required for app functionality.\n\n"
            "--- ATTACHMENT HANDLING ---\n"
            "At runtime, load attachments from data.json in repo root, formatted as:\n"
            f"{sample_data_json}\n"
            f"Attachments to process:\n{att_preview}\n\n"
            "--- MODES OF OPERATION ---\n"
            "Support BOTH modes:\n"
            "1. Development: 'python app.py' starts the Flask (if required) server on any standard port for dynamic serving.\n"
            f"2. Static Export: 'python app.py --export' generates all site pages, decoded attachments, CSS/JS/images, etc. in '{output_dir}/' for offline viewing. Script completes without user interaction.\n"
            "    - If Flask is used, always render templates/routes via Flask (not raw Jinja) before saving HTML to output_dir in export mode.\n"
            "    - Ensure exported HTML only contains evaluated content—no Jinja tags visible to users.\n\n"
            "--- SPECIFIC INSTRUCTIONS ---\n"
            "1. Parse data.json; decode all Data URIs at runtime. Do NOT hardcode attachments/sample data.\n"
            "2. Use attachments exactly as required in brief and functional checks.\n"
            "3. In export mode, save ALL output/binary/static files to output_dir and properly link/image/reference in HTML.\n"
            "4. Export CSS inline or as a separate file in output_dir for proper rendering.\n"
            "5. All exported files must reside in output_dir and be referenced correctly from HTML/pages.\n"
            "6. Static GitHub Pages export MUST be fully self-contained and directly validate every functional requirement (no external links, redirects, or missing data).\n"
            "7. Always use standard Python 3 indentation (4 spaces per level) and guarantee no syntax/indentation errors.\n"
            "8. When writing Python strings containing HTML, CSS, or JS for .format(), always escape literal curly braces as double braces ({{ and }}), except for format placeholders.\n"
            "9. Never include links or redirects outside exported static site for feature access or inspection in GitHub Pages.\n"
            "10. Strictly render all template logic before export—do NOT save raw templates with Jinja tags to output_dir.\n"
            "11. Do NOT display the words 'static export', 'exported', '--export', 'development mode', or other workflow/internal process keywords anywhere on the rendered pages, headings, or user-facing site text.\n"
            "12. All visible content must reflect only the required outputs, features, data, and UX as described in the BRIEF and functional checks. Hide all implementation details from users.\n"
            "13. - During '--export' mode, always wrap all template rendering (render_template, render_template_string) in 'with app.app_context():' \n"
            "14. All Jinja template expressions must use only valid Jinja2 syntax—never put colons (:) inside {{ ... }}. For default values, use {{ var or 'default' }} instead of {{ var:default }}.\n"
            "15. Ensure all template files render without any Jinja TemplateSyntaxError in Flask.\n"
            "16. **Important** : Strictly use the brief/checks to define all visible content, features, and UX.\n\n"
            "--- OUTPUT REQUIREMENT ---\n"
            "Output ONLY the full code for app.py—no markdown, code fences, sample input/output, or explanations. All code must be fully runnable and free of syntax/indentation errors."
        )

    else:
        prompt = (
            "TASK: Update ONLY the feature logic in app.py below based on the revised brief and functional checks for round 2.\n"
            "Keep ALL other code—structure, attachments/data.json handling, export/static/dual-mode routines, and file naming—exactly as-is.\n"
            "Change or add ONLY the code necessary to fulfill the new logic/routes/output per the updated requirements.\n"
            "Do NOT modify headers, comments, initialization, dual-mode switches, export routines, or static file patterns unless mandated by the new brief/checks.\n"
            "If using Flask, retain all existing routing/app setup and update only route logic as required.\n"
            "Preserve all import statements, file IO, and static export mechanisms unless core logic changes require updates.\n"
            "All code must use standard Python 3 indentation (4 spaces per level, no tabs), be free of syntax/indentation errors, and be ready to run without modification.\n\n"
            "--ROUND 2 BRIEF--\n"
            f"{brief}\n\n"
            "--- UPDATED FUNCTIONAL REQUIREMENTS ---\n"
            f"{checks_section}\n"
            "All new requirements must be reflected in app logic, view output, or site content as needed.\n"
            "Preserve dependencies, attachment decoding, and dual-mode support from previous version.\n\n"
            "--PREVIOUS APP.PY CODE (START)--\n"
            f"{app_code}\n"
            "--PREVIOUS APP.PY CODE (END)--\n\n"
            "--- OUTPUT REQUIREMENT ---\n"
            "Output ONLY the new, full code for app.py (with core logic updated for round 2, all else untouched). Do NOT include markdown, code fences, or explanations. Output must be fully runnable and free of syntax/indentation errors."
        )

    return llm_generate_file(prompt)


def generate_workflow(brief, code, attachments=None, checks=None, output_dir="output"):
    """
    Generates prompt for GitHub Actions workflow that exports Flask app as static site.
    """
    attachments = attachments or []
    checks = checks or []

    checks_section = "\n".join(
        f"- {chk}" for chk in checks) or "- None specified."
    att_names = "\n".join(
        f"- {att['name']}" for att in attachments) or "- None provided."

    prompt = (
        f"TASK: Generate a GitHub Actions workflow YAML (deploy.yml) to export and deploy the Flask app below as a static site to GitHub Pages.\n\n"
        "-- ORIGINAL APP BRIEF --\n"
        f"{brief}\n\n"
        "--- FUNCTIONAL REQUIREMENTS ---\n"
        f"{checks_section}\n\n"
        "--- REFERENCE APP CODE ---\n"
        "Use the following app.py code as the definitive reference for runtime, environment, dependencies, data handling, commands, and output structure:\n"
        "----- BEGIN app.py -----\n"
        f"{code}\n"
        "----- END app.py -----\n\n"
        "--- CRITICAL WORKFLOW REQUIREMENTS ---\n"
        "1. The workflow must ONLY be triggered by either:\n"
        "    - A push to the main branch where 'app.py' was changed (created, updated, or deleted)\n"
        "    - A manual workflow dispatch event (workflow_dispatch)\n"
        "2. Do not trigger for changes to other files.\n"
        "3. Set up Python 3.11+ environment as required by the code\n"
        "4. Install all dependencies from requirements.txt (must include flask and any others required by the reference code)\n"
        "5. Ensure data.json exists in repo root\n"
        f"6. **CRITICAL:** Run 'python app.py --export' as defined in the reference code to generate static files to '{output_dir}/'\n"
        f"7. Upload ONLY the '{output_dir}/' directory using actions/upload-pages-artifact@v4. Do not use any download-pages-artifact action; only use the official upload and deploy actions.\n"
        "8. Deploy using actions/deploy-pages@v4 in a separate job\n"
        "9. Set permissions: contents: read, pages: write, id-token: write\n"
        "10. Use concurrency group 'pages'\n"
        "11. Use TWO jobs: 'build' (export static files) and 'deploy' (deploy to pages)\n"
        "12. **SECURITY CHECK:** The first build step must scan the entire repo for secrets using Gitleaks (zricethezav/gitleaks-action@v2 or latest). If secrets are found, fail the build and do not deploy.\n\n"
        "--- EXACT STRUCTURE ---\n"
        "Job 1 'build':\n"
        "  - Checkout code (actions/checkout@v4)\n"
        "  - Setup Python 3.11 (actions/setup-python@v4)\n"
        "  - Run Gitleaks scan\n"
        "  - Install requirements (must match those in reference app.py)\n"
        "  - Verify data.json exists\n"
        "  - Run: python app.py --export\n"
        f"  - Upload '{output_dir}/' directory using actions/upload-pages-artifact@v4\n\n"
        "Job 2 'deploy':\n"
        "  - Needs: build\n"
        "  - Environment: github-pages\n"
        "  - Uses: actions/deploy-pages@v4\n\n"
        "--- OUTPUT FORMAT ---\n"
        "Output ONLY valid YAML for .github/workflows/deploy.yml.\n"
        "Use only: actions/checkout@v4, actions/setup-python@v4, actions/upload-pages-artifact@v4, actions/deploy-pages@v4.\n"
        "Do not use any download-pages-artifact step—GitHub deploy-pages will use the uploaded artifact automatically.\n"
        "Reference only the code and requirements shown above. No markdown fences, no explanations: produce pure YAML only."
    )

    # prompt = (
    #     f"TASK: Generate a GitHub Actions workflow YAML (deploy.yml) to export and deploy the Flask app below as a static site to GitHub Pages.\n\n"
    #     "-- ORIGINAL APP BRIEF --\n"
    #     f"{brief}\n\n"
    #     "--- FUNCTIONAL REQUIREMENTS ---\n"
    #     f"{checks_section}\n\n"
    #     "--- REFERENCE APP CODE ---\n"
    #     "Use the following app.py code as the definitive reference for runtime, environment, dependencies, data handling, commands, and output structure:\n"
    #     "----- BEGIN app.py -----\n"
    #     f"{code}\n"
    #     "----- END app.py -----\n\n"
    #     "--- CRITICAL WORKFLOW REQUIREMENTS ---\n"
    #     "1. The workflow must ONLY be triggered by either:\n"
    #     "    - A push to the main branch where 'app.py' was changed (created, updated, or deleted)\n"
    #     "    - A manual workflow dispatch event (workflow_dispatch)\n"
    #     "2. Do not trigger for changes to other files.\n"
    #     "3. Set up Python 3.11+ environment as required by the code\n"
    #     "4. Install all dependencies from requirements.txt (must include flask and any others required by the reference code)\n"
    #     "5. Ensure data.json exists in repo root\n"
    #     f"6. **CRITICAL:** Run 'python app.py --export' as defined in the reference code to generate static files to '{output_dir}/'\n"
    #     f"7. Upload ONLY the '{output_dir}/' directory using actions/upload-pages-artifact@v4\n"
    #     "8. Deploy using actions/deploy-pages@v4 in a separate job\n"
    #     "9. Set permissions: contents: read, pages: write, id-token: write\n"
    #     "10. Use concurrency group 'pages'\n"
    #     "11. Use TWO jobs: 'build' (export static files) and 'deploy' (deploy to pages)\n"
    #     "12. **SECURITY CHECK:** The first build step must scan the entire repo for secrets using Gitleaks (zricethezav/gitleaks-action@v2 or latest). If secrets are found, fail the build and do not deploy.\n\n"
    #     "--- EXACT STRUCTURE ---\n"
    #     "Job 1 'build':\n"
    #     "  - Checkout code\n"
    #     "  - Setup Python 3.11\n"
    #     "  - Run Gitleaks scan\n"
    #     "  - Install requirements (must match those referenced in app.py)\n"
    #     "  - Verify data.json exists\n"
    #     "  - Run: python app.py --export\n"
    #     f"  - Upload '{output_dir}/' directory using actions/upload-pages-artifact@v4\n\n"
    #     "Job 2 'deploy':\n"
    #     "  - Needs: build\n"
    #     "  - Environment: github-pages\n"
    #     "  - Uses: actions/deploy-pages@v4\n\n"
    #     "--- OUTPUT FORMAT ---\n"
    #     "Output ONLY valid YAML for .github/workflows/deploy.yml.\n"
    #     "Use: checkout@v4, setup-python@v4, upload-pages-artifact@v4, deploy-pages@v4, and reference only the code and requirements shown above. No markdown fences, no explanations: produce pure YAML only."
    # )

    # prompt = (
    #     f"TASK: Generate a GitHub Actions workflow YAML (deploy.yml) to export and deploy the Flask app below as a static site to GitHub Pages.\n\n"
    #     "-- ORIGINAL APP BRIEF --\n"
    #     f"{brief}\n\n"
    #     "--- FUNCTIONAL REQUIREMENTS ---\n"
    #     f"{checks_section}\n\n"
    #     "--- REFERENCE APP CODE ---\n"
    #     "Use the following app.py code as the definitive reference for runtime, environment, dependencies, data handling, commands, and output structure:\n"
    #     "----- BEGIN app.py -----\n"
    #     f"{code}\n"
    #     "----- END app.py -----\n\n"
    #     "--- CRITICAL WORKFLOW REQUIREMENTS ---\n"
    #     "1. Trigger on push to main branch and on manual dispatch\n"
    #     "2. Set up Python 3.11+ environment as required by the code\n"
    #     "3. Install all dependencies from requirements.txt (must include flask and any others required by the reference code)\n"
    #     "4. Ensure data.json exists in repo root\n"
    #     f"5. **CRITICAL:** Run 'python app.py --export' as defined in the reference code to generate static files to '{output_dir}/'\n"
    #     f"6. Upload ONLY the '{output_dir}/' directory using actions/upload-pages-artifact@v4\n"
    #     "7. Deploy using actions/deploy-pages@v4 in a separate job\n"
    #     "8. Set permissions: contents: read, pages: write, id-token: write\n"
    #     "9. Use concurrency group 'pages'\n"
    #     "10. Use TWO jobs: 'build' (export static files) and 'deploy' (deploy to pages)\n"
    #     "11. **SECURITY CHECK:** The first build step must scan the entire repo for secrets using Gitleaks (zricethezav/gitleaks-action@v2 or latest). If secrets are found, fail the build and do not deploy.\n\n"
    #     "--- EXACT STRUCTURE ---\n"
    #     "Job 1 'build':\n"
    #     "  - Checkout code\n"
    #     "  - Setup Python 3.11\n"
    #     "  - Run Gitleaks scan\n"
    #     "  - Install requirements (must match those referenced in app.py)\n"
    #     "  - Verify data.json exists\n"
    #     "  - Run: python app.py --export\n"
    #     f"  - Upload '{output_dir}/' directory using actions/upload-pages-artifact@v4\n\n"
    #     "Job 2 'deploy':\n"
    #     "  - Needs: build\n"
    #     "  - Environment: github-pages\n"
    #     "  - Uses: actions/deploy-pages@v4\n\n"
    #     "--- OUTPUT FORMAT ---\n"
    #     "Output ONLY valid YAML for .github/workflows/deploy.yml.\n"
    #     "Use: checkout@v4, setup-python@v4, upload-pages-artifact@v4, deploy-pages@v4, and reference only the code and requirements shown above. No markdown fences, no explanations: produce pure YAML only."
    # )

    # prompt = (
    #     f"TASK: Generate a GitHub Actions workflow YAML to export Flask app as static site and deploy to GitHub Pages.\n\n"
    #     "--- ORIGINAL APP BRIEF ---\n"
    #     f"{brief}\n\n"
    #     "--- FUNCTIONAL REQUIREMENTS ---\n"
    #     f"{checks_section}\n\n"
    #     "--- APP CODE CONTEXT ---\n"
    #     f"The app is a Flask Python script (app.py) that:\n"
    #     f"- Can run as a development server: python app.py\n"
    #     f"- Can export static files: python app.py --export\n"
    #     f"- Reads data from data.json in repo root\n"
    #     f"- Processes attachments: {att_names}\n"
    #     f"- Exports static site to '{output_dir}/' directory when run with --export flag\n\n"
    #     "--- CRITICAL WORKFLOW REQUIREMENTS ---\n"
    #     "1. Trigger on push to main branch and allow manual dispatch\n"
    #     "2. Set up Python 3.11+ environment\n"
    #     "3. Install dependencies from requirements.txt (must include flask)\n"
    #     "4. Ensure data.json exists in repo root\n"
    #     f"5. **CRITICAL**: Run 'python app.py --export' to generate static files in '{output_dir}/'\n"
    #     f"6. Upload ONLY the '{output_dir}/' directory using actions/upload-pages-artifact@v4\n"
    #     "7. Deploy using actions/deploy-pages@v4 in separate job\n"
    #     "8. Set permissions: contents: read, pages: write, id-token: write\n"
    #     "9. Use concurrency group 'pages'\n"
    #     "10. Use TWO jobs: 'build' (export static files) and 'deploy' (deploy to pages)\n"
    #     "11. **SECURITY CHECK:** The first job step MUST scan the entire repository for secrets using Gitleaks (zricethezav/gitleaks-action@v2 or latest). If gitleaks finds issues, fail the build and do not proceed to deploy.\n\n"
    #     "--- EXACT STRUCTURE ---\n"
    #     "Job 1 'build':\n"
    #     "  - Checkout code\n"
    #     "  - Setup Python 3.11\n"
    #     "  - Run gitleak scans\n"
    #     "  - Install requirements (flask must be included)\n"
    #     "  - Verify data.json exists\n"
    #     "  - Run: python app.py --export\n"
    #     f"  - Upload '{output_dir}/' with actions/upload-pages-artifact@v4\n\n"
    #     "Job 2 'deploy':\n"
    #     "  - Needs: build\n"
    #     "  - Environment: github-pages\n"
    #     "  - Uses: actions/deploy-pages@v4\n\n"
    #     "--- OUTPUT FORMAT ---\n"
    #     "Generate ONLY valid YAML for .github/workflows/deploy.yml\n"
    #     "Use: checkout@v4, setup-python@v4, upload-pages-artifact@v4, deploy-pages@v4\n"
    #     "No markdown fences, no explanations, pure YAML only."
    # )

    return llm_generate_file(prompt)


def generate_readme(repo_name, brief, round_num, github_user, code):
    # prompt = (
    #     f"Write a rich README.md for the GitHub repo '{repo_name}'. "
    #     "Ensure it is comprehensive and user-friendly."
    #     f"Include: Summary ({brief}), setup instructions, usage, code explanation (for app.py, LICENSE, README.md) - {code} "
    #     f"MIT License. Link to Pages: https://{github_user}.github.io/{repo_name}/. State it's AI-generated."
    # )

    prompt = (
        f"Write a comprehensive, user-friendly README.md for the GitHub repository '{repo_name}'.\n\n"
        "Use the following as your complete source of truth for code and features:\n"
        "--- START app.py ---\n"
        f"{code}\n"
        "--- END app.py ---\n\n"
        "--- PROJECT SUMMARY ---\n"
        f"{brief}\n\n"
        "--- SETUP INSTRUCTIONS ---\n"
        "1. Outline all prerequisites required by the code, including Python version and external dependencies.\n"
        "2. Provide steps for installing requirements and running the app (include any data.json preparation as seen in code).\n"
        "3. Give instructions for both development (dynamic) and static export modes, and describe the --export flag if used.\n"
        "--- USAGE GUIDE ---\n"
        "Describe how to run the app, use its options/flags, and where the exported files will appear according to the real code provided.\n\n"
        "--- CODE EXPLANATION ---\n"
        "Explain the logical structure of app.py based on the code above—major components, data flow, file handling, and any non-obvious routines.\n"
        "- Summarize the LICENSE and how it applies to this code.\n"
        "- For README.md, state that it is AI-generated for transparency.\n\n"
        "--- LICENSE ---\n"
        "MIT License (brief summary of permissions and limitations).\n\n"
        "--- LIVE DEMO LINK ---\n"
        f"[GitHub Pages live site](https://{github_user}.github.io/{repo_name}/)\n\n"
        "--- AI GENERATION NOTICE ---\n"
        "End by stating that this README and the code were generated with an AI tool."
    )

    logger.info("Generating README.md...")
    return llm_generate_file(prompt)


def generate_requirements(code):
    prompt = (
        f"For the attached code snippet, please gather and provide the requirements.txt content - {code}"
        "Do not include built-in Python modules."
        "List each package name and the required version (if known); otherwise, latest is fine."
        "Output requirements.txt as plain text only—do not use code fences, markdown, or add any explanations."
    )
    logger.info("Generating Requirements.txt")
    return llm_generate_file(prompt)


def generate_license():
    prompt = (
        "Based on FUNCTIONAL REQUIREMENTS & CHECK about license in the previous chat, create a license file."
        " If no license check mentioned, create MIT license as default"
        "Output license  as plain text only—do not use code fences, markdown, or add any explanations."
    )
    logger.info("Generating LICENSE")
    return llm_generate_file(prompt)


def get_run_id_for_commit(owner, repo, commit_sha, token, workflow_filename=None):
    """Finds the GitHub Actions run ID for a specific commit."""
    # This URL filters runs by the triggering commit
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}
    params = {"head_sha": commit_sha}

    logger.info(
        f"Searching for workflow run matching commit SHA: {commit_sha[:7]}")

    # Retry a few times in case the run hasn't been created yet
    for _ in range(5):
        try:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            runs = resp.json().get("workflow_runs", [])

            if runs:
                # If a workflow filename is provided, find the exact match
                if workflow_filename:
                    for run in runs:
                        if run['path'].endswith(workflow_filename):
                            logger.info(
                                f"Found run ID {run['id']} for workflow file '{workflow_filename}'.")
                            return run['id']
                else:
                    # Otherwise, assume the first one is the correct one
                    run_id = runs[0]['id']
                    workflow_name = runs[0]['name']
                    logger.info(
                        f"Found run ID {run_id} for workflow '{workflow_name}'.")
                    return run_id

            logger.info("Run not found yet, retrying in 5 seconds...")
            time.sleep(5)

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error finding run ID: {e} - {e.response.text}")
            return None

    logger.error("Could not find a workflow run for the specified commit.")
    return None


def wait_for_actions_run(owner, repo, commit_sha, token, workflow_filename=None, timeout=180):
    """Dynamically finds and waits for a GitHub Actions run to complete."""

    run_id = get_run_id_for_commit(
        owner, repo, commit_sha, token, workflow_filename)
    if not run_id:
        return False

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}"
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}
    start_time = time.time()

    logger.info(f"Polling status for Run ID: {run_id}")
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            run_data = resp.json()

            status = run_data.get("status")
            conclusion = run_data.get("conclusion")

            if status == "completed":
                logger.info(
                    f"Workflow completed with conclusion: {conclusion}")
                return conclusion == "success"

            logger.info(
                f"Workflow is still running (Status: {status})... waiting 10s.")
            time.sleep(10)

        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP Error polling run status: {e} - {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            time.sleep(10)

    logger.warning("Timed out waiting for workflow run to complete.")
    return False


def ensure_pages_enabled(owner, repo_name, token, max_retries=5):
    """
    Enable GitHub Pages with GitHub Actions as the source.
    Retries if needed to handle API delays.
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}/pages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # Check if Pages already exists
    resp = requests.get(url, headers=headers)
    logger.info(f"GET Pages status: {resp.status_code}")

    if resp.status_code == 200:
        pages_info = resp.json()
        logger.info(
            f"Pages already enabled. Source: {pages_info.get('source')}")
        return True

    if resp.status_code == 404:
        # Pages not enabled - create it with GitHub Actions source
        logger.info("Enabling GitHub Pages with GitHub Actions source...")

        payload = {
            "source": {
                "branch": "main",
                "path": "/"
            },
            "build_type": "workflow"  # This enables GitHub Actions deployment
        }

        for attempt in range(max_retries):
            resp = requests.post(url, headers=headers, json=payload)
            logger.info(
                f"POST Pages (attempt {attempt + 1}): {resp.status_code} - {resp.text[:200]}")

            if resp.status_code in [201, 200]:
                logger.info(
                    "✓ GitHub Pages enabled successfully with workflow source")
                time.sleep(5)  # Give GitHub time to process
                return True
            elif resp.status_code == 409:
                logger.info("Pages creation in progress, waiting...")
                time.sleep(3)
            else:
                logger.warning(f"Unexpected response: {resp.status_code}")
                time.sleep(2)

        logger.error("Failed to enable GitHub Pages after retries")
        return False

    logger.warning(f"Unexpected GET response: {resp.status_code}")
    return False


def process_request(req):
    try:
        logger.info(
            f"Processing new request for task '{req.get('task')}', round {req.get('round')}")
        email = req["email"]
        task = req["task"]
        round_num = req["round"]
        nonce = req["nonce"]
        brief = req["brief"]
        attachments = req.get("attachments", [])
        checks = req.get("checks", [])
        evaluation_url = req["evaluation_url"]
        # repo_id = str(uuid.uuid4()).split("-")[0]
        # repo_name = f"{task}-{repo_id}" if round_num == 1 else req.get("repo_name")
        user = gh.get_user()
        repo_name = get_repo_name_from_task(task)
        commit_sha = None

        try:
            # Try to find the repo created in a previous round.
            repo = user.get_repo(repo_name)
            logger.info(f"Found existing repo for task: {repo_name}")
        except UnknownObjectException:
            logger.info(f"No existing repo found for task: {repo_name}")
            logger.info(f"Creating new repo for task: {repo_name}")
            repo = user.create_repo(repo_name, private=False)
            # If it's not found, this MUST be Round 1. Create it.

        repo_url = repo.html_url
        pages_url = f"https://{GITHUB_USER}.github.io/{repo_name}/"
        workflow_path = ".github/workflows/deploy.yml"

        if round_num == 1:
            context_to_save = {
                "brief_history": [brief],
                "checks_history": checks,
                "attachment_history": attachments
            }

            context_content = json.dumps(context_to_save, indent=2)
            attachments_content = json.dumps(
                {"attachments": attachments}, indent=2)

            previous_code = None
            logger.info(
                "Step 1/8: Generating code file with export capability")
            code = generate_code(brief, previous_code,
                                 attachments, round_num, checks)

            logger.info("Step 2/8: Generating README.md")
            readme = generate_readme(
                repo_name, brief, round_num, GITHUB_USER, code)

            logger.info("Step 3/8: Generating requirements.txt")
            req_txt = generate_requirements(code)
            # Ensure Flask is included
            if "flask" not in req_txt.lower():
                req_txt = "flask\n" + req_txt

            logger.info("Step 4/8: Generating LICENSE")
            license_content = generate_license()

            logger.info("Step 5/8: Generating workflow YAML")
            workflow_content = generate_workflow(
                brief, code, attachments, checks, output_dir="output")

            # CRITICAL: Enable Pages BEFORE creating any files
            logger.info("Step 6/8: Enabling GitHub Pages with Actions source")
            pages_enabled = ensure_pages_enabled(
                GITHUB_USER, repo_name, GITHUB_TOKEN)
            if not pages_enabled:
                logger.warning("Failed to enable Pages, but continuing...")

            logger.info("Step 7/8: Creating repo files")

            upsert_github_file(repo, "data.json",
                               attachments_content, "Add attachments data")
            # upsert_github_file(repo, "app.py", code,
            #                    "Initial Flask app with export")
            upsert_github_file(repo, "requirements.txt",
                               req_txt, "Add dependencies")
            upsert_github_file(repo, "LICENSE", license_content, "Add license")
            upsert_github_file(repo, "README.md", readme, "Add README")
            upsert_github_file(repo, "context.json",
                               context_content, "Add context")
            upsert_github_file(
                repo, workflow_path, workflow_content, "Add Pages deployment workflow")
            time.sleep(5)  #
            result = upsert_github_file(repo, "app.py", code,
                                        "Initial Flask app with export")
            # repo.create_file(
            #     "data.json", "Add attachments data", attachments_content)
            # repo.create_file("app.py", "Initial Flask app with export", code)
            # repo.create_file("requirements.txt", "Add dependencies", req_txt)
            # repo.create_file("LICENSE", "Add license", license_content)
            # repo.create_file("README.md", "Add README", readme)
            # repo.create_file("context.json", "Add context", context_content)
            # result = repo.create_file(
            #     workflow_path, "Add Pages deployment workflow", workflow_content)
            if result is not None:
                commit_sha = result["commit"].sha
            else:
                commit_sha = repo.get_commits()[0].sha
        else:
            # Update existing repo for subsequent rounds
            logger.info(
                f"Updating existing repo files for round - {round_num}")

            try:
                context_file = repo.get_contents("context.json")
                previous_code = repo.get_contents(
                    "app.py").decoded_content.decode("utf-8")
                past_context = json.loads(
                    context_file.decoded_content.decode())

                existing_attachments = {
                    (att['name'], att['url']) for att in past_context.get("attachment_history", [])}
                for attachment in attachments:
                    if (attachment['name'], attachment['url']) not in existing_attachments:
                        past_context.setdefault(
                            "attachment_history", []).append(attachment)
                        logger.info(
                            f"Appended new attachment: '{attachment['name']}'")

                if brief not in past_context["brief_history"]:
                    past_context["brief_history"].append(brief)

                brief = "This is a cumulative brief...\n" + \
                    "\n".join(past_context["brief_history"])
                checks = past_context["checks_history"]
                full_attachments = past_context.get(
                    "attachment_history", attachments)
            except UnknownObjectException:
                logger.warning(
                    "context.json not found. Using current request's data only.")
                past_context = None
                previous_code = None
                full_attachments = attachments
            except Exception as e:
                logger.error(
                    f"Error loading context.json: {e}. Using current request's data only.")
                past_context = None
                previous_code = None
                full_attachments = attachments

            if past_context:
                context_content = json.dumps(past_context, indent=2)
                repo.update_file("context.json", f"Update context for round {round_num}",
                                 context_content, context_file.sha)

            # Update attachments
            attachments_content = json.dumps(
                {"attachments": full_attachments}, indent=2)

            logger.info("Step 1/4: Generating code for update")
            logging.info("Using previous code length: %d", len(
                previous_code) if previous_code else 0)
            code = generate_code(brief, previous_code,
                                 full_attachments, round_num, checks)
            logger.info("Step 2/4: Generating README.md for update")
            readme = generate_readme(
                repo_name, brief, round_num, GITHUB_USER, code)
            logger.info("Step 3/4: Generating workflow for update")
            workflow_content = generate_workflow(
                brief, code, full_attachments, checks, output_dir="output")

            logger.info("Step 4/4: Generating requirements.txt for update")
            req_txt = generate_requirements(code)

            upsert_github_file(repo, "data.json",
                               attachments_content, "Add attachments data for round {round_num}")
            upsert_github_file(repo, "requirements.txt",
                               req_txt, f"Update for round {round_num}")
            upsert_github_file(repo, "README.md", readme,
                               f"Update README for round {round_num}")
            upsert_github_file(
                repo, workflow_path, workflow_content, f"Update workflow for round {round_num}")
            time.sleep(5)  #
            result = upsert_github_file(repo, "app.py", code,
                                        f"Update for round {round_num}")
            # repo.update_file(
            #     "data.json", f"Add attachments data for round {round_num}", attachments_content,
            #                  repo.get_contents("data.json").sha)
            # repo.update_file("app.py", f"Update for round {round_num}", code,
            #                  repo.get_contents("app.py").sha)
            # repo.update_file("requirements.txt", f"Update for round {round_num}", req_txt,
            #                  repo.get_contents("requirements.txt").sha)
            # repo.update_file("README.md", f"Update README for round {round_num}", readme,
            #                  repo.get_contents("README.md").sha)
            # result = repo.update_file(workflow_path, f"Update workflow for round {round_num}",
            #                           workflow_content, repo.get_contents(workflow_path).sha)
            if result is not None:
                commit_sha = result["commit"].sha
            else:
                commit_sha = repo.get_commits()[0].sha

        logger.info("Step: Waiting for workflow to complete")
        actions_success = wait_for_actions_run(
            GITHUB_USER, repo_name, commit_sha, GITHUB_TOKEN,
            workflow_filename="deploy.yml",
            timeout=180  # 3 minutes
        )

        if actions_success:
            logger.info("✓ Workflow completed successfully")
        else:
            logger.warning("⚠ Workflow did not complete successfully")

        logger.info("Notifying evaluation endpoint")
        eval_payload = {
            "email": email, "task": task, "round": round_num, "nonce": nonce,
            "repo_url": repo_url, "commit_sha": commit_sha,
            "pages_url": pages_url,
        }

        # Retry logic for evaluation notification
        delay = 1
        for attempt in range(6):
            logger.info(f"Sending evaluation, attempt {attempt+1}")
            resp = requests.post(evaluation_url, json=eval_payload,
                                 headers={"Content-Type": "application/json"})
            if resp.status_code == 200:
                logger.info("✓ Evaluation notification sent")
                break
            time.sleep(delay)
            delay *= 2

        logger.info(f"✓ Process complete for {task} round {round_num}")

    except Exception as ex:
        logger.error(f"process_request error: {ex}", exc_info=True)


def ensure_pages_site(owner, repo_name, branch, token, path="/"):
    """
    Robustly create or update a GitHub Pages site.
    Tries GET; if 404, uses POST to create; if 200, uses PUT to update.
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}/pages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }
    # 1. Check if Pages site exists
    resp = requests.get(url, headers=headers)
    logger.info(f"GET Pages site: {resp.status_code} - {resp.text}")
    if resp.status_code == 404:
        # Create Pages site (POST)
        logger.info("Pages site not found, creating...")
        resp = requests.post(url, headers=headers, json={
            "source": {"branch": branch, "path": path}
        })
        logger.info(f"POST Pages site: {resp.status_code} - {resp.text}")
        time.sleep(3)  # Give GitHub time to initialize
    elif resp.status_code == 200:
        # Update source branch/path (PUT)
        logger.info("Pages site exists, updating source branch/path...")
        resp = requests.put(url, headers=headers, json={
            "source": {"branch": branch, "path": path}
        })
        logger.info(f"PUT Pages site: {resp.status_code} - {resp.text}")
    else:
        logger.warning(
            f"Unexpected GET response: {resp.status_code} - {resp.text}")
    try:
        result = resp.json()
    except Exception:
        result = resp.text
    return resp.status_code, result


# @app.route("/api-endpoint", methods=["POST"])
# def api_endpoint():
#     logger.info("API endpoint called.")
#     req = request.get_json()
#     if req.get("secret") != GOOGLE_FORM_SECRET:
#         logger.warning("Invalid secret provided.")
#         return jsonify({"error": "Invalid secret"}), 403
#     threading.Thread(target=process_request, args=(req,)).start()
#     logger.info("Request acknowledged and processing in background thread.")
#     return jsonify({"status": "acknowledged"}), 200

# In your Flask endpoint, put the request in the queue instead of spawning a thread directly:
@app.route("/api-endpoint", methods=["POST"])
def api_endpoint():
    logger.info("API endpoint called.")
    req = request.get_json()
    if req.get("secret") != GOOGLE_FORM_SECRET:
        logger.warning("Invalid secret provided.")
        return jsonify(error="Invalid secret"), 403
    task_queue.put(req)  # Add request to the queue
    logger.info("Request acknowledged and queued for background processing.")
    return jsonify(status="acknowledged"), 200


@app.route("/", methods=["GET"])
def home():
    logger.info("Health check: home endpoint.")
    return "API is running!", 200


if __name__ == "__main__":
    logger.info("Starting Flask server on port 7860...")
    app.run(host="0.0.0.0", port=7860)
