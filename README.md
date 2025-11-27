# LLM Quiz Solver

An intelligent, autonomous quiz-solving system powered by Large Language Models (LLMs) that can handle multi-step quizzes involving web scraping, data processing, audio transcription, and complex analysis tasks.

## Features

### 🤖 Intelligent Quiz Solving
- **Direct Answer Mode**: Solves simple questions directly from provided context
- **Web Scraping Mode**: Automatically scrapes additional data from linked pages
- **Code Generation Mode**: Generates and executes Python code for complex data processing tasks

### 🔄 Self-Correction & Retry Logic
- **Error-Based Self-Correction**: When generated code fails, the system re-prompts the LLM with error details for automatic debugging
- **Answer Retry Mechanism**: Automatically retries incorrect answers up to 3 times with error feedback
- **Script Execution Retry**: Up to 3 attempts for script execution failures with LLM-driven fixes

### 📊 Multi-Format Data Processing
- **CSV Processing**: Pandas-based data analysis and manipulation
- **PDF Processing**: Text extraction using PyPDF2
- **Audio Transcription**: Local transcription with faster-whisper
- **Image Processing**: Resize, crop, and compress images using Pillow
- **Web Content**: Selenium-based rendering and scraping

### 🎯 Robust Error Handling
- **Timeout Management**: 15-second timeout for submissions with retry capability
- **Empty Response Detection**: Immediately bails out on empty server responses
- **Type Normalization**: Handles both string and boolean values for response fields
- **JSON Parsing Fallback**: Uses `ast.literal_eval` for Python dict formats

### 🔗 Quiz Chain Management
- **Automatic Continuation**: Follows quiz chains automatically via next URLs
- **Loop Detection**: Stops when next URL matches current URL
- **Result Tracking**: Comprehensive logging of all attempts and results

## Requirements

### System Dependencies
- Python 3.12
- Chrome/Chromium (for Selenium)
- ChromeDriver

### Python Dependencies
```txt
fastapi
uvicorn[standard]
requests
selenium
pandas
beautifulsoup4
flask
PyPDF2
faster-whisper
scipy
sklearn
Pillow
```

## Installation

### Local Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd TDS-Project1
```

2. **Create virtual environment**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt
```

4. **Set environment variables**
```bash
export AIPIPE_TOKEN="your_aipipe_token"
export GOOGLE_FORM_SECRET="your_secret"
```

5. **Run the server**
```bash
flask run --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Build the image**
```bash
docker build -t llm-quiz-solver .
```

2. **Run the container**
```bash
docker run -p 7860:7860 \
  -e AIPIPE_TOKEN="your_aipipe_token" \
  -e GOOGLE_FORM_SECRET="your_secret" \
  llm-quiz-solver
```

## Usage

### API Endpoint

**POST** `/quiz`

**Request Body:**
```json
{
  "email": "your@email.com",
  "secret": "your_secret",
  "url": "https://quiz-url.com/start"
}
```

**Response:**
```json
{
  "message": "Quiz solving started in background worker. Check 'generated_solver_temp.py' for the last executed script.",
  "status": "acknowledged"
}
```

### Example Usage

```bash
curl -X POST "http://localhost:8000/quiz" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "secret": "powerhouse",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

## How It Works

### 1. Quiz Analysis
- Renders quiz page with Selenium (handles JavaScript)
- Extracts text content, links, and data files
- Downloads CSV, PDF, and audio files for processing
- Previews files (up to 2MB) for LLM context

### 2. Strategy Selection
The LLM chooses from three approaches:

**PRIORITY 1 - Direct Solving**
- For simple questions answerable from page content
- Returns answer immediately

**PRIORITY 2 - Web Scraping**
- When additional data must be fetched from linked pages
- Uses Selenium for JavaScript-heavy sites

**PRIORITY 3 - Code Generation**
- For complex data processing tasks
- Generates Python 3.12-compatible code
- Executes locally with injected credentials
- Supports pandas, PyPDF2, PIL, and more

### 3. Answer Submission & Retry
- Submits answer with 15-second timeout
- On incorrect answer: Re-prompts LLM with error reason
- Retries up to 3 times per question
- Continues to next URL after max retries or success

### 4. Quiz Chain Continuation
- Automatically follows `next_url` from responses
- Stops when URL repeats (loop detection)
- Logs comprehensive results for all attempts

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AIPIPE_TOKEN` | Authentication token for AIPipe LLM service | Yes |
| `OPENAI_API_KEY` | Fallback for AIPIPE_TOKEN | No |
| `GOOGLE_FORM_SECRET` | Secret for form submissions | Yes |

### Timeouts

- **Script Execution**: 60 seconds (configurable)
- **Submission Request**: 15 seconds
- **Max Quiz Chain Duration**: 300 seconds (5 minutes)

### Retry Limits

- **Script Execution Errors**: 3 attempts
- **Incorrect Answers**: 3 attempts
- **Timeout Errors**: 3 attempts

## Supported Libraries

Generated Python code can use:
- **Standard Library**: os, json, re, math, etc.
- **Data Processing**: pandas, numpy
- **Web**: requests, beautifulsoup4, selenium
- **PDF**: PyPDF2
- **Image**: PIL (Pillow)
- **Audio**: faster_whisper
- **ML/Stats**: scipy, sklearn

## Error Handling

### Script Execution
- Captures stderr for debugging
- Returns error details to LLM for self-correction
- Saves generated scripts to `generated_solver_temp.py`

### Answer Submission
- **Empty Response**: Bail out immediately
- **Invalid JSON**: Bail out immediately
- **Timeout**: Retry up to 3 times
- **Network Errors**: Bail out immediately

### Quiz Chain
- **Missing `correct` field**: Log error and stop
- **Same URL returned**: Recognize as completion
- **Time limit exceeded**: Stop gracefully

## Debugging

### Check Generated Scripts
```bash
cat generated_solver_temp.py
```

### View Logs
Application logs show:
- Prompts sent to LLM
- Script execution output
- Submission attempts and responses
- Error messages with context

### Common Issues

**Issue**: `ModuleNotFoundError` in generated script
- **Solution**: Library not in allowed list. Update prompt or add to requirements.txt

**Issue**: Script timeout
- **Solution**: Increase timeout in `execute_hybrid_script`

**Issue**: Quiz chain loops infinitely
- **Solution**: Same-URL detection should stop this. Check logs for `next_url` values.

## Deployment

### HuggingFace Spaces

1. Push code to HuggingFace Space repository
2. Add secrets in Space settings:
   - `AIPIPE_TOKEN`
   - `GOOGLE_FORM_SECRET`
3. Space will auto-build and deploy

### Production Considerations

- Use production WSGI server (gunicorn, uwsgi)
- Implement rate limiting
- Add request authentication
- Monitor background worker queue
- Set up log aggregation
- Configure resource limits

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
