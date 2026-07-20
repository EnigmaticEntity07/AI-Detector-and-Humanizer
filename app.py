import streamlit as st
import streamlit.components.v1 as components
import os
import json
import time

import random
import pandas as pd
# pyrefly: ignore [missing-import]
import google.generativeai as genai

# Configure Gemini strictly from environment for Railway
api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

# --- Model Configuration ---
# Fallback chain: each model has its own free-tier quota, so if one is exhausted
# the next model is tried automatically. Order: preferred → fallbacks.
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",   # lighter model, separate quota pool
    "gemini-flash-latest",         # alias that routes to latest available flash
]

MAX_RETRIES = 3       # retries per model for transient 429s (per-minute limits)
RETRY_BASE_DELAY = 5  # seconds; doubles each retry

API_TIMEOUT = 60  # seconds – maximum time to wait for a single Gemini call


def call_gemini(prompt, timeout: int = API_TIMEOUT):
    """Try each model in the fallback chain with retry/backoff for rate limits.

    Raises ``TimeoutError`` if no response is received within *timeout* seconds.
    """
    import threading

    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(MAX_RETRIES):
            # -- run the API call in a daemon thread so we can enforce a timeout --
            result_container: dict = {}

            def _call():
                try:
                    model = genai.GenerativeModel(model_name)
                    result_container["response"] = model.generate_content(prompt)
                except Exception as exc:
                    result_container["error"] = exc

            thread = threading.Thread(target=_call, daemon=True)
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                # Thread is still running → treat as timeout
                raise TimeoutError(
                    f"Gemini API call timed out after {timeout} seconds."
                )

            if "error" in result_container:
                e = result_container["error"]
                err_str = str(e)
                last_error = e
                # Transient per-minute rate limit → retry with backoff
                if "429" in err_str and "per" in err_str.lower() and "minute" in err_str.lower():
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
                # Daily quota exhausted or model unavailable → skip to next model
                if "429" in err_str or "404" in err_str:
                    break
                # Any other error → raise immediately
                raise e

            if "response" in result_container:
                return result_container["response"]
    # All models exhausted
    raise last_error  # type: ignore[misc]

def analyze_text(text):
    if not api_key:
        return None, "Error: GEMINI_API_KEY is not set in Streamlit secrets or environment variables."
    
    try:
        prompt = f"""
        Analyze the following text for perplexity and burstiness to determine if it was written by an AI.
        Return a JSON object with two keys:
        - "score": an integer from 0 to 100 representing the probability that the text is AI-generated (0 = completely human, 100 = completely AI).
        - "explanation": a brief explanation of your analysis, mentioning perplexity and burstiness.
        
        Text to analyze:
        {text}
        
        Return ONLY valid JSON.
        """
        response = call_gemini(prompt)
        # Parse JSON
        result_text = response.text.strip()
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        elif result_text.startswith('```'):
            result_text = result_text[3:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
            
        result = json.loads(result_text.strip())
        score = result.get("score", 0)
        if isinstance(score, str):
            score = int(score.replace('%', '').strip())
        else:
            score = int(score)
        return score, result.get("explanation", "No explanation provided.")
    except Exception as e:
        return None, f"An error occurred during analysis: {str(e)}"

# ---------------------------------------------------------------------------
# Few-Shot Humanizer – dynamically samples human-written examples each call
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "combined_dataset.csv.gz")

# Cache the human-only rows in memory so we don't re-read 100 MB on every click
@st.cache_data(show_spinner=False)
def _load_human_texts() -> list[str]:
    """Return all human-written texts (label == 0) from the training set."""
    df = pd.read_csv(DATASET_PATH)
    return df.loc[df["label"] == 0, "text"].dropna().tolist()


def get_human_examples(n: int = 3, max_chars: int = 500) -> list[str]:
    """Randomly select *n* human-written examples, truncated to *max_chars*.

    A fresh random sample is drawn on every call so the humanizer never
    develops a single repetitive style.
    """
    all_texts = _load_human_texts()
    samples = random.sample(all_texts, min(n, len(all_texts)))
    # Truncate long essays to keep the prompt within token limits
    truncated = []
    for s in samples:
        if len(s) > max_chars:
            # Cut at the last sentence boundary within the limit
            cut = s[:max_chars]
            last_period = cut.rfind(".")
            if last_period > max_chars // 2:
                cut = cut[: last_period + 1]
            truncated.append(cut)
        else:
            truncated.append(s)
    return truncated


def humanize_text(text):
    if not api_key:
        return None, "Error: GEMINI_API_KEY is not set in Streamlit secrets or environment variables."

    try:
        # Pull 3 fresh human-written examples for few-shot context
        examples = get_human_examples(3)
        examples_block = "\n\n---\n\n".join(
            f"Example {i + 1}:\n{ex}" for i, ex in enumerate(examples)
        )

        prompt_1 = f"""You are an expert humanizer. Study the varied sentence lengths, colloquial pacing, and natural imperfections in these three human-written examples:

{examples_block}

Now, rewrite the following predictable AI text to match the exact structural style and natural variance of those human examples.
Keep the original meaning and key points intact.
Return ONLY the rewritten text, without any additional explanations or markdown formatting.

You must introduce natural human variance in sentence length, structure, and vocabulary. However, under NO circumstances should you introduce grammatical errors, typos, slang, or factual inaccuracies to achieve this. The output must remain 100% grammatically correct, highly meaningful, and retain the exact original intent and professional quality of the input text.

Negative Constraints: You MUST absolutely avoid common AI filler words and transition tropes. DO NOT use words like: delve, tapestry, testament, beacon, furthermore, moreover, it is important to note, in conclusion, ultimately, a symphony of.

Structural Asymmetry (Burstiness): Human writing is messy. Force the text to have extreme variance in sentence length. Follow a very long, complex sentence with a tiny, punchy one (e.g., 3-5 words). Use occasional rhetorical questions.

Perspective: Shift the tone from 'objective observer' to a slightly more subjective, conversational voice where appropriate.

Text to rewrite:
{text}"""

        # Pass 1: Rewrite
        response_1 = call_gemini(prompt_1)
        rewritten_text = response_1.text.strip()
        
        # Pass 2: Critique
        prompt_2 = f"""Analyze this text for any lingering AI tropes, robotic transitions, or uniform sentence pacing. Edit it one final time to make it indistinguishable from a casual, highly educated human writer.
        
Return ONLY the final edited text, without any additional explanations or markdown formatting.

Text to edit:
{rewritten_text}"""
        
        response_2 = call_gemini(prompt_2)
        return response_2.text.strip(), None
    except TimeoutError:
        return None, "⏱️ The humanization request timed out. The API took too long to respond. Please try again in a moment."
    except Exception as e:
        return None, f"An error occurred during humanization: {str(e)}"

st.set_page_config(
    page_title="AI Detector & Humanizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
    /* ── Global reset for transparent Streamlit shell ── */
    .stApp, .stApp > div, [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 20, 0.75) !important;
        backdrop-filter: blur(12px);
    }

    /* ── Typography ── */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        color: #f9fafb;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 24px rgba(99, 102, 241, 0.4);
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        letter-spacing: 0.01em;
    }

    /* ── Text area ── */
    .stTextArea textarea {
        background-color: rgba(17, 24, 39, 0.85) !important;
        border: 1.5px solid rgba(99, 102, 241, 0.35) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        color: #f3f4f6 !important;
        backdrop-filter: blur(8px);
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.75rem 1.5rem;
        letter-spacing: 0.01em;
        border: none !important;
        transition: all 0.25s ease;
    }
    div[data-testid="column"]:nth-of-type(1) .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.35);
    }
    div[data-testid="column"]:nth-of-type(1) .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 2px 12px rgba(16, 185, 129, 0.35);
    }
    div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
        transform: translateY(-2px);
    }

    /* ── Metric cards / expanders — glassmorphism ── */
    [data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.7) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(8px);
        padding: 1rem !important;
    }
    .streamlit-expanderHeader {
        background: rgba(17, 24, 39, 0.7) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(8px);
    }
    .streamlit-expanderContent {
        background: rgba(10, 12, 20, 0.8) !important;
        backdrop-filter: blur(8px);
    }

    /* ── Spinner / info / warning banners ── */
    .stAlert {
        background: rgba(17, 24, 39, 0.8) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Import the local classifier
# ---------------------------------------------------------------------------
from classifier import predict as classify_text, is_model_available, load_bundle

@st.cache_resource
def get_model_bundle():
    """Cache the ML model bundle so it doesn't crash the server on reload."""
    return load_bundle()

# ---------------------------------------------------------------------------
# Lexicon Swarm — fullscreen Three.js background (InstancedMesh)
# ---------------------------------------------------------------------------
def render_lexicon_swarm(app_state: str = "idle"):
    """
    Render the Lexicon Swarm background.

    app_state:
        'idle'        – particles drift slowly and ambiently (indigo)
        'ai_detected' – particles snap into a rigid 3-D grid (red/orange)
        'humanized'   – particles swirl organically like a flock (blue/green)
    """
    html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{
    width: 100%; height: 100%;
    overflow: hidden;
    background: #070a14;
  }}
  canvas {{ display:block; }}
</style>
</head>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script>
(function() {{

// ── State passed from Python ─────────────────────────────────────────────────
const APP_STATE = "{app_state}";

// ── Scene setup ─────────────────────────────────────────────────────────────
const scene    = new THREE.Scene();
const camera   = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 500);
camera.position.set(0, 0, 55);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x070a14, 1);
document.body.appendChild(renderer.domElement);

// ── Escape iframe to fullscreen background ───────────────────────────────────
(function escapeIframe() {{
  try {{
    const iframe = window.frameElement;
    if (!iframe) return;
    const applyFixed = el => {{
      el.style.cssText +=
        'position:fixed!important;top:0!important;left:0!important;'
        +'width:100vw!important;height:100vh!important;'
        +'z-index:-1!important;border:none!important;'
        +'margin:0!important;padding:0!important;overflow:hidden!important;';
    }};
    applyFixed(iframe);
    iframe.style.pointerEvents = 'none';
    // Walk up DOM in the parent document and clamp wrappers
    let el = iframe.parentElement;
    for (let i = 0; i < 8 && el && el !== window.parent.document.body; i++) {{
      applyFixed(el);
      el = el.parentElement;
    }}
  }} catch(e) {{ /* cross-origin guard */ }}
}})();

// ── Mouse tracking from the PARENT window ────────────────────────────────────
const mouse = {{ nx: 0, ny: 0 }};  // normalised -1..1
try {{
  window.parent.addEventListener('mousemove', e => {{
    mouse.nx = (e.clientX / window.parent.innerWidth)  * 2 - 1;
    mouse.ny = -(e.clientY / window.parent.innerHeight) * 2 + 1;
  }}, {{ passive: true }});
}} catch(e) {{ /* cross-origin guard */ }}

// ── Particle setup ───────────────────────────────────────────────────────────
const COUNT = 3000;
const SPREAD = 38;

// Shared tiny plane geometry for all instances
const planeGeo = new THREE.PlaneGeometry(0.09, 0.09);
const mat = new THREE.MeshBasicMaterial({{
  color: 0x6366f1,
  transparent: true,
  opacity: 0.85,
  side: THREE.DoubleSide,
  depthWrite: false,
}});

const mesh = new THREE.InstancedMesh(planeGeo, mat, COUNT);
mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(mesh);

// Per-particle state arrays
const posArr   = new Float32Array(COUNT * 3);
const velArr   = new Float32Array(COUNT * 3);
const phaseArr = new Float32Array(COUNT);       // random phase for sine noise
const gridArr  = new Float32Array(COUNT * 3);   // target grid positions
const dummy    = new THREE.Object3D();

// Initialise random positions and velocities
for (let i = 0; i < COUNT; i++) {{
  posArr[i*3]   = (Math.random()-0.5) * SPREAD;
  posArr[i*3+1] = (Math.random()-0.5) * SPREAD;
  posArr[i*3+2] = (Math.random()-0.5) * 20;
  velArr[i*3]   = (Math.random()-0.5) * 0.012;
  velArr[i*3+1] = (Math.random()-0.5) * 0.012;
  velArr[i*3+2] = (Math.random()-0.5) * 0.008;
  phaseArr[i]   = Math.random() * Math.PI * 2;
}}

// Pre-compute grid target positions (20 × 10 × 15 = 3000 cells)
{{
  const GX = 20, GY = 10, GZ = 15;
  const SX = 2.6, SY = 2.6, SZ = 2.6;
  let idx = 0;
  for (let z = 0; z < GZ; z++) {{
    for (let y = 0; y < GY; y++) {{
      for (let x = 0; x < GX; x++) {{
        gridArr[idx*3]   = (x - GX/2 + 0.5) * SX;
        gridArr[idx*3+1] = (y - GY/2 + 0.5) * SY;
        gridArr[idx*3+2] = (z - GZ/2 + 0.5) * SZ;
        idx++;
      }}
    }}
  }}
}}

// ── Colour helpers ───────────────────────────────────────────────────────────
const COLORS = {{
  idle:        [[ 0.38, 0.40, 0.95 ], [ 0.48, 0.24, 0.90 ]],  // indigo/violet
  ai_detected: [[ 0.94, 0.27, 0.27 ], [ 0.98, 0.60, 0.14 ]],  // red/orange
  humanized:   [[ 0.06, 0.73, 0.51 ], [ 0.02, 0.71, 0.83 ]],  // green/cyan
}};

const c0 = COLORS[APP_STATE] ? COLORS[APP_STATE][0] : COLORS.idle[0];
const c1 = COLORS[APP_STATE] ? COLORS[APP_STATE][1] : COLORS.idle[1];

// Animated accent colour (oscillates between c0 and c1)
const accentColor = new THREE.Color();

// ── Subtle background fog ────────────────────────────────────────────────────
scene.fog = new THREE.FogExp2(0x070a14, 0.008);

// ── Bloom-like glow via a large additive point light in the centre ───────────
const glow = new THREE.PointLight(0x6366f1, 2.5, 120);
scene.add(glow);

// ── Animation loop ───────────────────────────────────────────────────────────
let clock = 0;
const LERP_SPEED_GRID  = 0.038;
const LERP_SPEED_BREAK = 0.025;
const LERP_BREAK       = LERP_SPEED_BREAK;  // alias used inside humanized branch
const MOUSE_REPEL_R    = 9;   // world-space radius
const MOUSE_REPEL_STR  = 0.22;

function animate() {{
  requestAnimationFrame(animate);
  clock += 0.008;

  // Oscillate accent colour
  const t = (Math.sin(clock * 1.2) + 1) * 0.5;
  accentColor.setRGB(
    c0[0] + (c1[0]-c0[0]) * t,
    c0[1] + (c1[1]-c0[1]) * t,
    c0[2] + (c1[2]-c0[2]) * t
  );
  mat.color.copy(accentColor);
  glow.color.copy(accentColor);

  // Map mouse from normalised (-1..1) to approximate world coords at z=0
  const mwx = mouse.nx * (innerWidth  / innerHeight) * 30;
  const mwy = mouse.ny * 30;

  for (let i = 0; i < COUNT; i++) {{
    const ix = i*3, iy = i*3+1, iz = i*3+2;
    let px = posArr[ix], py = posArr[iy], pz = posArr[iz];
    const ph = phaseArr[i];

    if (APP_STATE === 'idle') {{
      // Slow organic drift with sine-wave perturbation
      px += velArr[ix];
      py += velArr[iy];
      pz += velArr[iz];
      // Gentle sine oscillation
      px += Math.sin(clock * 0.7 + ph)          * 0.004;
      py += Math.cos(clock * 0.5 + ph * 1.3)    * 0.004;
      // Wrap particles that escape the spread box
      if (px >  SPREAD/2) px -= SPREAD;
      if (px < -SPREAD/2) px += SPREAD;
      if (py >  SPREAD/2) py -= SPREAD;
      if (py < -SPREAD/2) py += SPREAD;
      if (pz >  10)       pz -= 20;
      if (pz < -10)       pz += 20;

    }} else if (APP_STATE === 'ai_detected') {{
      // Snap toward rigid grid position
      const gx = gridArr[ix], gy = gridArr[iy], gz = gridArr[iz];
      // Lerp toward grid with a tiny jitter for shimmer effect
      const jitter = Math.sin(clock * 12 + ph) * 0.06;
      px += (gx + jitter - px) * LERP_SPEED_GRID;
      py += (gy + jitter - py) * LERP_SPEED_GRID;
      pz += (gz - pz)          * LERP_SPEED_GRID;

    }} else if (APP_STATE === 'humanized') {{
      // Organic flocking — sine-noise spirals
      const spd  = 0.22;
      const r    = 18 + Math.sin(clock * 0.4 + ph * 0.5) * 7;
      const ang  = clock * spd + ph;
      const rise = Math.cos(clock * 0.3 + ph * 0.8) * 0.06;
      const tx   = Math.cos(ang) * r * (0.6 + 0.4 * Math.sin(ph + clock*0.2));
      const ty   = py + rise + Math.sin(clock * 0.9 + ph) * 0.05;
      const tz   = Math.sin(ang) * r * 0.5;
      px += (tx - px) * LERP_BREAK;
      py += (ty - py) * 0.012;
      pz += (tz - pz) * LERP_BREAK;
      // Soft clamp on y
      if (py >  SPREAD * 0.48) py = SPREAD * 0.48;
      if (py < -SPREAD * 0.48) py = -SPREAD * 0.48;
    }}

    // ── Mouse cursor repulsion (always active) ──────────────────────────────
    const dx  = px - mwx;
    const dy  = py - mwy;
    const d2  = dx*dx + dy*dy;
    const rr  = MOUSE_REPEL_R * MOUSE_REPEL_R;
    if (d2 < rr && d2 > 0.001) {{
      const inv = MOUSE_REPEL_STR / Math.sqrt(d2);
      px += dx * inv;
      py += dy * inv;
    }}

    posArr[ix] = px;
    posArr[iy] = py;
    posArr[iz] = pz;

    dummy.position.set(px, py, pz);
    // Face camera for billboard effect
    dummy.lookAt(camera.position);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
  }}

  mesh.instanceMatrix.needsUpdate = true;

  // Slow camera drift
  camera.position.x = Math.sin(clock * 0.08) * 4;
  camera.position.y = Math.cos(clock * 0.06) * 2;
  camera.lookAt(0, 0, 0);

  renderer.render(scene, camera);
}}

animate();

// ── Resize handler ───────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

}})();
</script>
</body>
</html>
    """
    components.html(html_code, height=0)


bg_placeholder = st.empty()
with bg_placeholder:
    render_lexicon_swarm(app_state="idle")

# --- Three.js Score Visualizer Component ---
def render_score_visualizer(score_float, prob_pct, verdict):
    html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: transparent; }}
        canvas {{ display: block; }}
        .overlay {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: white;
            font-family: 'Inter', sans-serif;
            pointer-events: none;
            width: 100%;
        }}
        .score {{
            font-size: 4.5rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 0 4px 20px rgba(0,0,0,0.6);
            letter-spacing: -0.02em;
        }}
        .verdict {{
            font-size: 1.5rem;
            font-weight: 600;
            margin: 5px 0 0 0;
            opacity: 0.9;
            text-shadow: 0 2px 10px rgba(0,0,0,0.6);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
    </style>
</head>
<body>
    <div class="overlay">
        <h1 class="score" id="score-text">{prob_pct}%</h1>
        <h2 class="verdict">{verdict}</h2>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const score = {score_float}; 
        
        // Dynamically color the text based on score
        const scoreText = document.getElementById('score-text');
        // Interpolate HSL: Green (120deg) to Red (0deg)
        const hue = (1 - score) * 120;
        scoreText.style.color = `hsl(${{hue}}, 100%, 70%)`;
        
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / 400, 0.1, 100);
        camera.position.z = 6.5;

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, 400);
        document.body.appendChild(renderer.domElement);

        const particleCount = 4000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const basePositions = new Float32Array(particleCount * 3);
        const randomOffsets = new Float32Array(particleCount * 3);

        const radius = 2.2;
        for (let i = 0; i < particleCount; i++) {{
            const phi = Math.acos(-1 + (2 * i) / particleCount);
            const theta = Math.sqrt(particleCount * Math.PI) * phi;
            
            const x = radius * Math.cos(theta) * Math.sin(phi);
            const y = radius * Math.sin(theta) * Math.sin(phi);
            const z = radius * Math.cos(phi);
            
            basePositions[i*3] = x;
            basePositions[i*3+1] = y;
            basePositions[i*3+2] = z;
            
            randomOffsets[i*3] = (Math.random() - 0.5) * 10;
            randomOffsets[i*3+1] = (Math.random() - 0.5) * 10;
            randomOffsets[i*3+2] = (Math.random() - 0.5) * 10;
        }}
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const color = new THREE.Color();
        color.setHSL((1 - score) * 0.33, 1.0, 0.5);
        
        const material = new THREE.PointsMaterial({{
            color: color,
            size: 0.04,
            transparent: true,
            opacity: 0.8
        }});
        
        const particles = new THREE.Points(geometry, material);
        scene.add(particles);

        let time = 0;
        function animate() {{
            requestAnimationFrame(animate);
            time += 0.01;
            
            const posAttribute = geometry.attributes.position;
            const array = posAttribute.array;
            
            // Turbulence scales non-linearly with score
            const turbulence = Math.pow(score, 2.5) * 1.5; 
            
            for (let i = 0; i < particleCount; i++) {{
                const bx = basePositions[i*3];
                const by = basePositions[i*3+1];
                const bz = basePositions[i*3+2];
                
                const rx = randomOffsets[i*3];
                const ry = randomOffsets[i*3+1];
                const rz = randomOffsets[i*3+2];
                
                const dx = rx * Math.sin(time * 1.5 + rx);
                const dy = ry * Math.cos(time * 2.0 + ry);
                const dz = rz * Math.sin(time * 1.2 + rz);
                
                array[i*3] = bx + dx * turbulence;
                array[i*3+1] = by + dy * turbulence;
                array[i*3+2] = bz + dz * turbulence;
            }}
            
            posAttribute.needsUpdate = true;
            
            const rotationSpeed = 0.002 + (score * 0.01);
            particles.rotation.y += rotationSpeed;
            particles.rotation.x += rotationSpeed * 0.5;
            
            renderer.render(scene, camera);
        }}
        animate();
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / 400;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, 400);
        }});
    </script>
</body>
</html>
    """
    components.html(html_code, height=400)

# (render_humanizer_visualizer and set_humanizer_state have been replaced
#  by render_lexicon_swarm with app_state='humanized')

# Main UI layout
st.markdown('<div class="main-title">AI Detector & Humanizer 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze your text to check if it was written by AI, or rewrite it to sound more natural.</div>', unsafe_allow_html=True)

visualizer_placeholder = st.empty()

# Input section
text_input = st.text_area(
    "Paste your content here:",
    height=300,
    placeholder="Enter the text you want to analyze or humanize...",
    label_visibility="collapsed"
)

# Buttons layout
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    detect_button = st.button("🔍 Detect AI", use_container_width=True)

with col2:
    humanize_button = st.button("✨ Humanize Text", use_container_width=True)

# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------
if detect_button:
    text_content = text_input.strip()
    if text_content:
        if len(text_content) < 600:
            st.warning(f"⚠️ Please enter at least 600 characters for an accurate AI detection (currently {len(text_content)} characters).")
        elif not is_model_available():
            st.error(
                "⚠️ No trained model found. Run `python train_model.py` first "
                "to train the classifier."
            )
        else:
            with bg_placeholder:
                render_lexicon_swarm(app_state="ai_detected")
            with st.spinner("Running classifier analysis…"):
                bundle = get_model_bundle()
                result = classify_text(text_input, bundle=bundle)
            # Keep ai_detected state after analysis so the background
            # visually reflects the detection result while the score is shown
            with bg_placeholder:
                render_lexicon_swarm(app_state="ai_detected")

            prob_pct = round(result["probability"] * 100, 1)
            score_float = result["probability"]
            fp_prob = result.get("false_positive_probability")
            label = result["label"]
            verdict = result["verdict"]

            # Render the 3D visualizer above the text area
            with visualizer_placeholder:
                render_score_visualizer(score_float, prob_pct, verdict)

            # ---- AI DETECTED (only when probability > 35%) ----
            if label == 1 and result["probability"] > 0.35:
                fp_pct = round(fp_prob * 100, 2) if fp_prob is not None and fp_prob >= 0 else None

                # ---- False Positive Probability ----
                if fp_pct is not None:
                    # Color-code: low FP = green (strong signal), high FP = amber (weak signal)
                    if fp_pct <= 2:
                        fp_color = "#4ade80"  # green-400
                        fp_bg = "rgba(74, 222, 128, 0.1)"
                        fp_border = "rgba(74, 222, 128, 0.3)"
                        fp_icon = "✅"
                        fp_note = "Very low false-positive risk. The AI signal is strong."
                    elif fp_pct <= 10:
                        fp_color = "#fbbf24"  # amber-400
                        fp_bg = "rgba(251, 191, 36, 0.1)"
                        fp_border = "rgba(251, 191, 36, 0.3)"
                        fp_icon = "⚠️"
                        fp_note = "Moderate false-positive risk. Consider reviewing manually."
                    else:
                        fp_color = "#f87171"  # red-400
                        fp_bg = "rgba(248, 113, 113, 0.1)"
                        fp_border = "rgba(248, 113, 113, 0.3)"
                        fp_icon = "🔴"
                        fp_note = "High false-positive risk. This could easily be a human who writes very predictably."

                    st.markdown(f"""
                    <div style="
                        background: {fp_bg};
                        border: 2px solid {fp_border};
                        border-radius: 12px;
                        padding: 1.5rem 2rem;
                        margin: 0.75rem 0 1.5rem 0;
                    ">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                            <span style="font-size: 1.3rem;">{fp_icon}</span>
                            <span style="
                                font-size: 1rem;
                                font-weight: 700;
                                color: {fp_color};
                                text-transform: uppercase;
                                letter-spacing: 0.05em;
                            ">
                                False Positive Probability
                            </span>
                        </div>
                        <div style="
                            font-size: 2rem;
                            font-weight: 800;
                            color: {fp_color};
                            margin-bottom: 0.5rem;
                        ">
                            {fp_pct}%
                        </div>
                        <div style="
                            font-size: 0.9rem;
                            color: #d1d5db;
                            line-height: 1.5;
                        ">
                            There is a <strong>{fp_pct}%</strong> chance this text was actually written
                            by a human who writes predictably. In our training dataset,
                            <strong>{fp_pct}%</strong> of known human-written texts scored at or above
                            {prob_pct}% AI confidence.
                        </div>
                        <div style="
                            font-size: 0.85rem;
                            color: #9ca3af;
                            margin-top: 0.5rem;
                            font-style: italic;
                        ">
                            {fp_note}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(
                        "ℹ️ False Positive Probability is unavailable. "
                        "Re-train the model to enable calibration data."
                    )

            # ---- Feature breakdown (collapsible) ----
            with st.expander("📊 Feature Breakdown"):
                features = result["features"]
                feat_cols = st.columns(3)
                nice_names = {
                    "avg_sentence_length": ("📏 Avg Sentence Length", "words"),
                    "sentence_length_std": ("📐 Sentence Burstiness (StdDev)", ""),
                    "vocab_richness": ("📖 Vocabulary Richness (TTR)", ""),
                    "stopword_freq": ("🔤 Stop-word Frequency", ""),
                    "sentence_count": ("📝 Sentence Count", ""),
                    "avg_word_length": ("🔡 Avg Word Length", "chars"),
                    "punctuation_ratio": ("✏️ Punctuation Ratio", ""),
                    "flesch_kincaid_grade": ("🎓 Flesch-Kincaid Grade", ""),
                    "paragraph_symmetry": ("⚖️ Paragraph Symmetry", ""),
                    "trope_count": ("🤖 LLM Trope Count", "phrases"),
                    "gemini_predictability": ("🧠 Gemini Predictability", "/1.0"),
                    "gemini_trope_presence": ("🧠 Gemini Trope Score", "/1.0"),
                }
                for idx, (key, (label, unit)) in enumerate(nice_names.items()):
                    val = features.get(key)
                    if val is None:
                        display_val = "N/A"
                    elif isinstance(val, float) and val < 1:
                        display_val = f"{val:.3f}{unit}"
                    else:
                        display_val = f"{val:.1f}{unit}" if isinstance(val, float) else f"{val}{unit}"
                    feat_cols[idx % 3].metric(label, display_val)

    else:
        st.warning("Please enter some text to analyze.")

# ---------------------------------------------------------------------------
# Humanize results
# ---------------------------------------------------------------------------
if humanize_button:
    text_content = text_input.strip()
    if text_content:
        if len(text_content) < 600:
            st.warning(f"⚠️ Please enter at least 600 characters for an accurate analysis and humanization (currently {len(text_content)} characters).")
        else:
            # --- Conditional execution: skip humanization if text already reads as human ---
            skip_humanize = False
            if is_model_available():
                with st.spinner("Pre-checking AI detection score…"):
                    bundle = get_model_bundle()
                    pre_check = classify_text(text_input, bundle=bundle)
                pre_prob = pre_check["probability"]
                if pre_prob <= 0.35:
                    skip_humanize = True
                    st.info(
                        "🟢 **This text already reads as human-written. "
                        "No humanization required.**\n\n"
                        f"AI probability is only **{round(pre_prob * 100, 1)}%** "
                        "(below the 40% threshold)."
                    )

            if not skip_humanize:
                # Switch background to humanized swirl immediately
                with bg_placeholder:
                    render_lexicon_swarm(app_state="humanized")

                with st.spinner("Humanizing text… this may take a few seconds"):
                    rewritten_text, error = humanize_text(text_input)

                if rewritten_text is not None:

                    st.subheader("✨ Humanized Text")
                    # Styled output container
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #022c22 0%, #064e3b 100%);
                        border: 2px solid #059669;
                        border-radius: 16px;
                        padding: 2rem 2.5rem;
                        margin: 1rem 0;
                        font-size: 1.05rem;
                        line-height: 1.8;
                        color: #ecfdf5;
                        box-shadow: 0 4px 24px rgba(16, 185, 129, 0.15);
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    ">{rewritten_text}</div>
                    """, unsafe_allow_html=True)

                    # Copy-friendly fallback in a collapsed expander
                    with st.expander("📋 Copy-friendly plain text"):
                        st.text_area(
                            "Humanized output",
                            value=rewritten_text,
                            height=250,
                            label_visibility="collapsed",
                        )
                else:
                    st.error(f"❌ **Humanization failed.** {error}")
    else:
        st.warning("Please enter some text to humanize.")

# Default idle visualizer — score widget area is empty when nothing has run
if not detect_button and not humanize_button:
    pass  # visualizer_placeholder stays empty; background is already idle
