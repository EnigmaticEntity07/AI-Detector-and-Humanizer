# AI Text Detector & Humanizer 🛡️✍️

A production-grade web application built to strictly detect AI-generated text and intelligently humanize it. This tool prioritizes detection transparency by leaning toward false positives (strict AI flagging) while actively calculating and displaying the **False Positive Probability** to ensure fairness.

# 🚀 Features

* **Strict AI Detection:** Powered by a Gradient Boosting classifier calibrated with Isotonic Regression to achieve high-confidence (99%) detection scoring.
* **Transparent Probability:** Displays the statistical margin of error (False Positive Probability) for every detection run.
* **Intelligent Humanizer:** Leverages Google's Gemini Pro via a dataset-driven, few-shot prompting architecture to rewrite text, stripping out modern LLM tropes while maintaining flawless grammar and original meaning.
* **Premium UI/UX:** Built with Streamlit, featuring a side-by-side comparative layout, dynamic word-count enforcement (600-word minimum), and an immersive, responsive 3D "Lexicon Swarm" background rendered in Three.js.

# 🏗️ Architecture & Tech Stack

* **Frontend:** Streamlit, Custom CSS, HTML/JS injection for Three.js 3D animations.
* **Backend Inference:** Python, Scikit-learn (Gradient Boosting, Isotonic Regression).
* **LLM Integration:** Google Generative AI (Gemini Pro API).
* **Development/Deployment:** Google Antigravity (Agentic IDE), Railway (Port 8501).
* **Data:** Calibrated using human vs. AI Kaggle datasets for authentic baseline comparisons.

# 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/EnigmaticEntity7/AI-Detector-and-Humanizer.git](https://github.com/EnigmaticEntity7/AI-Detector-and-Humanizer.git)
   cd AI-Detector-and-Humanizer
