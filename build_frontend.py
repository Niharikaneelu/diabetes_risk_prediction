import base64
from pathlib import Path

def get_b64(rel_path):
    p = Path(rel_path)
    if p.exists():
        return 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode('utf-8')
    return ''

# Result plots
img_beeswarm = get_b64('results/shap_summary_beeswarm.png')
img_shap_bar = get_b64('results/shap_summary_bar.png')
img_roc = get_b64('results/roc_curve_comparison.png')
img_cm_xgb = get_b64('results/confusion_matrix_xgboost.png')
img_cm_rf = get_b64('results/confusion_matrix_random_forest.png')
img_metrics = get_b64('results/metric_comparison.png')
img_rf_imp = get_b64('results/random_forest_feature_importance.png')
img_xgb_imp = get_b64('results/xgboost_feature_importance.png')
img_imp_comp = get_b64('results/feature_importance_comparison.png')

# Dependence plots
img_dep_gluc = get_b64('results/shap_dependence_glucose.png')
img_dep_bmi = get_b64('results/shap_dependence_bmi.png')
img_dep_age = get_b64('results/shap_dependence_age.png')
img_dep_ped = get_b64('results/shap_dependence_diabetespedigreefunction.png')
img_dep_preg = get_b64('results/shap_dependence_pregnancies.png')

html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
  <title>Diabetes Risk AI — XGBoost & Random Forest with SHAP Explainability</title>
  <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/lenis@1.1.18/dist/lenis.min.js"></script>
  <style>
    @layer base {{
      html {{
        font-size: 18px;
        zoom: 1.08;
        scroll-behavior: smooth;
      }}
      html, body {{
        margin: 0;
        padding: 0;
        background-color: #faf8ff;
        font-family: 'Inter', sans-serif;
        color: #131b2e;
        -webkit-font-smoothing: antialiased;
        scroll-behavior: smooth;
        -webkit-overflow-scrolling: touch;
      }}
      body {{ overscroll-behavior-y: contain; }}
      main > :first-child {{ margin-top: 0 !important; }}
      main > :last-child {{ margin-bottom: 0 !important; }}
    }}
    
    /* Smooth Momentum Scrolling */
    html.lenis, html.lenis body {{
      height: auto;
    }}
    .lenis.lenis-smooth {{
      scroll-behavior: auto !important;
    }}
    .lenis.lenis-smooth [data-lenis-prevent] {{
      overscroll-behavior: contain;
    }}
    .lenis.lenis-stopped {{
      overflow: hidden;
    }}
    .lenis.lenis-scrolling iframe {{
      pointer-events: none;
    }}
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #f2f3ff; }}
    ::-webkit-scrollbar-thumb {{ background: #c3c6d7; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #737686; }}

    /* Remove number input spinner arrows */
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {{
      -webkit-appearance: none;
      margin: 0;
    }}
    input[type="number"] {{
      -moz-appearance: textfield;
      appearance: textfield;
    }}

    .font-code-num {{ font-variant-numeric: tabular-nums; font-family: 'Inter', monospace; }}
    .transition-smooth {{ transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }}
    
    @media print {{
      header, footer, #statusToast, .no-print {{ display: none !important; }}
      body {{ background: white !important; color: black !important; }}
    }}
  </style>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      darkMode: "class",
      theme: {{
        extend: {{
          colors: {{
            "background": "#faf8ff",
            "on-background": "#131b2e",
            "surface": "#faf8ff",
            "surface-dim": "#d2d9f4",
            "surface-bright": "#faf8ff",
            "surface-container-lowest": "#ffffff",
            "surface-container-low": "#f2f3ff",
            "surface-container": "#eaedff",
            "surface-container-high": "#e2e7ff",
            "surface-container-highest": "#dae2fd",
            "on-surface": "#131b2e",
            "on-surface-variant": "#434655",
            "inverse-surface": "#283044",
            "inverse-on-surface": "#eef0ff",
            "outline": "#737686",
            "outline-variant": "#c3c6d7",
            "surface-tint": "#0053db",
            "primary": "#004ac6",
            "on-primary": "#ffffff",
            "primary-container": "#2563eb",
            "on-primary-container": "#eeefff",
            "primary-fixed": "#dbe1ff",
            "primary-fixed-dim": "#b4c5ff",
            "on-primary-fixed": "#00174b",
            "on-primary-fixed-variant": "#003ea8",
            "secondary": "#006a61",
            "on-secondary": "#ffffff",
            "secondary-container": "#86f2e4",
            "on-secondary-container": "#006f66",
            "secondary-fixed": "#89f5e7",
            "secondary-fixed-dim": "#6bd8cb",
            "on-secondary-fixed": "#00201d",
            "tertiary": "#006329",
            "on-tertiary": "#ffffff",
            "tertiary-container": "#007f36",
            "on-tertiary-container": "#c7ffca",
            "tertiary-fixed": "#7ffc97",
            "tertiary-fixed-dim": "#62df7d",
            "error": "#ba1a1a",
            "on-error": "#ffffff",
            "error-container": "#ffdad6",
            "on-error-container": "#93000a",
            "surface-variant": "#dae2fd"
          }},
          borderRadius: {{
            "DEFAULT": "0.25rem",
            "lg": "0.5rem",
            "xl": "0.75rem",
            "2xl": "1rem",
            "full": "9999px"
          }},
          spacing: {{
            "space-xs": "0.25rem",
            "space-sm": "0.5rem",
            "space-md": "1rem",
            "space-lg": "1.5rem",
            "space-xl": "2.5rem",
            "gutter": "1.5rem",
            "margin": "2rem"
          }},
          fontSize: {{
            "display-lg": ["3.4rem", {{"lineHeight": "3.9rem", "letterSpacing": "-0.025em", "fontWeight": "700"}}],
            "headline-lg": ["2.35rem", {{"lineHeight": "2.85rem", "letterSpacing": "-0.02em", "fontWeight": "700"}}],
            "headline-md": ["1.75rem", {{"lineHeight": "2.25rem", "letterSpacing": "-0.015em", "fontWeight": "600"}}],
            "headline-sm": ["1.4rem", {{"lineHeight": "1.9rem", "letterSpacing": "-0.01em", "fontWeight": "600"}}],
            "body-lg": ["1.25rem", {{"lineHeight": "1.85rem", "fontWeight": "400"}}],
            "body-md": ["1.1rem", {{"lineHeight": "1.65rem", "fontWeight": "400"}}],
            "body-sm": ["0.95rem", {{"lineHeight": "1.45rem", "fontWeight": "400"}}],
            "label-lg": ["1rem", {{"lineHeight": "1.45rem", "letterSpacing": "0.01em", "fontWeight": "600"}}],
            "label-md": ["0.875rem", {{"lineHeight": "1.25rem", "letterSpacing": "0.02em", "fontWeight": "500"}}],
            "label-sm": ["0.775rem", {{"lineHeight": "1.05rem", "letterSpacing": "0.03em", "fontWeight": "600"}}],
            "code-num": ["1.35rem", {{"lineHeight": "1.6rem", "letterSpacing": "-0.01em", "fontWeight": "700"}}]
          }}
        }}
      }}
    }};
  </script>
</head>
<body class="bg-surface text-on-surface antialiased selection:bg-primary-fixed selection:text-on-primary-fixed min-h-screen flex flex-col">

  <!-- Top Sticky Navigation Bar -->
  <header class="fixed top-0 w-full z-50 bg-surface-container-lowest/90 backdrop-blur-xl shadow-[0_1px_8px_rgba(0,0,0,0.04)] no-print" style="transform: translateZ(0); will-change: transform;">
    <div class="h-16 max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between gap-space-sm sm:gap-space-md">
      
      <!-- Left Logo & Title -->
      <div class="flex items-center gap-space-md">
        <div class="flex items-center gap-space-sm cursor-pointer" onclick="switchTab('risk-prediction')">
          <div class="w-8 h-8 rounded-lg bg-primary text-on-primary flex items-center justify-center font-bold text-lg shadow-sm">
            <span class="material-symbols-outlined text-[20px]">vital_signs</span>
          </div>
          <span class="text-headline-sm text-on-surface tracking-tight font-bold hidden sm:inline">Diabetes Risk AI</span>
        </div>



        <!-- Navigation Tabs -->
        <nav class="hidden xl:inline-flex items-center gap-space-xs p-1 bg-surface-container-low rounded-xl w-fit shrink-0" id="navTabs">
          <button id="nav-risk-prediction" class="px-space-md py-1.5 rounded-lg transition-all bg-primary text-on-primary shadow-sm font-semibold text-label-md" onclick="switchTab('risk-prediction')">
            Risk Assessment
          </button>
          <button id="nav-prediction-results" class="flex items-center gap-1.5 px-space-md py-1.5 rounded-lg text-label-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-all" onclick="switchTab('prediction-results')">
            <span>Prediction Results</span>
            <span id="navResultsBadge" class="hidden px-1.5 py-0.5 rounded bg-secondary-container text-on-secondary-container text-[11px] font-bold">Ready</span>
          </button>
          <button id="nav-explainability-xai" class="flex items-center gap-1.5 px-space-md py-1.5 rounded-lg text-label-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-all" onclick="switchTab('explainability-xai')">
            <span id="shapNavBadge" class="px-1.5 py-0.5 rounded bg-primary-fixed text-primary text-[11px] font-bold">SHAP</span>
            <span>SHAP Explainability (XAI)</span>
          </button>
          <button id="nav-model-benchmark-and-insights" class="px-space-md py-1.5 rounded-lg text-label-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-all" onclick="switchTab('model-benchmark-and-insights')">
            Model Benchmark (RF vs XGB)
          </button>
        </nav>
      </div>

      <!-- Right Action Items -->
      <div class="flex items-center gap-space-sm sm:gap-space-md"></div>
    </div>
  </header>

  <!-- Interactive State Management and Notification Toast -->
  <div class="fixed bottom-6 right-6 z-50 transform translate-y-20 opacity-0 transition-all duration-300 pointer-events-none flex items-center gap-space-sm bg-inverse-surface text-inverse-on-surface px-space-md py-space-sm rounded-xl shadow-xl text-label-md" id="statusToast">
    <span class="material-symbols-outlined text-[18px] text-secondary-fixed">check_circle</span>
    <span id="statusToastText">Sample cohort data loaded</span>
  </div>

  <!-- MAIN VIEWPORT -->
  <main class="w-full pt-16 bg-surface flex-1 flex flex-col">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 pt-6 pb-6 w-full">

      <!-- ========================================================================= -->
      <!-- TAB 1: RISK PREDICTION (MAIN WORKSTATION)                                 -->
      <!-- ========================================================================= -->
      <div id="tab-content-risk-prediction" class="flex flex-col w-full">

        <!-- MODEL SWITCHER BANNER STRIP -->
        <div class="w-full bg-surface-container-lowest rounded-2xl p-4 mb-4 shadow-sm border border-outline-variant/30 flex flex-wrap items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <span class="w-9 h-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center font-bold">
              <span class="material-symbols-outlined text-[22px]">tune</span>
            </span>
            <div>
              <span class="text-label-sm uppercase tracking-wider text-outline font-semibold block">Active Predictive Architecture</span>
              <span class="text-label-lg font-bold text-on-surface" id="bannerModelTitle">Random Forest Classifier (Selected Best • Threshold: 0.48)</span>
            </div>
          </div>
          <!-- Toggle Buttons -->
          <div class="flex items-center gap-2 bg-surface-container-low p-1 rounded-xl border border-outline-variant/30">
            <button id="toggleBtnRf" class="px-4 py-1.5 rounded-lg text-label-md font-bold transition-all bg-primary text-on-primary shadow-xs" onclick="changeModel('random_forest')">
              Random Forest (Best)
            </button>
            <button id="toggleBtnXgb" class="px-4 py-1.5 rounded-lg text-label-md font-medium text-on-surface-variant hover:text-on-surface transition-all" onclick="changeModel('xgboost')">
              XGBoost
            </button>
          </div>
        </div>

        <!-- Hero Header Area -->
        <section class="relative w-full rounded-2xl bg-surface-container-lowest p-5 lg:p-6 mb-6 overflow-hidden shadow-sm">
          <div class="absolute -right-24 -top-24 w-96 h-96 rounded-full bg-surface-variant/40 blur-3xl pointer-events-none"></div>
          <div class="absolute right-1/3 -bottom-20 w-80 h-80 rounded-full bg-secondary-container/20 blur-2xl pointer-events-none"></div>
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center relative z-10">
            <div class="lg:col-span-8 flex flex-col gap-2.5">
              <h1 class="text-headline-lg text-on-surface tracking-tight font-bold">
                Understand Your Diabetes Risk
              </h1>
              <p class="text-body-md text-on-surface-variant max-w-2xl">
                Statistical risk estimation combining biometric clinical parameters with transparent, feature-level attribution calculated via Shapley additive explanations (SHAP).
              </p>
              
              <div class="flex items-center gap-2 mt-3 text-label-sm text-outline">
                <span class="material-symbols-outlined text-[16px] text-primary">touch_app</span>
                <span>Select a clinical test scenario below or enter custom biometrics into the form</span>
              </div>
            </div>

            <!-- Right Side Abstract Healthcare Neural Graph -->
            <div class="lg:col-span-4 flex justify-center lg:justify-end">
              <div class="relative w-full max-w-sm h-48 bg-surface-container-low rounded-xl p-4 flex flex-col justify-between overflow-hidden shadow-inner border border-outline-variant/20">
                <div class="flex items-center justify-between">
                  <span class="text-label-sm text-on-surface-variant font-medium flex items-center gap-1">
                    <span class="material-symbols-outlined text-[16px] text-primary">hub</span>
                    Biometric Feature Tensor
                  </span>
                  <span class="text-label-sm text-secondary bg-surface px-2 py-0.5 rounded-full font-medium" id="heroModelBadge">Random Forest Active</span>
                </div>
                <!-- Abstract SVG Biological Neural Mesh -->
                <svg class="w-full h-24 overflow-visible" fill="none" viewBox="0 0 320 90" xmlns="http://www.w3.org/2000/svg">
                  <path class="opacity-40" d="M10 45 C 50 15, 80 75, 120 40 C 160 10, 200 80, 240 35 C 270 5, 290 55, 310 45" stroke="#004ac6" stroke-dasharray="4 4" stroke-linecap="round" stroke-width="2"></path>
                  <path d="M10 50 C 45 70, 90 20, 135 55 C 180 85, 220 20, 265 60 C 290 80, 305 40, 315 48" stroke="#006a61" stroke-linecap="round" stroke-width="2.5"></path>
                  <!-- Neural Graph Nodes -->
                  <g class="fill-surface stroke-primary" stroke-width="2">
                    <circle class="animate-pulse" cx="45" cy="65" r="4"></circle>
                    <circle cx="120" cy="40" r="4"></circle>
                    <circle cx="160" cy="18" r="5"></circle>
                    <circle cx="220" cy="28" r="4"></circle>
                    <circle class="stroke-error fill-surface" cx="265" cy="60" r="5"></circle>
                    <circle cx="310" cy="46" r="3.5"></circle>
                  </g>
                  <!-- Vertical Data Scan Line -->
                  <line opacity="0.3" stroke="#0053db" stroke-dasharray="2 2" stroke-width="1.5" x1="160" x2="160" y1="0" y2="90"></line>
                  <text class="text-[9px]" fill="#434655" x="165" y="14" font-weight="500">Base Expected Value</text>
                </svg>
                <div class="flex items-center justify-between text-label-sm text-on-surface-variant pt-2">
                  <span class="flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-primary" id="legendDot"></span><span id="legendText">Random Forest Predictor</span></span>
                  <span class="flex items-center gap-1 cursor-pointer font-semibold text-secondary hover:underline" onclick="scrollToShap()"><span class="w-1.5 h-1.5 rounded-full bg-secondary"></span>SHAP Attribution</span>
                  <span class="text-tertiary font-semibold">Ready</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- ========================================================================= -->
        <!-- CLINICAL TEST CASES (Clickable Scenario Cards)                             -->
        <!-- ========================================================================= -->
        <section class="w-full mb-8">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3.5">
            <div class="flex items-center gap-2.5">
              <span class="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center font-bold">
                <span class="material-symbols-outlined text-[20px]">science</span>
              </span>
              <div>
                <h2 class="text-headline-sm text-on-surface font-bold text-[18px] leading-tight">Clinical Test Cases</h2>
                <span class="text-body-sm text-on-surface-variant font-normal">Predefined benchmark scenarios to evaluate model risk predictions and local SHAP explanations</span>
              </div>
            </div>
            <span class="text-label-sm text-outline font-medium flex items-center gap-1 self-start sm:self-auto bg-surface-container-low px-3 py-1 rounded-full border border-outline-variant/30">
              <span class="material-symbols-outlined text-[16px] text-primary">touch_app</span> Click card to select
            </span>
          </div>

          <!-- 3-Column Responsive Grid -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4" id="testCaseGrid">
            
            <!-- Card 1: Test 01 -->
            <div id="testCard-01" class="test-case-card group relative p-4 rounded-xl border border-outline-variant/40 bg-surface-container-lowest hover:border-primary/50 hover:shadow-md transition-all cursor-pointer flex flex-col justify-between gap-3" onclick="selectTestCase('test-01')" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();selectTestCase('test-01');}}" role="button" tabindex="0" aria-pressed="false" title="Click to select Test 01 scenario">
              <div class="flex items-start justify-between gap-2">
                <div class="flex items-center gap-2">
                  <span class="test-badge-num px-2 py-0.5 rounded-md text-[11px] font-bold font-mono tracking-wider bg-surface-container text-outline border border-outline-variant/40">TEST 01</span>
                  <h3 class="text-label-lg font-bold text-on-surface">Elevated Glycemia</h3>
                </div>
                <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-error-container text-on-error-container shrink-0">
                  <span class="w-1.5 h-1.5 rounded-full bg-error"></span>
                  Elevated Risk Profile
                </span>
              </div>
              <p class="text-body-sm text-on-surface-variant leading-relaxed">
                Case A: Marked post-challenge hyperglycemia with elevated body mass index and familial history.
              </p>
              <div class="pt-2.5 border-t border-outline-variant/30 flex items-center justify-between text-label-sm">
                <span class="font-code-num text-on-surface-variant text-[12px] font-medium">Glucose 154 • BMI 31.4 • Age 48</span>
                <span class="text-outline font-medium flex items-center gap-1 text-[12px] group-hover:text-primary">
                  <span class="material-symbols-outlined text-[16px] test-card-status-icon">radio_button_unchecked</span>
                  <span class="test-card-status-text">Select</span>
                </span>
              </div>
            </div>

            <!-- Card 2: Test 02 -->
            <div id="testCard-02" class="test-case-card group relative p-4 rounded-xl border border-outline-variant/40 bg-surface-container-lowest hover:border-primary/50 hover:shadow-md transition-all cursor-pointer flex flex-col justify-between gap-3" onclick="selectTestCase('test-02')" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();selectTestCase('test-02');}}" role="button" tabindex="0" aria-pressed="false" title="Click to select Test 02 scenario">
              <div class="flex items-start justify-between gap-2">
                <div class="flex items-center gap-2">
                  <span class="test-badge-num px-2 py-0.5 rounded-md text-[11px] font-bold font-mono tracking-wider bg-surface-container text-outline border border-outline-variant/40">TEST 02</span>
                  <h3 class="text-label-lg font-bold text-on-surface">Normoglycemic Baseline</h3>
                </div>
                <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-tertiary-container/50 text-on-tertiary-fixed-variant shrink-0">
                  <span class="w-1.5 h-1.5 rounded-full bg-tertiary"></span>
                  Lower Risk Profile
                </span>
              </div>
              <p class="text-body-sm text-on-surface-variant leading-relaxed">
                Case B: Normal fasting &amp; post-challenge glucose profile with healthy biometric markers.
              </p>
              <div class="pt-2.5 border-t border-outline-variant/30 flex items-center justify-between text-label-sm">
                <span class="font-code-num text-on-surface-variant text-[12px] font-medium">Glucose 88 • BMI 21.8 • Age 28</span>
                <span class="text-outline font-medium flex items-center gap-1 text-[12px] group-hover:text-primary">
                  <span class="material-symbols-outlined text-[16px] test-card-status-icon">radio_button_unchecked</span>
                  <span class="test-card-status-text">Select</span>
                </span>
              </div>
            </div>

            <!-- Card 3: Test 03 -->
            <div id="testCard-03" class="test-case-card group relative p-4 rounded-xl border border-outline-variant/40 bg-surface-container-lowest hover:border-primary/50 hover:shadow-md transition-all cursor-pointer flex flex-col justify-between gap-3" onclick="selectTestCase('test-03')" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();selectTestCase('test-03');}}" role="button" tabindex="0" aria-pressed="false" title="Click to select Test 03 scenario">
              <div class="flex items-start justify-between gap-2">
                <div class="flex items-center gap-2">
                  <span class="test-badge-num px-2 py-0.5 rounded-md text-[11px] font-bold font-mono tracking-wider bg-surface-container text-outline border border-outline-variant/40">TEST 03</span>
                  <h3 class="text-label-lg font-bold text-on-surface">Impaired Fasting Glucose</h3>
                </div>
                <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-100 text-amber-900 shrink-0">
                  <span class="w-1.5 h-1.5 rounded-full bg-amber-600"></span>
                  Moderate Risk Profile
                </span>
              </div>
              <p class="text-body-sm text-on-surface-variant leading-relaxed">
                Case C: Borderline glycemic markers near the decision boundary with intermediate adiposity.
              </p>
              <div class="pt-2.5 border-t border-outline-variant/30 flex items-center justify-between text-label-sm">
                <span class="font-code-num text-on-surface-variant text-[12px] font-medium">Glucose 126 • BMI 28.2 • Age 41</span>
                <span class="text-outline font-medium flex items-center gap-1 text-[12px] group-hover:text-primary">
                  <span class="material-symbols-outlined text-[16px] test-card-status-icon">radio_button_unchecked</span>
                  <span class="test-card-status-text">Select</span>
                </span>
              </div>
            </div>

          </div>
        </section>

        <!-- Assessment Error Banner (Shown if inference fails) -->
        <div id="assessmentErrorBanner" class="hidden w-full mb-6 p-4 rounded-xl bg-error-container text-on-error-container border border-error/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-sm">
          <div class="flex items-center gap-3">
            <span class="material-symbols-outlined text-[24px] text-error shrink-0">error</span>
            <div>
              <h4 class="text-label-md font-bold text-on-error-container" id="errorBannerTitle">Inference Service Unreachable</h4>
              <p class="text-body-sm text-on-error-container/90 mt-0.5" id="errorBannerMsg">Backend Error: Could not connect to inference service.</p>
            </div>
          </div>
          <div class="flex items-center gap-2 self-end sm:self-center shrink-0">
            <button class="px-3.5 py-1.5 rounded-lg bg-error text-on-error hover:bg-error/90 text-label-sm font-semibold flex items-center gap-1 shadow-xs transition-colors" onclick="calculateRisk()" type="button">
              <span class="material-symbols-outlined text-[16px]">refresh</span>
              Retry Inference
            </button>
            <button class="px-2.5 py-1.5 rounded-lg text-label-sm font-medium hover:bg-error-container/80 transition-colors" onclick="dismissErrorBanner()" type="button">
              Dismiss
            </button>
          </div>
        </div>

        <!-- Main Workstation: Patient Information & Clinical Measurements Form -->
        <div class="w-full bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm flex flex-col gap-6 min-w-0 border border-outline-variant/30 mb-8">

            <!-- Patient Information Introduction -->
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-outline-variant/20 w-full">
              <div>
                <h2 class="text-headline-sm text-on-surface font-bold text-[22px]">Patient Information</h2>
                <p class="text-body-sm text-on-surface-variant mt-0.5">Enter standard physiological markers to evaluate individualized metabolic risk.</p>
              </div>
              <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-surface-container text-on-surface-variant text-label-sm shrink-0 self-start sm:self-auto font-medium">
                <span class="material-symbols-outlined text-[15px]">checklist</span>
                8 Features Required
              </span>
            </div>

            <form class="flex flex-col gap-6 w-full min-w-0" id="predictionForm" onsubmit="event.preventDefault(); calculateRisk();">
              
              <!-- Section 1: Demographic Profile -->
              <div class="flex flex-col gap-3 w-full">
                <div class="flex items-center gap-2">
                  <span class="w-6 h-6 rounded-md bg-primary-fixed/60 text-primary flex items-center justify-center text-label-sm font-bold">1</span>
                  <h3 class="text-label-lg text-on-surface font-semibold text-[17px]">Demographic Profile</h3>
                </div>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-5 w-full">
                  <!-- Age Field -->
                  <div class="flex flex-col gap-1.5 w-full min-w-0">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputAge">
                        Age
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Patient chronological age in completed years">info</span>
                      </label>
                      <span class="text-label-sm text-outline">years</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputAge" max="100" min="18" placeholder="e.g. 48" type="number" value="" required/>
                  </div>
                  <!-- Pregnancies Field -->
                  <div class="flex flex-col gap-1.5 w-full min-w-0">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputPregnancies">
                        Pregnancies
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Total number of completed pregnancies / parity">info</span>
                      </label>
                      <span class="text-label-sm text-outline">count</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputPregnancies" max="20" min="0" placeholder="e.g. 2" type="number" value="" required/>
                  </div>
                </div>
              </div>

              <!-- Section 2: Clinical & Laboratory Measurements -->
              <div class="flex flex-col gap-3.5 w-full">
                <div class="flex items-center gap-2">
                  <span class="w-6 h-6 rounded-md bg-primary-fixed/60 text-primary flex items-center justify-center text-label-sm font-bold">2</span>
                  <h3 class="text-label-lg text-on-surface font-semibold text-[17px]">Clinical &amp; Laboratory Measurements</h3>
                </div>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-5 w-full">
                  <!-- Glucose Field (Full Width Highlighted Measurement Row) -->
                  <div class="sm:col-span-2 p-3.5 sm:p-4 rounded-xl bg-surface-container-low/70 border border-outline-variant/30 flex flex-col gap-2 w-full min-w-0">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-semibold flex items-center gap-1" for="inputGlucose">
                        Glucose (2-hr OGTT)
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Plasma glucose concentration at 2 hours in an oral glucose tolerance test">help</span>
                      </label>
                      <span class="text-label-sm text-on-surface-variant font-medium">mg/dL</span>
                    </div>
                    <div class="flex flex-col sm:flex-row sm:items-center gap-3 w-full">
                      <div class="relative flex-1 min-w-0">
                        <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container transition-colors" id="inputGlucose" max="300" min="40" placeholder="e.g. 154" type="number" value="" required/>
                      </div>
                      <div class="flex flex-col sm:items-end text-left sm:text-right shrink-0">
                        <span class="hidden" id="glucoseBadge"></span>
                        <span class="text-[10px] text-outline mt-0.5">Ref: Normal &lt;100 mg/dL</span>
                      </div>
                    </div>
                  </div>

                  <!-- Blood Pressure Field -->
                  <div class="flex flex-col gap-1.5 w-full min-w-0">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputBP">
                        Blood Pressure
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Diastolic blood pressure (mm Hg)">info</span>
                      </label>
                      <span class="text-label-sm text-outline">mm Hg</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputBP" max="160" min="30" placeholder="e.g. 78" type="number" value="" required/>
                  </div>

                  <!-- BMI Field -->
                  <div class="flex flex-col gap-1.5 w-full min-w-0">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputBMI">
                        Body Mass Index (BMI)
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Weight in kg / (height in m)^2">info</span>
                      </label>
                      <span class="text-label-sm text-outline">kg/m²</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputBMI" max="70" min="10" step="0.1" placeholder="e.g. 31.4" type="number" value="" required/>
                  </div>

                  <!-- Serum Insulin Field -->
                  <div class="flex flex-col gap-1.5 w-full min-w-0">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputInsulin">
                        Serum Insulin
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="2-Hour serum insulin (µU/mL)">info</span>
                      </label>
                      <span class="text-label-sm text-outline">µU/mL</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputInsulin" max="900" min="0" placeholder="e.g. 130" type="number" value="" required/>
                  </div>

                  <!-- Triceps Skinfold Field -->
                  <div class="flex flex-col gap-1.5 w-full min-w-0">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputSkin">
                        Triceps Skinfold
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Triceps skin fold thickness in millimeters">info</span>
                      </label>
                      <span class="text-label-sm text-outline">mm</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputSkin" max="99" min="0" placeholder="e.g. 32" type="number" value="" required/>
                  </div>

                  <!-- Diabetes Pedigree Function Field -->
                  <div class="flex flex-col gap-1.5 sm:col-span-2 w-full min-w-0">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputPedigree">
                        Diabetes Pedigree Function
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Genetic synthesis score based on family history &amp; relative relations">info</span>
                      </label>
                      <span class="text-label-sm text-outline">score (0.05 - 2.50)</span>
                    </div>
                    <div class="flex flex-col sm:flex-row sm:items-center gap-3 w-full">
                      <input class="flex-1 h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors min-w-0" id="inputPedigree" max="3.0" min="0.01" step="0.001" placeholder="e.g. 0.672" type="number" value="" required/>
                      <span id="pedigreeBadge" class="hidden inline-flex items-center text-label-sm text-on-surface-variant bg-surface-container px-3.5 py-2.5 rounded-lg border border-outline-variant/30 font-medium shrink-0"></span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Form Action Row -->
              <div class="flex flex-col sm:flex-row items-center gap-3 pt-3 w-full">
                <button class="w-full sm:flex-1 h-12 bg-primary text-on-primary hover:bg-surface-tint text-label-lg rounded-xl flex items-center justify-center gap-2 shadow-sm transition-all active:scale-[0.99]" id="btnCalculate" type="submit">
                  <span class="material-symbols-outlined text-[20px]" id="calcIcon">calculate</span>
                  <span id="calcText">Estimate Diabetes Risk</span>
                </button>
                <button class="w-full sm:w-auto px-5 h-12 rounded-xl bg-surface-container text-on-surface hover:bg-surface-container-high text-label-md transition-colors flex items-center justify-center gap-1.5 font-medium" onclick="resetForm()" type="button">
                  <span class="material-symbols-outlined text-[18px]">restart_alt</span>
                  Clear Fields
                </button>
              </div>
            </form>
        </div>

      </div>

      <!-- ========================================================================= -->
      <!-- FOUR-STAGE ANIMATED INFERENCE TRANSITION MODAL                            -->
      <!-- ========================================================================= -->
      <div id="inferenceTransitionModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-md p-4 transition-all duration-300">
        <div class="relative w-full max-w-lg bg-surface-container-lowest/95 backdrop-blur-2xl rounded-3xl p-6 sm:p-8 shadow-[0_25px_60px_-15px_rgba(0,83,219,0.25)] border border-primary-fixed/40 flex flex-col gap-6 overflow-hidden">
          <!-- Ambient Blue/Cyan/Lavender Radial Glows -->
          <div class="absolute -top-20 -left-20 w-44 h-44 bg-secondary-container/30 rounded-full blur-3xl pointer-events-none"></div>
          <div class="absolute -bottom-20 -right-20 w-44 h-44 bg-primary/20 rounded-full blur-3xl pointer-events-none"></div>
          <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-36 h-36 bg-[#dae2fd]/25 rounded-full blur-2xl pointer-events-none"></div>

          <!-- Header -->
          <div class="flex items-center gap-3.5 relative z-10">
            <div class="w-12 h-12 rounded-2xl bg-gradient-to-tr from-primary to-secondary-container text-on-primary flex items-center justify-center shadow-md animate-pulse">
              <span class="material-symbols-outlined text-[26px]">vital_signs</span>
            </div>
            <div>
              <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 rounded-full text-[10px] font-bold font-mono tracking-wider bg-primary/10 text-primary border border-primary/20 uppercase">Inference Engine</span>
                <span class="w-2 h-2 rounded-full bg-secondary animate-ping"></span>
              </div>
              <h3 class="text-headline-sm text-on-surface font-bold text-[19px] mt-0.5">Evaluating Diabetes Risk</h3>
            </div>
          </div>

          <!-- Animated Progress Bar (Blue / Cyan / Lavender gradient) -->
          <div class="w-full bg-surface-container h-2 rounded-full overflow-hidden relative z-10">
            <div id="inferenceProgressBar" class="h-full bg-gradient-to-r from-primary via-secondary-container to-[#b4c5ff] transition-all duration-300 rounded-full" style="width: 25%;"></div>
          </div>

          <!-- 4 Stages List -->
          <div class="flex flex-col gap-3 relative z-10" id="inferenceStagesContainer">
            <!-- Stage 1: Cyan / Tealing -->
            <div id="stageItem1" class="flex items-start gap-3 p-3 rounded-xl bg-surface-container-high border-2 border-primary shadow-sm transition-all">
              <div id="stageIcon1" class="w-7 h-7 rounded-lg bg-primary text-on-primary flex items-center justify-center text-label-sm font-bold shrink-0 mt-0.5 shadow-xs">
                <span class="material-symbols-outlined text-[16px] animate-spin">sync</span>
              </div>
              <div class="flex-1">
                <div class="flex items-center justify-between">
                  <span class="text-label-md font-bold text-on-surface">1. Biometric Ingestion &amp; Preprocessing</span>
                  <span id="stageStatus1" class="text-[11px] font-mono font-bold text-primary">Validating...</span>
                </div>
                <p class="text-[12px] text-on-surface-variant mt-0.5">Validating 8 physiological markers &amp; applying median imputation protocol</p>
              </div>
            </div>

            <!-- Stage 2: Blue -->
            <div id="stageItem2" class="flex items-start gap-3 p-3 rounded-xl bg-surface-container-low/50 border border-outline-variant/20 transition-all opacity-60">
              <div id="stageIcon2" class="w-7 h-7 rounded-lg bg-surface-container text-outline flex items-center justify-center text-label-sm font-bold shrink-0 mt-0.5">
                <span>2</span>
              </div>
              <div class="flex-1">
                <div class="flex items-center justify-between">
                  <span class="text-label-md font-bold text-on-surface">2. Ensemble Model Execution</span>
                  <span id="stageStatus2" class="text-[11px] font-mono font-bold text-outline">Queued</span>
                </div>
                <p class="text-[12px] text-on-surface-variant mt-0.5">Evaluating trained estimators against calibrated decision threshold</p>
              </div>
            </div>

            <!-- Stage 3: Lavender -->
            <div id="stageItem3" class="flex items-start gap-3 p-3 rounded-xl bg-surface-container-low/50 border border-outline-variant/20 transition-all opacity-60">
              <div id="stageIcon3" class="w-7 h-7 rounded-lg bg-surface-container text-outline flex items-center justify-center text-label-sm font-bold shrink-0 mt-0.5">
                <span>3</span>
              </div>
              <div class="flex-1">
                <div class="flex items-center justify-between">
                  <span class="text-label-md font-bold text-on-surface">3. TreeSHAP Feature Attribution</span>
                  <span id="stageStatus3" class="text-[11px] font-mono font-bold text-outline">Queued</span>
                </div>
                <p class="text-[12px] text-on-surface-variant mt-0.5">Computing exact additive Shapley contributions for patient biometrics</p>
              </div>
            </div>

            <!-- Stage 4: Cyan/Emerald -->
            <div id="stageItem4" class="flex items-start gap-3 p-3 rounded-xl bg-surface-container-low/50 border border-outline-variant/20 transition-all opacity-60">
              <div id="stageIcon4" class="w-7 h-7 rounded-lg bg-surface-container text-outline flex items-center justify-center text-label-sm font-bold shrink-0 mt-0.5">
                <span>4</span>
              </div>
              <div class="flex-1">
                <div class="flex items-center justify-between">
                  <span class="text-label-md font-bold text-on-surface">4. Risk Calibration &amp; Synthesis</span>
                  <span id="stageStatus4" class="text-[11px] font-mono font-bold text-outline">Queued</span>
                </div>
                <p class="text-[12px] text-on-surface-variant mt-0.5">Synthesizing clinical risk tier &amp; explainability report</p>
              </div>
            </div>
          </div>

          <!-- Footer Note -->
          <div class="flex items-center justify-between pt-2 text-[12px] text-outline relative z-10 border-t border-outline-variant/20">
            <span class="flex items-center gap-1">
              <span class="material-symbols-outlined text-[14px] text-secondary">security</span>
              Zero-Retention Client Ingestion
            </span>
            <span id="inferenceModelTag" class="font-mono text-primary font-semibold">Random Forest</span>
          </div>
        </div>
      </div>

      <!-- ========================================================================= -->
      <!-- TAB / PAGE 2: DEDICATED PREDICTION RESULTS                                 -->
      <!-- ========================================================================= -->
      <div id="tab-content-prediction-results" class="hidden flex flex-col w-full">
        
        <!-- Top Action & Navigation Header -->
        <div class="w-full bg-surface-container-lowest rounded-2xl p-4 mb-6 shadow-sm border border-outline-variant/30 flex flex-wrap items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <button class="px-3.5 py-2 rounded-xl bg-surface-container hover:bg-surface-container-high text-on-surface text-label-md font-semibold flex items-center gap-1.5 transition-colors shadow-xs" onclick="switchTab('risk-prediction')">
              <span class="material-symbols-outlined text-[18px]">arrow_back</span>
              <span>Back to Assessment</span>
            </button>
            <div class="h-6 w-px bg-outline-variant/40 hidden sm:block"></div>
            <div class="hidden sm:flex items-center gap-2">
              <span class="text-label-sm text-outline">Assessment Record:</span>
              <span class="font-mono text-label-sm font-bold text-primary bg-primary/10 px-2 py-0.5 rounded" id="resultRecordBadge">Awaiting Assessment</span>
            </div>
          </div>
          
          <div class="flex items-center gap-2.5">
            <button class="px-4 py-2 rounded-xl bg-surface-container hover:bg-surface-container-high text-on-surface text-label-md font-semibold flex items-center gap-1.5 transition-colors" onclick="newAssessment()">
              <span class="material-symbols-outlined text-[18px]">add_circle</span>
              <span>New Assessment</span>
            </button>
            <button class="px-4 py-2 rounded-xl bg-primary text-on-primary hover:bg-surface-tint text-label-md font-semibold flex items-center gap-1.5 shadow-xs transition-all" onclick="exportReport()">
              <span class="material-symbols-outlined text-[18px]">print</span>
              <span>Export Report</span>
            </button>
          </div>
        </div>

        <!-- GRACEFUL EMPTY / MISSING RESULTS STATE (Shown when no active prediction exists) -->
        <div id="resultsEmptyState" class="w-full bg-surface-container-lowest rounded-2xl p-10 lg:p-14 shadow-sm border border-outline-variant/30 flex flex-col items-center justify-center text-center gap-4 my-4">
          <div class="w-16 h-16 rounded-2xl bg-secondary-container/40 text-secondary flex items-center justify-center shadow-xs">
            <span class="material-symbols-outlined text-[36px]">biotech</span>
          </div>
          <div class="max-w-md flex flex-col gap-1.5">
            <h3 class="text-headline-sm font-bold text-on-surface text-[22px]">No Assessment Results Available</h3>
            <p class="text-body-sm text-on-surface-variant leading-relaxed" id="emptyStateMsg">
              No active prediction calculation was found for the current patient profile. Enter physiological markers on the Assessment page and click "Estimate Diabetes Risk" to generate model inference and TreeSHAP attributions.
            </p>
          </div>
          <button class="mt-2 px-6 py-2.5 rounded-xl bg-primary text-on-primary hover:bg-surface-tint text-label-md font-bold shadow-sm transition-all flex items-center gap-2" onclick="switchTab('risk-prediction')">
            <span class="material-symbols-outlined text-[20px]">assignment</span>
            <span>Go to Risk Assessment</span>
          </button>
        </div>

        <!-- ACTIVE RESULTS CONTAINER (Shown when currentResultRecord exists) -->
        <div id="resultsContent" class="hidden flex flex-col w-full gap-6">
          
          <!-- Active Model & Timestamp Strip -->
          <div class="w-full bg-surface-container-low rounded-xl p-3.5 px-4.5 border border-outline-variant/30 flex flex-wrap items-center justify-between gap-3 text-label-sm">
            <div class="flex items-center gap-2.5">
              <span class="material-symbols-outlined text-primary text-[20px]" id="resultBannerIcon">forest</span>
              <span class="text-on-surface font-semibold">Evaluated Architecture:</span>
              <span class="font-bold text-primary" id="resultModelName">Random Forest Classifier (Selected Best)</span>
              <span class="text-secondary bg-surface px-2 py-0.5 rounded-md font-mono text-[12px] border border-outline-variant/30" id="resultThresholdBadge">Decision Threshold: 0.48</span>
            </div>
            <div class="flex items-center gap-2 text-outline">
              <span class="material-symbols-outlined text-[16px]">schedule</span>
              <span id="resultTimestamp">Generated: Just now</span>
            </div>
          </div>

          <!-- 2-COLUMN GRID: Probability Gauge Card (Left 6 cols) & Patient Inputs Summary (Right 6 cols) -->
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            <!-- LEFT (6 cols): Primary Risk Gauge Card -->
            <div class="lg:col-span-6 flex flex-col gap-6">
              <!-- Primary Risk Gauge Card -->
            <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm flex flex-col items-center text-center relative overflow-hidden">
              
              <!-- Header Note with Active Model Display & Switcher Button -->
              <div class="w-full flex items-center justify-between pb-3 border-b border-outline-variant/20">
                <div class="flex items-center gap-1.5">
                  <span class="material-symbols-outlined text-[18px] text-primary" id="gaugeModelIcon">forest</span>
                  <span class="text-label-md font-bold text-on-surface" id="gaugeModelLabel">Random Forest Inference</span>
                </div>
                <!-- Quick Model Switcher button right on the card -->
                <button class="px-2.5 py-1 rounded-full bg-surface-container hover:bg-surface-container-high text-primary text-label-sm font-semibold transition-colors flex items-center gap-1" onclick="toggleModelFromCard()" title="Switch active ML model">
                  <span class="material-symbols-outlined text-[14px]">swap_horiz</span>
                  <span id="cardSwitchBtnText">Switch to XGBoost</span>
                </button>
              </div>

              <!-- Circular Risk Gauge -->
              <div class="relative w-56 h-56 my-2 flex items-center justify-center">
                <svg class="w-full h-full -rotate-90 transform" viewBox="0 0 200 200">
                  <!-- Background Track -->
                  <circle cx="100" cy="100" fill="transparent" r="82" stroke="#eaedff" stroke-width="16"></circle>
                  <!-- Segment Progress Arc (Circumference 515.2) -->
                  <circle class="transition-all duration-700 ease-out" cx="100" cy="100" fill="transparent" id="riskGaugeCircle" r="82" stroke="#ba1a1a" stroke-dasharray="515.2" stroke-dashoffset="164.8" stroke-linecap="round" stroke-width="16"></circle>
                </svg>
                <div class="absolute inset-0 flex flex-col items-center justify-center">
                  <span class="text-display-lg font-bold text-on-surface tracking-tight" id="riskGaugeValue">--%</span>
                  <span class="text-label-md uppercase tracking-wider text-on-surface-variant font-semibold">Estimated Risk</span>
                  <span class="text-label-sm text-outline mt-0.5" id="riskGaugeThreshold">Threshold: 0.48</span>
                </div>
              </div>

              <!-- Risk Categorization Pill -->
              <div class="mt-1 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-surface-container text-on-surface-variant text-label-lg font-bold" id="riskPill">
                <span class="material-symbols-outlined text-[18px]" id="riskPillIcon">pending</span>
                <span id="riskPillText">Awaiting Assessment</span>
              </div>
              <p class="text-body-sm text-on-surface-variant mt-2 max-w-xs" id="riskPhenotypeDesc">
                Estimated risk probability evaluated against the active model decision threshold.
              </p>

              <!-- Spectrum Visualizer -->
              <div class="w-full mt-4 flex flex-col gap-1.5">
                <div class="flex items-center justify-between text-label-sm text-on-surface-variant">
                  <span id="specLabelLow">Lower (&lt;30%)</span>
                  <span id="specLabelMod">Moderate (30-60%)</span>
                  <span id="specLabelHigh" class="font-bold text-error">Elevated (&gt;60%)</span>
                </div>
                <!-- 3-Segment Color Bar -->
                <div class="w-full h-2.5 rounded-full bg-surface-container flex overflow-hidden p-0.5 gap-0.5 relative">
                  <div class="w-[30%] h-full rounded-l-full bg-tertiary-container/80"></div>
                  <div class="w-[30%] h-full bg-secondary-fixed-dim"></div>
                  <div class="w-[40%] h-full rounded-r-full bg-error"></div>
                  <!-- Marker Pin -->
                  <div class="absolute top-[-3px] w-2 h-4 bg-on-surface rounded-full shadow-md transition-all duration-500" id="spectrumMarker" style="left: 50%;"></div>
                </div>
              </div>

              <!-- ========================================================== -->
              <!-- IMMEDIATE SHAP SUMMARY BOX (Right on Screen Above the Fold) -->
              <!-- ========================================================== -->
              <div class="w-full mt-4 p-4 rounded-xl bg-secondary-container/20 border border-secondary-container/50 text-left">
                <div class="flex items-center justify-between mb-2">
                  <span class="text-label-sm font-bold text-on-secondary-container flex items-center gap-1.5">
                    <span class="material-symbols-outlined text-[18px] text-secondary">psychology</span>
                    SHAP Local Risk Decomposition
                  </span>
                  <span class="text-[11px] font-mono font-bold text-secondary bg-surface px-2 py-0.5 rounded-full" id="shapSummaryBadge">Net SHAP</span>
                </div>
                
                <div class="flex flex-col gap-1.5 text-label-sm" id="instantShapPills">
                  <!-- Injected via JS -->
                </div>

                <!-- Jump Button -->
                <div class="mt-3 pt-2.5 border-t border-secondary/20 flex items-center justify-between">
                  <button class="text-label-sm font-bold text-secondary hover:text-on-secondary-container flex items-center gap-1 transition-colors" onclick="scrollToShap()">
                    <span>Full Waterfall Breakdown</span>
                    <span class="material-symbols-outlined text-[16px]">arrow_downward</span>
                  </button>
                  <button class="text-label-sm font-bold text-primary hover:underline flex items-center gap-1" onclick="switchTab('explainability-xai')">
                    <span>Global SHAP Suite</span>
                    <span class="material-symbols-outlined text-[16px]">arrow_forward</span>
                  </button>
                </div>
              </div>

              <!-- Model Prediction & Decision Boundary Footer -->
              <div class="w-full mt-4 pt-3 bg-surface-container-low rounded-xl p-3.5 flex items-center justify-between text-left">
                <div class="flex items-center gap-2.5">
                  <div class="w-8 h-8 rounded-lg bg-secondary-container text-on-secondary-container flex items-center justify-center">
                    <span class="material-symbols-outlined text-[18px]">analytics</span>
                  </div>
                  <div>
                    <span class="text-label-sm text-on-surface-variant block">Predicted Risk Probability</span>
                    <span class="text-label-md text-on-surface font-bold" id="confidenceValue">--% Predicted Probability</span>
                  </div>
                </div>
                <span class="text-label-sm text-secondary bg-surface px-2 py-1 rounded-md font-medium border border-outline-variant/30" id="brierScore">Threshold: 0.48</span>
              </div>
            </div>
            </div>

            <!-- RIGHT (6 cols): Concise Patient-Input Summary Card -->
            <div class="lg:col-span-6 bg-surface-container-lowest rounded-2xl p-6 lg:p-7 shadow-sm flex flex-col gap-4 border border-outline-variant/30" id="patientInputSummaryCard">
              <div class="flex items-center justify-between pb-3 border-b border-outline-variant/20">
                <div>
                  <div class="flex items-center gap-2">
                    <span class="w-7 h-7 rounded-lg bg-primary/10 text-primary flex items-center justify-center font-bold">
                      <span class="material-symbols-outlined text-[16px]">fact_check</span>
                    </span>
                    <h3 class="text-label-lg font-bold text-on-surface">Patient Inputs Used for this Assessment</h3>
                  </div>
                  <p class="text-body-sm text-on-surface-variant mt-0.5">Biometric vector recorded and evaluated for this prediction:</p>
                </div>
                <span class="text-label-sm text-secondary bg-secondary-container/40 px-2.5 py-0.5 rounded-full font-bold">8 Biomarkers</span>
              </div>

              <!-- 8 Biometric Summary Tiles -->
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-3" id="patientInputSummaryTiles">
                <!-- Dynamically populated via JS -->
              </div>

              <!-- Inputs Bound & Invalidation Notice -->
              <div class="p-3 rounded-xl bg-surface-container-low text-body-sm text-on-surface-variant flex items-center gap-2.5 border border-outline-variant/20">
                <span class="material-symbols-outlined text-[18px] text-primary shrink-0">info</span>
                <span class="text-[12px] leading-snug">
                  <strong>Inputs Snapshot:</strong> These parameters are bound to the active prediction record. Modifying inputs on the Assessment page or switching ML model will invalidate these results.
                </span>
              </div>
            </div>
          </div>

          <!-- SECTION 1: Local SHAP Explanation -->
          <!-- ========================================================================= -->
        <!-- SECTION 1: INDIVIDUAL PREDICTION EXPLANATION (WHY DID MODEL PREDICT THIS?) -->
        <!-- ========================================================================= -->
        <section class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm mb-8 flex flex-col gap-6" id="shap-analysis-section">
          
          <!-- Header with Title, Action and Explanatory Formula -->
          <div class="flex flex-col gap-3 pb-2 border-b border-outline-variant/30">
            <div class="flex flex-wrap items-center justify-between gap-3">
              <div class="flex items-center gap-2.5">
                <span class="w-9 h-9 rounded-xl bg-secondary-container text-on-secondary-container flex items-center justify-center font-bold">
                  <span class="material-symbols-outlined text-[22px]">psychology</span>
                </span>
                <div>
                  <h2 class="text-headline-sm text-on-surface font-bold text-[20px]">
                    Why did the model make this prediction?
                  </h2>
                  <p class="text-body-sm text-on-surface-variant">
                    Individual TreeSHAP feature attributions decomposing the deviation between baseline and predicted risk.
                  </p>
                </div>
              </div>

              <!-- Context Pill with Active Model, Threshold & Class -->
              <div class="flex items-center gap-2">
                <span class="px-3 py-1 rounded-full text-label-sm font-semibold bg-surface-container text-on-surface border border-outline-variant/30" id="explainerModelBadge">Random Forest • Th: 0.48</span>
                <button class="hidden sm:inline-flex items-center gap-1 px-3 py-1 rounded-full bg-secondary-container/40 text-on-secondary-container hover:bg-secondary-container text-label-sm font-semibold transition-colors" onclick="switchTab('explainability-xai')">
                  <span class="material-symbols-outlined text-[16px]">open_in_new</span>
                  <span>Deep Dive</span>
                </button>
              </div>
            </div>

            <!-- Mathematical Space & Additive Formula -->
            <div class="p-3 rounded-xl bg-surface-container-low text-body-sm text-on-surface-variant flex flex-wrap items-center justify-between gap-2 border border-outline-variant/20">
              <span id="waterfallFormulaText">
                Features contribute additively in probability space from expected baseline <code class="bg-surface px-1.5 py-0.5 rounded text-primary font-mono font-semibold" id="waterfallBaseFormula">E[f(x)] = 0.498</code> to predicted risk probability <code class="bg-surface px-1.5 py-0.5 rounded text-error font-mono font-semibold" id="waterfallPredictedFormula">p = --</code>.
              </span>
              <span class="text-label-sm font-mono font-bold text-secondary bg-surface px-2.5 py-0.5 rounded-full" id="explainabilityNetBadge">+0.00 Net SHAP</span>
            </div>
          </div>

          <!-- Main Layout Grid: Waterfall (Left 7 Cols) + Top Drivers (Right 5 Cols) -->
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            <!-- LEFT (7 cols): Single Horizontal Waterfall Divergence Chart -->
            <div class="lg:col-span-7 bg-surface-container-low rounded-xl p-4 sm:p-5 border border-outline-variant/30 flex flex-col gap-3">
              <div class="flex items-center justify-between pb-2 border-b border-outline-variant/30">
                <span class="text-label-md font-bold text-on-surface flex items-center gap-1.5">
                  <span class="material-symbols-outlined text-primary text-[18px]">waterfall_chart</span>
                  Patient Feature Attributions (SHAP)
                </span>
                <span class="text-[12px] text-outline font-medium">Exact signed contribution (φ)</span>
              </div>

              <!-- Divergence Axis Labels -->
              <div class="grid grid-cols-12 text-center text-label-sm text-on-surface-variant pb-1 font-semibold">
                <div class="col-span-4 text-left">Clinical Biomarker</div>
                <div class="col-span-3 text-right text-secondary">← Protective (-φ)</div>
                <div class="col-span-1 text-center font-mono font-bold text-outline">0.0</div>
                <div class="col-span-4 text-left text-error">Risk Push (+φ) →</div>
              </div>

              <!-- Divergence Rows Container -->
              <div id="divergenceRows" class="flex flex-col gap-1.5">
                <!-- Dynamically populated via JS -->
              </div>
            </div>

            <!-- RIGHT (5 cols): Compact Key Drivers Summary -->
            <div class="lg:col-span-5 flex flex-col gap-4">
              
              <!-- Top Increasing Factor(s) -->
              <div class="bg-surface-container-low rounded-xl p-4 border border-outline-variant/30 flex flex-col gap-2.5">
                <div class="flex items-center justify-between pb-2 border-b border-outline-variant/20">
                  <div class="flex items-center gap-2">
                    <span class="w-7 h-7 rounded-lg bg-error-container text-on-error-container flex items-center justify-center font-bold">
                      <span class="material-symbols-outlined text-[16px]">arrow_upward</span>
                    </span>
                    <span class="text-label-md font-bold text-on-surface">Top Risk-Increasing Drivers</span>
                  </div>
                  <span class="px-2.5 py-0.5 rounded-full bg-error-container font-code-num text-label-sm text-on-error-container font-bold" id="netPosBadge">+0.00 Net</span>
                </div>
                <div class="flex flex-col gap-2" id="increasingFactorsList">
                  <!-- Injected via JS -->
                </div>
              </div>

              <!-- Top Decreasing / Protective Factor(s) -->
              <div class="bg-surface-container-low rounded-xl p-4 border border-outline-variant/30 flex flex-col gap-2.5">
                <div class="flex items-center justify-between pb-2 border-b border-outline-variant/20">
                  <div class="flex items-center gap-2">
                    <span class="w-7 h-7 rounded-lg bg-secondary-container text-on-secondary-container flex items-center justify-center font-bold">
                      <span class="material-symbols-outlined text-[16px]">arrow_downward</span>
                    </span>
                    <span class="text-label-md font-bold text-on-surface">Top Protective Drivers</span>
                  </div>
                  <span class="px-2.5 py-0.5 rounded-full bg-secondary-container font-code-num text-label-sm text-on-secondary-container font-bold" id="netNegBadge">-0.00 Net</span>
                </div>
                <div class="flex flex-col gap-2" id="decreasingFactorsList">
                  <!-- Injected via JS -->
                </div>
              </div>

              <!-- Primary Driver Clinical Interpretation -->
              <div class="p-3.5 rounded-xl bg-surface-container border border-outline-variant/20 flex items-start gap-2.5 text-body-sm text-on-surface-variant font-medium" id="primaryDriverText">
                <span class="material-symbols-outlined text-[18px] text-primary shrink-0 mt-0.5">info</span>
                <span>Calculating feature attribution drivers...</span>
              </div>

            </div>

          </div>
        </section>

        <!-- ========================================================================= -->
        <!-- SECTION 2: GLOBAL FEATURE IMPORTANCE                                      -->
        <!-- ========================================================================= -->
        <section class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm mb-8 flex flex-col gap-5">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <div class="flex items-center gap-2 mb-0.5">
                <span class="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center font-bold">
                  <span class="material-symbols-outlined text-[20px]">leaderboard</span>
                </span>
                <h2 class="text-headline-sm text-on-surface font-bold text-[20px]" id="importanceSectionTitle">
                  Global Feature Importance
                </h2>
              </div>
              <p class="text-body-sm text-on-surface-variant" id="importanceSectionSubtitle">
                Which features influence the model most across the evaluated dataset?
              </p>
            </div>

            <!-- Model Switching Control -->
            <div class="flex items-center gap-1.5 bg-surface-container p-1 rounded-xl self-start sm:self-auto border border-outline-variant/30">
              <button class="px-3.5 py-1.5 rounded-lg text-label-sm font-semibold transition-all bg-primary text-on-primary shadow-xs" id="btnImpRf" onclick="switchImportanceView('rf')">
                Random Forest (Gini)
              </button>
              <button class="px-3.5 py-1.5 rounded-lg text-label-sm font-medium transition-all bg-transparent text-on-surface-variant hover:text-on-surface" id="btnImpXgb" onclick="switchImportanceView('xgb')">
                XGBoost (SHAP)
              </button>
            </div>
          </div>

          <!-- Single Ranked Horizontal Bar Chart -->
          <div id="importanceBarsContainer" class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3.5 pt-1">
            <!-- Dynamically populated via JS -->
          </div>

          <!-- Compact Methodological & Architectural Metadata Strip -->
          <div class="pt-3 border-t border-outline-variant/20 flex flex-wrap items-center justify-between gap-3 text-label-sm text-on-surface-variant">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-primary text-[18px]">verified</span>
              <span class="font-bold text-on-surface" id="kpiModelName">Random Forest (Selected Best)</span>
              <span class="text-primary font-semibold" id="kpiModelAuc">Holdout AUC: 0.8333</span>
              <span class="text-outline" id="kpiModelF1">• Holdout F1: 68.22% (Th: 0.48)</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-secondary text-[18px]">bloodtype</span>
              <span>Dominant Predictor: <strong class="text-on-surface">Glucose</strong></span>
              <span class="text-secondary font-semibold" id="kpiFeatureScore">Mean |SHAP|: 0.141</span>
              <span class="text-outline" id="kpiFeatureShare">• 34.1% RF Gini • Rank 1</span>
            </div>
          </div>
        </section>

        <!-- SECTION 3: Model Specifications & Technical Metadata Strip -->
        <section class="bg-surface-container-low rounded-2xl p-4 lg:p-5 mb-8 shadow-xs border border-outline-variant/30">
          <div class="flex flex-wrap items-center justify-between gap-4 text-label-sm text-on-surface-variant">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-primary text-[18px]">verified</span>
              <span class="font-semibold text-on-surface">Models:</span> Random Forest (Selected Best • n=200) &amp; XGBoost (Tuned • n=150)
            </div>
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-secondary text-[18px]">hub</span>
              <span class="font-semibold text-on-surface">Explainability:</span> TreeSHAP (Lundberg et al., Nature MI)
            </div>
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-outline text-[18px]">dataset</span>
              <span class="font-semibold text-on-surface">Dataset:</span> Pima Indians Diabetes (N=769, 8 Features, Holdout N=154)
            </div>
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-tertiary text-[18px]">security</span>
              <span class="font-semibold text-on-surface">Ingress:</span> Zero-Retention Client Ingestion
            </div>
          </div>
        </section>

          <!-- Clinical Disclaimer Card -->
          <div class="w-full bg-surface-container-low rounded-2xl p-5 border border-outline-variant/30 flex items-start gap-3.5 mb-6">
            <span class="material-symbols-outlined text-secondary text-[24px] shrink-0 mt-0.5">shield</span>
            <div class="flex flex-col gap-1">
              <h4 class="text-label-md font-bold text-on-surface">Clinical Decision Support Notice &amp; Disclaimer</h4>
              <p class="text-body-sm text-on-surface-variant leading-relaxed">
                This system is an investigational research prototype designed for metabolic risk stratification and interpretability exploration. Model probability estimates and TreeSHAP feature attributions do not replace standardized diagnostic evaluations (fasting plasma glucose, HbA1c, oral glucose tolerance test) or professional medical judgment. All clinical management decisions must be confirmed by qualified medical professionals according to established clinical practice guidelines.
              </p>
            </div>
          </div>

        </div>

      </div>

      <!-- ========================================================================= -->
      <!-- TAB 2: SHAP EXPLAINABILITY (XAI DEEP DIVE)                                -->
      <!-- ========================================================================= -->
      <div id="tab-content-explainability-xai" class="hidden flex flex-col w-full gap-6">
        
        <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <div class="w-12 h-12 rounded-xl bg-secondary-container text-on-secondary-container flex items-center justify-center">
              <span class="material-symbols-outlined text-[28px]">psychology</span>
            </div>
            <div>
              <h2 class="text-headline-sm text-on-surface font-bold">SHAP Explainability Suite (TreeSHAP)</h2>
              <p class="text-body-sm text-on-surface-variant">Comprehensive local and global interpretability powered by cooperative game theoretic Shapley values.</p>
            </div>
          </div>
          <button class="px-4 py-2 rounded-xl bg-primary text-on-primary text-label-md font-semibold flex items-center gap-2 shadow-xs" onclick="switchTab('risk-prediction'); scrollToShap();">
            <span class="material-symbols-outlined text-[18px]">arrow_back</span>
            <span>Return to Patient Risk Waterfall</span>
          </button>
        </div>

        <!-- Global SHAP Plots (Beeswarm & Bar) -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-4">
                <div>
                  <h3 class="text-label-lg font-bold text-on-surface">SHAP Summary Beeswarm Plot</h3>
                  <p class="text-body-sm text-on-surface-variant">Distribution of individual patient SHAP impacts across holdout test set.</p>
                </div>
                <span class="text-label-sm text-secondary bg-secondary-container/40 px-2.5 py-1 rounded-full font-bold">Cohort Distribution</span>
              </div>
              <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-2">
                <img src="{img_beeswarm}" alt="SHAP Summary Beeswarm" class="w-full h-auto object-contain max-h-[380px]"/>
              </div>
            </div>
            <div class="mt-4 pt-3 text-body-sm text-on-surface-variant border-t border-outline-variant/20">
              <strong>Key Insight:</strong> Red points signify high feature values. High Glucose concentrations strongly push SHAP values rightward (&gt;+1.5), dramatically increasing diabetes likelihood.
            </div>
          </div>

          <!-- Feature Importance Bar Chart -->
          <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-4">
                <div>
                  <h3 class="text-label-lg font-bold text-on-surface">Global Feature Importance (Mean |SHAP|)</h3>
                  <p class="text-body-sm text-on-surface-variant">Mean absolute attribution magnitude per physiological biomarker.</p>
                </div>
                <span class="text-label-sm text-primary bg-primary-container/20 px-2.5 py-1 rounded-full font-bold">Global Magnitude</span>
              </div>
              <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-2">
                <img src="{img_shap_bar}" alt="SHAP Bar Plot" class="w-full h-auto object-contain max-h-[380px]"/>
              </div>
            </div>
            <div class="mt-4 pt-3 text-body-sm text-on-surface-variant border-t border-outline-variant/20">
              <strong>Key Insight:</strong> Glucose (0.932) and BMI (0.523) are the primary drivers across the entire population, followed by Age and hereditary Pedigree Function.
            </div>
          </div>
        </div>

        <!-- SHAP Feature Dependence Explorer -->
        <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
            <div>
              <h3 class="text-label-lg font-bold text-on-surface flex items-center gap-2">
                <span class="material-symbols-outlined text-secondary text-[20px]">show_chart</span>
                SHAP Feature Dependence Plots
              </h3>
              <p class="text-body-sm text-on-surface-variant">Observe non-linear thresholds and feature interaction effects across clinical biomarkers.</p>
            </div>
            <!-- Dependence Selector -->
            <div class="flex items-center gap-2 bg-surface-container p-1 rounded-xl">
              <button id="depBtn-glucose" class="px-3 py-1.5 rounded-lg text-label-sm font-bold bg-primary text-on-primary transition-all" onclick="showDependencePlot('glucose')">Glucose</button>
              <button id="depBtn-bmi" class="px-3 py-1.5 rounded-lg text-label-sm font-medium text-on-surface-variant hover:text-on-surface transition-all" onclick="showDependencePlot('bmi')">BMI</button>
              <button id="depBtn-age" class="px-3 py-1.5 rounded-lg text-label-sm font-medium text-on-surface-variant hover:text-on-surface transition-all" onclick="showDependencePlot('age')">Age</button>
              <button id="depBtn-pedigree" class="px-3 py-1.5 rounded-lg text-label-sm font-medium text-on-surface-variant hover:text-on-surface transition-all" onclick="showDependencePlot('pedigree')">Pedigree</button>
              <button id="depBtn-preg" class="px-3 py-1.5 rounded-lg text-label-sm font-medium text-on-surface-variant hover:text-on-surface transition-all" onclick="showDependencePlot('preg')">Pregnancies</button>
            </div>
          </div>

          <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-4">
            <img id="depPlotImage" src="{img_dep_gluc}" alt="SHAP Dependence Plot" class="w-full max-w-2xl h-auto object-contain max-h-[420px]"/>
          </div>
          <p class="text-body-sm text-on-surface-variant mt-3" id="depPlotDesc">
            <strong>Glucose Effect:</strong> Clear inflection point observed at ~125 mg/dL. Values below 100 mg/dL yield negative SHAP (protective), whereas values exceeding 140 mg/dL exhibit steep risk elevation.
          </p>
        </div>

        <!-- XGBoost vs Random Forest Feature Importance Comparison -->
        <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h3 class="text-label-lg font-bold text-on-surface">Feature Importance Comparison: XGBoost vs Random Forest</h3>
              <p class="text-body-sm text-on-surface-variant">Comparing model-native Gini/gain feature importances across both ensemble algorithms.</p>
            </div>
            <span class="text-label-sm font-bold text-primary bg-primary-fixed/40 px-3 py-1 rounded-full">RF vs XGBoost</span>
          </div>
          <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-4">
            <img src="{img_imp_comp}" alt="Feature Importance Comparison" class="w-full max-w-3xl h-auto object-contain max-h-[400px]"/>
          </div>
          <p class="text-body-sm text-on-surface-variant mt-3">
            Both architectures independently establish <strong>Glucose</strong> as the #1 predictive biomarker (~31-32% contribution), followed by <strong>BMI</strong> (~15-18%) and <strong>Age</strong> (~11-12%). Random Forest attributes slightly higher weight to hereditary pedigree score compared to gradient boosting.
          </p>
        </div>

        <!-- Mathematical Foundations Card -->
        <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm">
          <h3 class="text-label-lg font-bold text-on-surface mb-3 flex items-center gap-2">
            <span class="material-symbols-outlined text-primary text-[20px]">functions</span>
            Mathematical Rigor & Theoretical Guarantees of TreeSHAP
          </h3>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
            <div class="p-4 rounded-xl bg-surface-container-low">
              <span class="text-label-md font-bold text-primary block mb-1">1. Local Accuracy (Additivity)</span>
              <p class="text-body-sm text-on-surface-variant">
                The sum of feature attributions strictly equals the difference between the model's output and its expected baseline: <code>f(x) = E[f(x)] + Σ φ_i</code>.
              </p>
            </div>
            <div class="p-4 rounded-xl bg-surface-container-low">
              <span class="text-label-md font-bold text-secondary block mb-1">2. Consistency</span>
              <p class="text-body-sm text-on-surface-variant">
                If a model changes such that a feature's marginal contribution increases, its assigned attribution cannot decrease, preventing misleading comparisons.
              </p>
            </div>
            <div class="p-4 rounded-xl bg-surface-container-low">
              <span class="text-label-md font-bold text-tertiary block mb-1">3. Missingness</span>
              <p class="text-body-sm text-on-surface-variant">
                Features with zero influence on the prediction are guaranteed a SHAP value of zero, eliminating noise and false associations.
              </p>
            </div>
          </div>
        </div>

      </div>

      <!-- ========================================================================= -->
      <!-- TAB 3: MODEL BENCHMARK (RANDOM FOREST VS XGBOOST)                         -->
      <!-- ========================================================================= -->
      <div id="tab-content-model-benchmark-and-insights" class="hidden flex flex-col w-full gap-6">
        
        <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm">
          <div class="flex items-center gap-3 mb-2">
            <div class="w-10 h-10 rounded-xl bg-primary-fixed text-on-primary-fixed flex items-center justify-center">
              <span class="material-symbols-outlined text-[22px]">leaderboard</span>
            </div>
            <div>
              <h2 class="text-headline-sm text-on-surface font-bold">Model Benchmark: Random Forest vs. XGBoost</h2>
              <p class="text-body-sm text-on-surface-variant">Direct clinical validation matrix comparing both trained classifiers on holdout test partitions (20% split, n=154).</p>
            </div>
          </div>
        </div>

        <!-- Benchmark Comparison Table Card -->
        <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm overflow-x-auto">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-label-lg font-bold text-on-surface">Phase 2 Model Evaluation Metrics Matrix</h3>
            <span class="text-label-sm text-outline font-medium">Evaluation Dataset: Pima Holdout Test (n=154)</span>
          </div>
          <table class="w-full text-left text-body-sm border-collapse">
            <thead>
              <tr class="border-b border-outline-variant/50 text-label-sm uppercase tracking-wider text-outline">
                <th class="py-3 px-4 font-semibold">Model Architecture</th>
                <th class="py-3 px-4 font-semibold">Accuracy</th>
                <th class="py-3 px-4 font-semibold">Precision</th>
                <th class="py-3 px-4 font-semibold">Recall (Sensitivity)</th>
                <th class="py-3 px-4 font-semibold">F1-Score</th>
                <th class="py-3 px-4 font-semibold">ROC-AUC</th>
                <th class="py-3 px-4 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-outline-variant/30">
              <tr class="hover:bg-surface-container-low transition-colors bg-primary-fixed/20 font-semibold" id="tableRowRf">
                <td class="py-3.5 px-4 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-primary"></span>
                  Random Forest (Selected Best • Th: 0.48)
                </td>
                <td class="py-3.5 px-4 font-code-num text-primary font-bold">73.38%</td>
                <td class="py-3.5 px-4 font-code-num">58.67%</td>
                <td class="py-3.5 px-4 font-code-num">81.48%</td>
                <td class="py-3.5 px-4 font-code-num text-primary font-bold">68.22%</td>
                <td class="py-3.5 px-4 font-code-num text-primary font-bold">0.8333</td>
                <td class="py-3.5 px-4">
                  <button class="px-3 py-1 rounded-full bg-primary text-on-primary text-label-sm font-semibold shadow-xs" onclick="changeModel('random_forest'); switchTab('risk-prediction');">
                    Use Random Forest
                  </button>
                </td>
              </tr>
              <tr class="hover:bg-surface-container-low transition-colors" id="tableRowXgb">
                <td class="py-3.5 px-4 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-secondary"></span>
                  XGBoost Classifier (Tuned • Th: 0.42)
                </td>
                <td class="py-3.5 px-4 font-code-num">74.03%</td>
                <td class="py-3.5 px-4 font-code-num">58.97%</td>
                <td class="py-3.5 px-4 font-code-num text-secondary font-bold">85.19%</td>
                <td class="py-3.5 px-4 font-code-num">69.70%</td>
                <td class="py-3.5 px-4 font-code-num">0.8404</td>
                <td class="py-3.5 px-4">
                  <button class="px-3 py-1 rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface text-label-sm font-semibold transition-colors" onclick="changeModel('xgboost'); switchTab('risk-prediction');">
                    Use XGBoost
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
          <div class="mt-4 p-3 rounded-lg bg-surface-container-low text-body-sm text-on-surface-variant flex items-center gap-2">
            <span class="material-symbols-outlined text-secondary text-[18px]">lightbulb</span>
            <span><strong>Model Comparison:</strong> Random Forest was selected based on training-side 5-fold cross-validation composite scoring. Both architectures exhibit strong discriminatory performance on the holdout evaluation split (AUC 0.833 vs 0.840) under their tuned decision thresholds.</span>
          </div>
        </div>

        <!-- Evaluation Plots Row: ROC-AUC & Both Confusion Matrices -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
            <div>
              <h3 class="text-label-lg font-bold text-on-surface mb-2">ROC-AUC Curves Comparison</h3>
              <p class="text-body-sm text-on-surface-variant mb-3">XGBoost (AUC 0.840) vs Random Forest (AUC 0.833).</p>
              <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-2">
                <img src="{img_roc}" alt="ROC Comparison" class="w-full h-auto object-contain max-h-[300px]"/>
              </div>
            </div>
            <p class="text-body-sm text-on-surface-variant mt-3 pt-2 border-t border-outline-variant/20">
              Evaluates true positive rate vs false positive rate across operational decision cutoffs.
            </p>
          </div>

          <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <h3 class="text-label-lg font-bold text-on-surface">Confusion Matrix: Random Forest</h3>
                <span class="text-label-sm text-primary font-bold bg-primary-fixed/40 px-2 py-0.5 rounded">Selected Best</span>
              </div>
              <p class="text-body-sm text-on-surface-variant mb-3">Accuracy: 73.38% • Recall: 81.48% (Th: 0.48)</p>
              <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-2">
                <img src="{img_cm_rf}" alt="Confusion Matrix Random Forest" class="w-full h-auto object-contain max-h-[300px]"/>
              </div>
            </div>
            <p class="text-body-sm text-on-surface-variant mt-3 pt-2 border-t border-outline-variant/20">
              Random Forest confusion matrix evaluated at optimal decision threshold 0.48.
            </p>
          </div>

          <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <h3 class="text-label-lg font-bold text-on-surface">Confusion Matrix: XGBoost</h3>
                <span class="text-label-sm text-secondary font-bold bg-secondary-container/40 px-2 py-0.5 rounded">Tuned</span>
              </div>
              <p class="text-body-sm text-on-surface-variant mb-3">Accuracy: 74.03% • Recall: 85.19% (Th: 0.42)</p>
              <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-2">
                <img src="{img_cm_xgb}" alt="Confusion Matrix XGBoost" class="w-full h-auto object-contain max-h-[300px]"/>
              </div>
            </div>
            <p class="text-body-sm text-on-surface-variant mt-3 pt-2 border-t border-outline-variant/20">
              XGBoost confusion matrix evaluated at optimal decision threshold 0.42.
            </p>
          </div>
        </div>

      </div>



    </div>
  </main>

  <!-- Export Report Modal -->
  <div id="reportModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-inverse-surface/60 backdrop-blur-sm p-4">
    <div class="bg-surface-container-lowest rounded-2xl shadow-2xl max-w-2xl w-full p-6 lg:p-8 flex flex-col gap-6 max-h-[90vh] overflow-y-auto">
      <div class="flex items-center justify-between border-b border-outline-variant/30 pb-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-xl bg-primary text-on-primary flex items-center justify-center font-bold">
            <span class="material-symbols-outlined text-[20px]">picture_as_pdf</span>
          </div>
          <div>
            <h3 class="text-headline-sm font-bold text-on-surface">Clinical Decision Support Report</h3>
            <span class="text-label-sm text-on-surface-variant">TreeSHAP Explainable Risk Dossier</span>
          </div>
        </div>
        <button class="w-8 h-8 rounded-lg flex items-center justify-center text-outline hover:text-on-surface hover:bg-surface-container" onclick="closeReportModal()">
          <span class="material-symbols-outlined">close</span>
        </button>
      </div>

      <div class="flex flex-col gap-4 text-body-sm text-on-surface" id="reportBody">
        <!-- Content inserted dynamically via JS -->
      </div>

      <div class="flex items-center justify-end gap-3 pt-4 border-t border-outline-variant/30">
        <button class="px-4 py-2 rounded-xl bg-surface-container text-on-surface text-label-md font-medium" onclick="closeReportModal()">Close</button>
        <button class="px-5 py-2 rounded-xl bg-primary text-on-primary text-label-md font-semibold flex items-center gap-2 shadow-sm" onclick="window.print()">
          <span class="material-symbols-outlined text-[18px]">print</span>
          Print / Save PDF
        </button>
      </div>
    </div>
  </div>

  <!-- Settings Modal -->
  <div id="settingsModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-inverse-surface/60 backdrop-blur-sm p-4">
    <div class="bg-surface-container-lowest rounded-2xl shadow-2xl max-w-md w-full p-6 flex flex-col gap-5">
      <div class="flex items-center justify-between border-b border-outline-variant/30 pb-3">
        <h3 class="text-headline-sm font-bold text-on-surface flex items-center gap-2">
          <span class="material-symbols-outlined text-primary text-[20px]">settings</span>
          System Settings &amp; Model Selection
        </h3>
        <button class="w-8 h-8 rounded-lg flex items-center justify-center text-outline hover:text-on-surface hover:bg-surface-container" onclick="closeSettingsModal()">
          <span class="material-symbols-outlined">close</span>
        </button>
      </div>
      <div class="flex flex-col gap-4 text-body-sm">
        <div>
          <label class="font-medium text-on-surface block mb-1">Active Machine Learning Model</label>
          <select id="modalModelSelector" onchange="changeModel(this.value)" class="w-full h-10 px-3 bg-surface-container-low rounded-lg border border-outline-variant/40 font-semibold text-on-surface outline-none">
            <option value="random_forest" selected>Random Forest Classifier (Selected Best • Th: 0.48)</option>
            <option value="xgboost">XGBoost Classifier (Tuned • Th: 0.42)</option>
          </select>
        </div>
        <label class="flex items-center justify-between cursor-pointer">
          <span class="font-medium text-on-surface">Auto-recalculate on keystroke</span>
          <input type="checkbox" id="settingLiveRecalc" checked class="w-4 h-4 text-primary rounded"/>
        </label>
        <label class="flex items-center justify-between cursor-pointer">
          <span class="font-medium text-on-surface">Classification Threshold</span>
          <span class="text-label-sm font-mono bg-surface-container px-2 py-0.5 rounded" id="modalThresholdVal">0.48</span>
        </label>
      </div>
      <div class="flex justify-end pt-3">
        <button class="px-4 py-2 rounded-xl bg-primary text-on-primary text-label-md font-semibold" onclick="closeSettingsModal()">Done</button>
      </div>
    </div>
  </div>



  <!-- DYNAMIC CLIENT LOGIC -->
  <script>
    let activeModel = 'random_forest'; // 'random_forest' (selected best) or 'xgboost'
    let currentResultRecord = null;
    let activeRequestId = 0;
    let isCalculating = false;

    const DEP_PLOTS = {{
      glucose: {{
        img: '{img_dep_gluc}',
        desc: '<strong>Glucose Effect:</strong> In tree ensemble SHAP analysis, values below ~100 mg/dL yield negative SHAP contributions, whereas values exceeding ~140 mg/dL contribute strongly toward higher estimated risk.'
      }},
      bmi: {{
        img: '{img_dep_bmi}',
        desc: '<strong>BMI Effect:</strong> SHAP values turn positive above ~26.0 kg/m² and contribute higher positive risk adjustments as BMI increases.'
      }},
      age: {{
        img: '{img_dep_age}',
        desc: '<strong>Age Effect:</strong> Contribution increases steadily above age 35, leveling off in older age brackets.'
      }},
      pedigree: {{
        img: '{img_dep_ped}',
        desc: '<strong>Diabetes Pedigree Function Effect:</strong> Higher familial risk scores correlate with higher positive SHAP contributions to model output.'
      }},
      preg: {{
        img: '{img_dep_preg}',
        desc: "<strong>Pregnancies Parity Effect:</strong> Higher pregnancy counts (above 4) contribute positively to the model's risk output."
      }}
    }};

    function showDependencePlot(featureKey) {{
      const keys = ['glucose', 'bmi', 'age', 'pedigree', 'preg'];
      keys.forEach(k => {{
        const btn = document.getElementById('depBtn-' + k);
        if (btn) {{
          if (k === featureKey) {{
            btn.className = 'px-3 py-1.5 rounded-lg text-label-sm font-bold bg-primary text-on-primary transition-all';
          }} else {{
            btn.className = 'px-3 py-1.5 rounded-lg text-label-sm font-medium text-on-surface-variant hover:text-on-surface transition-all';
          }}
        }}
      }});

      const item = DEP_PLOTS[featureKey];
      if (item) {{
        document.getElementById('depPlotImage').src = item.img;
        document.getElementById('depPlotDesc').innerHTML = item.desc;
      }}
    }}

    function showToast(msg) {{
      let toast = document.getElementById('globalToastNotification');
      if (!toast) {{
        toast = document.createElement('div');
        toast.id = 'globalToastNotification';
        toast.className = 'fixed bottom-5 right-5 z-50 px-4 py-2.5 rounded-xl bg-on-surface text-surface text-label-sm font-semibold shadow-lg transition-all duration-300 opacity-0 pointer-events-none transform translate-y-2';
        document.body.appendChild(toast);
      }}
      toast.innerText = msg;
      toast.classList.remove('opacity-0', 'pointer-events-none', 'translate-y-2');
      toast.classList.add('opacity-100', 'translate-y-0');
      clearTimeout(window.__toastTimer);
      window.__toastTimer = setTimeout(() => {{
        toast.classList.remove('opacity-100', 'translate-y-0');
        toast.classList.add('opacity-0', 'pointer-events-none', 'translate-y-2');
      }}, 2600);
    }}

    function changeModel(modelName) {{
      if (activeModel !== modelName) {{
        invalidateStaleResults('model_changed');
      }}
      activeModel = modelName;
      
      // Update header selectors
      const hSel = document.getElementById('headerModelSelector');
      if (hSel) hSel.value = modelName;
      const mSel = document.getElementById('modalModelSelector');
      if (mSel) mSel.value = modelName;

      const toggleXgb = document.getElementById('toggleBtnXgb');
      const toggleRf = document.getElementById('toggleBtnRf');
      const bannerTitle = document.getElementById('bannerModelTitle');
      const gaugeModelLabel = document.getElementById('gaugeModelLabel');
      const gaugeModelIcon = document.getElementById('gaugeModelIcon');
      const cardSwitchBtnText = document.getElementById('cardSwitchBtnText');
      const kpiModelName = document.getElementById('kpiModelName');
      const kpiModelAuc = document.getElementById('kpiModelAuc');
      const kpiModelF1 = document.getElementById('kpiModelF1');
      const heroModelBadge = document.getElementById('heroModelBadge');
      const heroEngineTag = document.getElementById('heroEngineTag');
      const legendText = document.getElementById('legendText');

      const mTh = document.getElementById('modalThresholdVal');
      const gTh = document.getElementById('riskGaugeThreshold');
      const bScore = document.getElementById('brierScore');

      if (modelName === 'xgboost') {{
        toggleXgb.className = 'px-4 py-1.5 rounded-lg text-label-md font-bold transition-all bg-primary text-on-primary shadow-xs';
        toggleRf.className = 'px-4 py-1.5 rounded-lg text-label-md font-medium text-on-surface-variant hover:text-on-surface transition-all';
        bannerTitle.innerText = 'XGBoost Classifier (Tuned • Threshold: 0.42)';
        gaugeModelLabel.innerText = 'XGBoost Inference';
        gaugeModelIcon.innerText = 'memory';
        cardSwitchBtnText.innerText = 'Switch to Random Forest';
        kpiModelName.innerText = 'XGBoost (Tuned)';
        kpiModelAuc.innerText = 'Holdout AUC: 0.8404';
        kpiModelF1.innerText = '• Holdout F1: 69.70% (Th: 0.42)';
        const featScore = document.getElementById('kpiFeatureScore');
        const featShare = document.getElementById('kpiFeatureShare');
        if (featScore) featScore.innerText = 'Mean |SHAP|: 0.932';
        if (featShare) featShare.innerText = '• 31.7% XGB Gain • Rank 1';
        heroModelBadge.innerText = 'XGBoost Active';
        if (heroEngineTag) heroEngineTag.innerText = 'Explainable Machine Learning • TreeSHAP Engine';
        legendText.innerText = 'XGBoost Predictor';
        const explainerModelBadge = document.getElementById('explainerModelBadge');
        if (explainerModelBadge) explainerModelBadge.innerText = 'XGBoost (Tuned) • Th: 0.42';
        if (mTh) mTh.innerText = '0.42';
        if (gTh) gTh.innerText = 'Threshold: 0.42';
        if (bScore) bScore.innerText = 'Threshold: 0.42';
        showToast('Switched to XGBoost Model (Decision Threshold: 0.42)');
      }} else {{
        toggleRf.className = 'px-4 py-1.5 rounded-lg text-label-md font-bold transition-all bg-primary text-on-primary shadow-xs';
        toggleXgb.className = 'px-4 py-1.5 rounded-lg text-label-md font-medium text-on-surface-variant hover:text-on-surface transition-all';
        bannerTitle.innerText = 'Random Forest Classifier (Selected Best • Threshold: 0.48)';
        gaugeModelLabel.innerText = 'Random Forest Inference';
        gaugeModelIcon.innerText = 'forest';
        cardSwitchBtnText.innerText = 'Switch to XGBoost';
        kpiModelName.innerText = 'Random Forest (Selected Best)';
        kpiModelAuc.innerText = 'Holdout AUC: 0.8333';
        kpiModelF1.innerText = '• Holdout F1: 68.22% (Th: 0.48)';
        const featScore = document.getElementById('kpiFeatureScore');
        const featShare = document.getElementById('kpiFeatureShare');
        if (featScore) featScore.innerText = 'Mean |SHAP|: 0.141';
        if (featShare) featShare.innerText = '• 34.1% RF Gini • Rank 1';
        heroModelBadge.innerText = 'Random Forest Active';
        if (heroEngineTag) heroEngineTag.innerText = 'Random Forest Ensemble • TreeSHAP (Class 1)';
        legendText.innerText = 'Random Forest Predictor';
        const explainerModelBadge = document.getElementById('explainerModelBadge');
        if (explainerModelBadge) explainerModelBadge.innerText = 'Random Forest (Selected Best) • Th: 0.48';
        if (mTh) mTh.innerText = '0.48';
        if (gTh) gTh.innerText = 'Threshold: 0.48';
        if (bScore) bScore.innerText = 'Threshold: 0.48';
        showToast('Switched to Random Forest Model (Decision Threshold: 0.48)');
      }}

      // Recompute prediction with new model characteristics
      calculateRisk(true);
    }}

    function toggleModelFromCard() {{
      if (activeModel === 'xgboost') {{
        changeModel('random_forest');
      }} else {{
        changeModel('xgboost');
      }}
    }}

    function scrollToShap() {{
      switchTab('risk-prediction');
      setTimeout(() => {{
        const el = document.getElementById('shap-analysis-section');
        if (el) {{
          if (window.lenisInstance) {{
            window.lenisInstance.scrollTo(el, {{ offset: -70, duration: 0.9 }});
          }} else {{
            el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
          }}
          showToast('Navigated to SHAP Local Attribution');
        }}
      }}, 80);
    }}

    function switchImportanceView(view) {{
      const btnXgb = document.getElementById('btnImpXgb');
      const btnRf = document.getElementById('btnImpRf');
      const title = document.getElementById('importanceSectionTitle');
      const subtitle = document.getElementById('importanceSectionSubtitle');

      if (view === 'xgb') {{
        btnXgb.className = 'px-3 py-1 rounded-full text-label-sm font-semibold transition-colors bg-primary text-on-primary';
        btnRf.className = 'px-3 py-1 rounded-full text-label-sm font-medium transition-colors bg-surface-container text-on-surface-variant hover:text-on-surface';
        title.innerText = 'Global Feature Importance (XGBoost TreeSHAP)';
        subtitle.innerText = 'Mean absolute SHAP impact magnitude per biomarker across holdout test sets.';
        renderImportanceBars([
          {{ name: '1. Glucose Concentration', score: '0.932', pct: 100 }},
          {{ name: '2. Body Mass Index (BMI)', score: '0.523', pct: 56.1 }},
          {{ name: '3. Age', score: '0.277', pct: 29.7 }},
          {{ name: '4. Diabetes Pedigree', score: '0.231', pct: 24.8 }},
          {{ name: '5. Pregnancies Count', score: '0.152', pct: 16.3 }},
          {{ name: '6. Insulin Serum', score: '0.134', pct: 14.4 }},
          {{ name: '7. Skinfold Thickness', score: '0.055', pct: 5.9 }},
          {{ name: '8. Blood Pressure', score: '0.042', pct: 4.5 }}
        ]);
      }} else {{
        btnRf.className = 'px-3 py-1 rounded-full text-label-sm font-semibold transition-colors bg-primary text-on-primary';
        btnXgb.className = 'px-3 py-1 rounded-full text-label-sm font-medium transition-colors bg-surface-container text-on-surface-variant hover:text-on-surface';
        title.innerText = 'Global Feature Importance (Random Forest Gini Impurity)';
        subtitle.innerText = 'Normalized mean decrease in Gini impurity across 200 decision trees.';
        renderImportanceBars([
          {{ name: '1. Glucose Concentration', score: '0.341', pct: 100 }},
          {{ name: '2. Body Mass Index (BMI)', score: '0.185', pct: 54.2 }},
          {{ name: '3. Age', score: '0.130', pct: 38.0 }},
          {{ name: '4. Insulin Serum', score: '0.094', pct: 27.5 }},
          {{ name: '5. Diabetes Pedigree', score: '0.080', pct: 23.3 }},
          {{ name: '6. Pregnancies Count', score: '0.066', pct: 19.4 }},
          {{ name: '7. Skinfold Thickness', score: '0.056', pct: 16.4 }},
          {{ name: '8. Blood Pressure', score: '0.050', pct: 14.6 }}
        ]);
      }}
    }}

    function renderImportanceBars(items) {{
      const container = document.getElementById('importanceBarsContainer');
      container.innerHTML = '';
      items.forEach(it => {{
        const el = document.createElement('div');
        el.className = 'flex flex-col gap-1';
        el.innerHTML = `
          <div class="flex justify-between text-label-md text-on-surface">
            <span class="font-medium">${{it.name}}</span>
            <span class="font-code-num font-semibold">${{it.score}}</span>
          </div>
          <div class="h-2 rounded-full bg-surface-container overflow-hidden">
            <div class="h-full ${{activeModel === 'random_forest' ? 'bg-secondary' : 'bg-primary'}} rounded-full" style="width: ${{it.pct}}%;"></div>
          </div>
        `;
        container.appendChild(el);
      }});
    }}

    // Tab switching with URL Hash Routing & Browser History
    function switchTab(tabId, pushHistory = true) {{
      let targetTab = tabId;
      if (tabId === 'assessment' || tabId === 'risk-assessment') targetTab = 'risk-prediction';
      if (tabId === 'results' || tabId === 'prediction-results') targetTab = 'prediction-results';

      const tabs = ['risk-prediction', 'prediction-results', 'explainability-xai', 'model-benchmark-and-insights'];
      tabs.forEach(t => {{
        const content = document.getElementById('tab-content-' + t);
        const btn = document.getElementById('nav-' + t);
        if (content) {{
          if (t === targetTab) {{
            content.classList.remove('hidden');
          }} else {{
            content.classList.add('hidden');
          }}
        }}
        if (btn) {{
          const isXai = (t === 'explainability-xai');
          const isResults = (t === 'prediction-results');
          const prefix = (isXai || isResults) ? 'flex items-center gap-1.5 ' : '';
          
          if (t === targetTab) {{
            btn.className = prefix + 'px-space-md py-1.5 rounded-lg transition-all bg-primary text-on-primary shadow-sm font-semibold text-label-md';
            if (isXai) {{
              const b = document.getElementById('shapNavBadge');
              if (b) b.className = 'px-1.5 py-0.5 rounded bg-white/20 text-white text-[11px] font-bold';
            }}
          }} else {{
            btn.className = prefix + 'px-space-md py-1.5 rounded-lg text-label-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-all';
            if (isXai) {{
              const b = document.getElementById('shapNavBadge');
              if (b) b.className = 'px-1.5 py-0.5 rounded bg-primary-fixed text-primary text-[11px] font-bold';
            }}
          }}
        }}
      }});

      // Show/hide empty state if navigating to results
      if (targetTab === 'prediction-results') {{
        if (!currentResultRecord) {{
          document.getElementById('resultsEmptyState')?.classList.remove('hidden');
          document.getElementById('resultsContent')?.classList.add('hidden');
        }} else {{
          document.getElementById('resultsEmptyState')?.classList.add('hidden');
          document.getElementById('resultsContent')?.classList.remove('hidden');
        }}
      }}

      // Update URL hash for browser history
      if (pushHistory) {{
        const routeHash = targetTab === 'risk-prediction' ? 'assessment' : 
                          targetTab === 'prediction-results' ? 'results' : targetTab;
        if (window.location.hash !== '#' + routeHash) {{
          try {{
            history.pushState(null, '', '#' + routeHash);
          }} catch (e) {{
            window.location.hash = routeHash;
          }}
        }}
      }}

      if (window.lenisInstance) {{
        window.lenisInstance.scrollTo(0, {{ immediate: false, duration: 0.5 }});
      }} else {{
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}
      syncFrameHeight();
    }}

    // Browser back/forward navigation handler
    window.addEventListener('popstate', () => {{
      const hash = window.location.hash.replace('#', '');
      if (hash === 'results' || hash === 'prediction-results') {{
        switchTab('prediction-results', false);
      }} else if (hash === 'explainability-xai') {{
        switchTab('explainability-xai', false);
      }} else if (hash === 'model-benchmark-and-insights' || hash === 'benchmark') {{
        switchTab('model-benchmark-and-insights', false);
      }} else {{
        switchTab('risk-prediction', false);
      }}
    }});

    function syncFrameHeight() {{
      setTimeout(() => {{
        const h = Math.max(
          document.body ? document.body.scrollHeight : 0,
          document.documentElement ? document.documentElement.scrollHeight : 0,
          document.body ? document.body.offsetHeight : 0
        );
        if (h > 0) {{
          try {{
            if (window.frameElement) {{
              window.frameElement.style.height = h + 'px';
            }}
            if (window.parent && window.parent.document) {{
              const iframes = window.parent.document.querySelectorAll('iframe');
              iframes.forEach(f => {{
                f.style.height = h + 'px';
              }});
            }}
          }} catch (e) {{}}
          try {{
            window.parent.postMessage({{
              type: "streamlit:setFrameHeight",
              height: h
            }}, "*");
          }} catch (e) {{}}
        }}
      }}, 60);
    }}

    let activeTestCase = null;

    const TEST_CASES = {{
      'test-01': {{
        id: 'test-01',
        title: 'Test 01: Elevated Glycemia',
        badgeText: 'Test 01: Elevated Glycemia',
        tier: 'elevated',
        values: {{
          age: 48,
          preg: 2,
          gluc: 154,
          bp: 78,
          bmi: 31.4,
          ins: 130,
          skin: 32,
          ped: 0.672
        }},
        toastMsg: 'Loaded Test 01: Elevated Glycemia (Glucose 154 mg/dL)'
      }},
      'test-02': {{
        id: 'test-02',
        title: 'Test 02: Normoglycemic Baseline',
        badgeText: 'Test 02: Normoglycemic',
        tier: 'low',
        values: {{
          age: 28,
          preg: 0,
          gluc: 88,
          bp: 68,
          bmi: 21.8,
          ins: 65,
          skin: 19,
          ped: 0.214
        }},
        toastMsg: 'Loaded Test 02: Normoglycemic Baseline (Glucose 88 mg/dL)'
      }},
      'test-03': {{
        id: 'test-03',
        title: 'Test 03: Impaired Fasting Glucose',
        badgeText: 'Test 03: Impaired Glucose',
        tier: 'moderate',
        values: {{
          age: 41,
          preg: 1,
          gluc: 126,
          bp: 74,
          bmi: 28.2,
          ins: 95,
          skin: 24,
          ped: 0.450
        }},
        toastMsg: 'Loaded Test 03: Impaired Fasting Glucose (Glucose 126 mg/dL)'
      }}
    }};

    function updateGlucoseBadge(val) {{
      const badge = document.getElementById('glucoseBadge');
      if (!badge) return;
      if (val === '' || val === null || val === undefined) {{
        badge.className = 'hidden';
        badge.innerText = '';
        return;
      }}
      const num = parseFloat(val);
      if (isNaN(num)) {{
        badge.className = 'hidden';
        badge.innerText = '';
      }} else if (num >= 126) {{
        badge.className = 'inline-flex items-center px-2 py-0.5 rounded-full bg-error-container text-on-error-container text-label-sm font-semibold';
        badge.innerText = 'Elevated (≥126)';
      }} else if (num >= 100) {{
        badge.className = 'inline-flex items-center px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 text-label-sm font-semibold';
        badge.innerText = 'Impaired (100-125)';
      }} else {{
        badge.className = 'inline-flex items-center px-2 py-0.5 rounded-full bg-tertiary-container/50 text-on-tertiary-fixed-variant text-label-sm font-semibold';
        badge.innerText = 'Normal (<100)';
      }}
    }}

    function updatePedigreeBadge(val) {{
      const badge = document.getElementById('pedigreeBadge');
      if (!badge) return;
      if (val === '' || val === null || val === undefined) {{
        badge.className = 'hidden';
        badge.innerText = '';
        return;
      }}
      const num = parseFloat(val);
      if (isNaN(num)) {{
        badge.className = 'hidden';
        badge.innerText = '';
      }} else if (num >= 0.5) {{
        badge.className = 'inline-flex items-center text-label-sm text-error bg-error-container/40 px-3.5 py-2.5 rounded-lg border border-error/20 font-medium shrink-0';
        badge.innerText = 'High Familial Risk Factor';
      }} else if (num >= 0.25) {{
        badge.className = 'inline-flex items-center text-label-sm text-amber-800 bg-amber-100/60 px-3.5 py-2.5 rounded-lg border border-amber-300/40 font-medium shrink-0';
        badge.innerText = 'Moderate Familial Factor';
      }} else {{
        badge.className = 'inline-flex items-center text-label-sm text-tertiary bg-tertiary-container/40 px-3.5 py-2.5 rounded-lg border border-tertiary/20 font-medium shrink-0';
        badge.innerText = 'Low Familial Factor';
      }}
    }}

    function selectTestCase(caseId) {{
      const tc = TEST_CASES[caseId];
      if (!tc) return;

      activeTestCase = caseId;

      document.getElementById('inputAge').value = tc.values.age;
      document.getElementById('inputPregnancies').value = tc.values.preg;
      document.getElementById('inputGlucose').value = tc.values.gluc;
      document.getElementById('inputBP').value = tc.values.bp;
      document.getElementById('inputBMI').value = tc.values.bmi;
      document.getElementById('inputInsulin').value = tc.values.ins;
      document.getElementById('inputSkin').value = tc.values.skin;
      document.getElementById('inputPedigree').value = tc.values.ped;

      const cohortLabel = document.getElementById('currentCohortLabel');
      if (cohortLabel) cohortLabel.innerText = tc.badgeText;

      updateTestCaseCardsUI(caseId);
      updateGlucoseBadge(tc.values.gluc);
      updatePedigreeBadge(tc.values.ped);
      invalidateStaleResults('test_case_selected');
      showToast(tc.toastMsg);
    }}

    function updateTestCaseCardsUI(selectedId) {{
      ['01', '02', '03'].forEach(num => {{
        const cardId = 'test-' + num;
        const card = document.getElementById('testCard-' + num);
        if (!card) return;

        const isSelected = (selectedId !== null && cardId === selectedId);
        card.setAttribute('aria-pressed', isSelected ? 'true' : 'false');

        const statusText = card.querySelector('.test-card-status-text');
        const statusIcon = card.querySelector('.test-card-status-icon');
        const testBadge = card.querySelector('.test-badge-num');

        if (isSelected) {{
          card.className = 'test-case-card group relative p-4 rounded-xl border-2 border-primary bg-primary/5 shadow-sm ring-2 ring-primary/20 transition-all cursor-pointer flex flex-col justify-between gap-3';
          if (statusText) {{
            statusText.innerText = 'Active';
            statusText.parentElement.className = 'text-primary font-bold flex items-center gap-1 text-[12px]';
          }}
          if (statusIcon) statusIcon.innerText = 'check_circle';
          if (testBadge) testBadge.className = 'test-badge-num px-2 py-0.5 rounded-md text-[11px] font-bold font-mono tracking-wider bg-surface-container-high text-primary border border-primary/20';
        }} else {{
          card.className = 'test-case-card group relative p-4 rounded-xl border border-outline-variant/40 bg-surface-container-lowest hover:border-primary/50 hover:shadow-md transition-all cursor-pointer flex flex-col justify-between gap-3';
          if (statusText) {{
            statusText.innerText = 'Select';
            statusText.parentElement.className = 'text-outline font-medium flex items-center gap-1 text-[12px] group-hover:text-primary';
          }}
          if (statusIcon) statusIcon.innerText = 'radio_button_unchecked';
          if (testBadge) testBadge.className = 'test-badge-num px-2 py-0.5 rounded-md text-[11px] font-bold font-mono tracking-wider bg-surface-container text-outline border border-outline-variant/40';
        }}
      }});
    }}

    function handleFormInputChange(fieldId) {{
      if (activeTestCase !== null) {{
        activeTestCase = null;
        updateTestCaseCardsUI(null);
        const cohortLabel = document.getElementById('currentCohortLabel');
        if (cohortLabel) cohortLabel.innerText = 'Custom Biometrics';
      }}
      if (fieldId === 'inputGlucose') {{
        updateGlucoseBadge(document.getElementById('inputGlucose').value);
      }}
      if (fieldId === 'inputPedigree') {{
        updatePedigreeBadge(document.getElementById('inputPedigree').value);
      }}
    }}

    function loadCohort(type) {{
      if (type === 'high') selectTestCase('test-01');
      else if (type === 'low') selectTestCase('test-02');
      else if (type === 'moderate') selectTestCase('test-03');
    }}

    function resetForm() {{
      document.getElementById('inputAge').value = '';
      document.getElementById('inputPregnancies').value = '';
      document.getElementById('inputGlucose').value = '';
      document.getElementById('inputBP').value = '';
      document.getElementById('inputBMI').value = '';
      document.getElementById('inputInsulin').value = '';
      document.getElementById('inputSkin').value = '';
      document.getElementById('inputPedigree').value = '';
      
      activeTestCase = null;
      updateTestCaseCardsUI(null);
      updateGlucoseBadge('');
      updatePedigreeBadge('');

      const cohortLabel = document.getElementById('currentCohortLabel');
      if (cohortLabel) cohortLabel.innerText = 'Custom Biometrics';

      invalidateStaleResults('form_reset');
      dismissErrorBanner();
      showToast('Patient form reset to blank');
    }}

    function newAssessment() {{
      resetForm();
      currentResultRecord = null;
      const navBadge = document.getElementById('navResultsBadge');
      if (navBadge) navBadge.classList.add('hidden');
      switchTab('risk-prediction');
      showToast('Started new assessment. Form fields cleared.');
    }}

    function invalidateStaleResults(reason) {{
      if (currentResultRecord) {{
        currentResultRecord = null;
        const navBadge = document.getElementById('navResultsBadge');
        if (navBadge) {{
          navBadge.className = 'px-1.5 py-0.5 rounded bg-surface-container-high text-outline text-[11px] font-semibold';
          navBadge.innerText = 'Stale';
        }}
        const emptyMsg = document.getElementById('emptyStateMsg');
        if (emptyMsg) {{
          if (reason === 'model_changed') {{
            emptyMsg.innerText = 'Predictive model architecture was switched since the last calculation. Please return to the Assessment page and click "Estimate Diabetes Risk" to generate updated predictions for ' + (activeModel === 'random_forest' ? 'Random Forest' : 'XGBoost') + '.';
          }} else {{
            emptyMsg.innerText = 'Patient biometric inputs were modified since the last risk assessment. Please return to the Assessment page and click "Estimate Diabetes Risk" to re-calculate risk.';
          }}
        }}
        document.getElementById('resultsEmptyState')?.classList.remove('hidden');
        document.getElementById('resultsContent')?.classList.add('hidden');
      }}
    }}

    function showAssessmentError(title, msg) {{
      const banner = document.getElementById('assessmentErrorBanner');
      const titleEl = document.getElementById('errorBannerTitle');
      const msgEl = document.getElementById('errorBannerMsg');
      if (banner && titleEl && msgEl) {{
        titleEl.innerText = title;
        msgEl.innerText = msg;
        banner.classList.remove('hidden');
        banner.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
      }}
    }}

    function dismissErrorBanner() {{
      const banner = document.getElementById('assessmentErrorBanner');
      if (banner) banner.classList.add('hidden');
    }}

    function renderPatientInputSummary(inputs) {{
      const container = document.getElementById('patientInputSummaryTiles');
      if (!container) return;

      const items = [
        {{ label: 'Glucose (OGTT)', val: inputs.Glucose, unit: 'mg/dL', status: inputs.Glucose >= 126 ? 'Elevated' : (inputs.Glucose >= 100 ? 'Impaired' : 'Normal') }},
        {{ label: 'BMI', val: inputs.BMI, unit: 'kg/m²', status: inputs.BMI >= 30 ? 'Obese' : (inputs.BMI >= 25 ? 'Overweight' : 'Normal') }},
        {{ label: 'Blood Pressure', val: inputs.BloodPressure, unit: 'mm Hg', status: inputs.BloodPressure >= 90 ? 'High' : (inputs.BloodPressure >= 80 ? 'Pre-HTN' : 'Normal') }},
        {{ label: 'Age', val: inputs.Age, unit: 'years', status: inputs.Age >= 45 ? 'Mature' : 'Younger' }},
        {{ label: 'Pregnancies', val: inputs.Pregnancies, unit: '', status: inputs.Pregnancies >= 3 ? 'Multiparous' : 'Low Parity' }},
        {{ label: 'Serum Insulin', val: inputs.Insulin, unit: 'µU/mL', status: inputs.Insulin > 166 ? 'Elevated' : 'Normal' }},
        {{ label: 'Triceps Skinfold', val: inputs.SkinThickness, unit: 'mm', status: inputs.SkinThickness >= 30 ? 'Elevated' : 'Normal' }},
        {{ label: 'Pedigree Function', val: inputs.DiabetesPedigreeFunction, unit: '', status: inputs.DiabetesPedigreeFunction >= 0.5 ? 'High Risk' : 'Low Risk' }}
      ];

      container.innerHTML = items.map(item => `
        <div class="p-2.5 rounded-xl bg-surface-container-low border border-outline-variant/20 flex flex-col justify-between">
          <span class="text-[11px] text-outline truncate" title="${{item.label}}">${{item.label}}</span>
          <div class="flex items-baseline gap-1 my-0.5">
            <span class="text-label-lg font-bold font-code-num text-on-surface">${{item.val}}</span>
            ${{item.unit ? `<span class="text-[10px] text-outline">${{item.unit}}</span>` : ''}}
          </div>
          <span class="text-[10px] font-semibold ${{item.status.includes('Elevated') || item.status.includes('High') || item.status.includes('Obese') ? 'text-error' : 'text-secondary'}}">${{item.status}}</span>
        </div>
      `).join('');
    }}

    function renderResults(record) {{
      if (!record || !record.data) {{
        document.getElementById('resultsEmptyState')?.classList.remove('hidden');
        document.getElementById('resultsContent')?.classList.add('hidden');
        return;
      }}

      document.getElementById('resultsEmptyState')?.classList.add('hidden');
      document.getElementById('resultsContent')?.classList.remove('hidden');

      const data = record.data;
      const prob = data.probability;
      const probPercent = Math.round(prob * 100);
      const threshold = data.threshold;
      const tier = data.risk_tier;
      const isHigh = prob >= threshold;
      const label = isHigh ? 
        ('Elevated Risk (Threshold: ' + threshold.toFixed(2) + ')') : 
        ('Lower Risk (Threshold: ' + threshold.toFixed(2) + ')');
      const shapMap = data.shap_contributions;
      const baseVal = data.base_value;
      const outputSpace = data.output_space || (record.model === 'random_forest' ? 'probability' : 'log_odds');
      const marginOutput = data.margin_output !== undefined ? data.margin_output : (baseVal + Object.values(shapMap).reduce((acc, c) => acc + c.shap, 0));

      // Update Result Top Banner
      const recordBadge = document.getElementById('resultRecordBadge');
      if (recordBadge) recordBadge.innerText = record.id;
      const resultTimestamp = document.getElementById('resultTimestamp');
      if (resultTimestamp) resultTimestamp.innerText = 'Generated: ' + record.timestamp;
      const resultModelName = document.getElementById('resultModelName');
      if (resultModelName) {{
        resultModelName.innerText = record.model === 'random_forest' ? 
          'Random Forest Classifier (Selected Best • Optimal Threshold: ' + threshold.toFixed(2) + ')' : 
          'XGBoost Classifier (Tuned • Optimal Threshold: ' + threshold.toFixed(2) + ')';
      }}
      const resultBannerIcon = document.getElementById('resultBannerIcon');
      if (resultBannerIcon) {{
        resultBannerIcon.innerText = record.model === 'random_forest' ? 'forest' : 'memory';
      }}
      const resultThresholdBadge = document.getElementById('resultThresholdBadge');
      if (resultThresholdBadge) resultThresholdBadge.innerText = 'Decision Threshold: ' + threshold.toFixed(2);

      // Update Gauge, Pills, Spectrum, and Local SHAP Waterfall
      updateRiskOutput(probPercent, label, tier, shapMap, baseVal, threshold, outputSpace, marginOutput, prob);

      // Render Concise Patient Input Summary
      renderPatientInputSummary(record.inputs);

      // Update Navigation Badge
      const navBadge = document.getElementById('navResultsBadge');
      if (navBadge) {{
        navBadge.classList.remove('hidden');
        navBadge.className = 'px-1.5 py-0.5 rounded bg-secondary-container text-on-secondary-container text-[11px] font-bold';
        navBadge.innerText = probPercent + '% Risk';
      }}
    }}

    function updateRiskOutput(riskPercent, label, tier, shapValues, baseValue = 0.498, threshold = 0.48, outputSpace = 'probability', marginOutput = 0, prob = 0) {{
      const gaugeValue = document.getElementById('riskGaugeValue');
      const gaugeCircle = document.getElementById('riskGaugeCircle');
      const riskPill = document.getElementById('riskPill');
      const riskPillText = document.getElementById('riskPillText');
      const riskPillIcon = document.getElementById('riskPillIcon');
      const glucoseBadge = document.getElementById('glucoseBadge');
      const spectrumMarker = document.getElementById('spectrumMarker');
      const adviceTitle = document.getElementById('adviceTitle');
      const adviceText = document.getElementById('adviceText');
      const adviceIcon = document.getElementById('adviceIcon');
      const confidenceValue = document.getElementById('confidenceValue');
      const riskPhenotypeDesc = document.getElementById('riskPhenotypeDesc');
      const riskGaugeThreshold = document.getElementById('riskGaugeThreshold');
      const brierScore = document.getElementById('brierScore');

      gaugeValue.innerText = riskPercent + '%';
      
      // Circumference = 2 * PI * 82 ≈ 515.2
      const totalCircumference = 515.2;
      const offset = totalCircumference - (totalCircumference * (riskPercent / 100));
      gaugeCircle.style.strokeDashoffset = offset;

      spectrumMarker.style.left = Math.min(Math.max(riskPercent, 4), 94) + '%';

      if (riskGaugeThreshold) riskGaugeThreshold.innerText = 'Threshold: ' + threshold.toFixed(2);
      if (brierScore) brierScore.innerText = 'Threshold: ' + threshold.toFixed(2);

      const displayProb = prob > 0 ? (prob * 100).toFixed(1) : riskPercent;
      confidenceValue.innerText = displayProb + '% Predicted Probability';

      if (tier === 'elevated') {{
        gaugeCircle.setAttribute('stroke', '#ba1a1a');
        riskPill.className = 'mt-1 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-error-container text-on-error-container text-label-lg font-bold';
        riskPillText.innerText = label;
        riskPillIcon.innerText = 'warning';
        glucoseBadge.className = 'inline-flex items-center px-2 py-0.5 rounded-full bg-error-container text-on-error-container text-label-sm font-semibold';
        glucoseBadge.innerText = 'Elevated (≥126)';
        riskPhenotypeDesc.innerText = 'Estimated diabetes risk (' + displayProb + '%) meets or exceeds the decision threshold (' + threshold.toFixed(2) + ').';
        if (adviceTitle) adviceTitle.innerText = 'Model Assessment Context';
        if (adviceText) adviceText.innerText = 'The model estimates elevated risk based on patient physiological inputs. Model output is an educational/research estimate and does not constitute a clinical diagnosis.';
        if (adviceIcon) {{
          adviceIcon.innerText = 'clinical_notes';
          adviceIcon.className = 'material-symbols-outlined text-error text-[22px] mt-0.5';
        }}
      }} else if (tier === 'low') {{
        gaugeCircle.setAttribute('stroke', '#006329');
        riskPill.className = 'mt-1 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-tertiary-fixed text-on-tertiary-fixed-variant text-label-lg font-bold';
        riskPillText.innerText = label;
        riskPillIcon.innerText = 'check_circle';
        glucoseBadge.className = 'inline-flex items-center px-2 py-0.5 rounded-full bg-tertiary-container/50 text-on-tertiary-fixed-variant text-label-sm font-semibold';
        glucoseBadge.innerText = 'Normal (<100)';
        riskPhenotypeDesc.innerText = 'Estimated diabetes risk (' + displayProb + '%) is below the decision threshold (' + threshold.toFixed(2) + ').';
        if (adviceTitle) adviceTitle.innerText = 'Model Assessment Context';
        if (adviceText) adviceText.innerText = 'The model estimates low risk for this feature profile relative to the decision threshold. Routine preventative screening remains standard practice.';
        if (adviceIcon) {{
          adviceIcon.innerText = 'verified';
          adviceIcon.className = 'material-symbols-outlined text-tertiary text-[22px] mt-0.5';
        }}
      }} else if (tier === 'moderate') {{
        gaugeCircle.setAttribute('stroke', '#d97706');
        riskPill.className = 'mt-1 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-amber-100 text-amber-900 text-label-lg font-bold';
        riskPillText.innerText = label;
        riskPillIcon.innerText = 'info';
        glucoseBadge.className = 'inline-flex items-center px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 text-label-sm font-semibold';
        glucoseBadge.innerText = 'Impaired (100-125)';
        riskPhenotypeDesc.innerText = 'Estimated diabetes risk (' + displayProb + '%) is near the decision threshold (' + threshold.toFixed(2) + ').';
        if (adviceTitle) adviceTitle.innerText = 'Model Assessment Context';
        if (adviceText) adviceText.innerText = 'The model estimates intermediate risk for this profile. Clinical laboratory evaluation is recommended for definitive glycemic assessment.';
        if (adviceIcon) {{
          adviceIcon.innerText = 'alarm';
          adviceIcon.className = 'material-symbols-outlined text-amber-600 text-[22px] mt-0.5';
        }}
      }} else {{
        gaugeCircle.setAttribute('stroke', '#737686');
        riskPill.className = 'mt-1 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-surface-container text-on-surface-variant text-label-lg font-bold';
        riskPillText.innerText = label;
        riskPillIcon.innerText = 'pending';
        glucoseBadge.className = 'hidden';
      }}

      // Update Instant SHAP Summary inside gauge card
      renderInstantShapSummary(shapValues);

      // Compute Net SHAP and Driver Stats
      let netShap = 0;
      Object.values(shapValues).forEach(v => {{
        if (v && typeof v.shap === 'number') {{
          netShap += v.shap;
        }}
      }});

      // Update Header Badges in Explainability Section
      const expBadge = document.getElementById('explainerModelBadge');
      if (expBadge) {{
        expBadge.innerText = (activeModel === 'random_forest' ? 'Random Forest (Selected Best) • Th: ' : 'XGBoost (Tuned) • Th: ') + threshold.toFixed(2);
      }}
      const netBadge = document.getElementById('explainabilityNetBadge');
      if (netBadge) {{
        netBadge.innerText = (netShap >= 0 ? '+' : '') + netShap.toFixed(3) + ' Net SHAP';
      }}

      // Update Waterfall text and formula
      const waterfallFormulaText = document.getElementById('waterfallFormulaText');
      if (waterfallFormulaText) {{
        const pStr = (prob > 0 ? prob.toFixed(3) : (riskPercent / 100).toFixed(3));
        if (outputSpace === 'probability') {{
          waterfallFormulaText.innerHTML = 'Features contribute additively in <strong>probability space</strong>: Expected baseline <code class="bg-surface px-1.5 py-0.5 rounded text-primary font-mono font-semibold">E[f(x)] = ' + baseValue.toFixed(3) + '</code> + Net SHAP <code class="bg-surface px-1.5 py-0.5 rounded text-secondary font-mono font-semibold">' + (netShap >= 0 ? '+' : '') + netShap.toFixed(3) + '</code> = Predicted Risk Probability <code class="bg-surface px-1.5 py-0.5 rounded text-error font-mono font-bold">p = ' + pStr + '</code>.';
        }} else {{
          waterfallFormulaText.innerHTML = 'Features contribute additively in <strong>log-odds margin space</strong>: Baseline margin <code class="bg-surface px-1.5 py-0.5 rounded text-primary font-mono font-semibold">E[f(x)] = ' + baseValue.toFixed(3) + '</code> + Net SHAP <code class="bg-surface px-1.5 py-0.5 rounded text-secondary font-mono font-semibold">' + (netShap >= 0 ? '+' : '') + netShap.toFixed(3) + '</code> = Margin <code class="bg-surface px-1.5 py-0.5 rounded text-secondary font-mono font-semibold">f(x) = ' + marginOutput.toFixed(3) + '</code> → <code class="bg-surface px-1.5 py-0.5 rounded text-error font-mono font-bold">p = ' + pStr + '</code> (via sigmoid).';
        }}
      }}
      
      // Update Divergence Rows (single waterfall visualization)
      renderDivergenceChart(shapValues);
      
      // Update Compact Key Drivers (positive & negative summaries)
      renderFactorCards(shapValues);
    }}

    function renderInstantShapSummary(shapMap) {{
      const container = document.getElementById('instantShapPills');
      const badge = document.getElementById('shapSummaryBadge');
      if (!container) return;
      container.innerHTML = '';

      let net = 0;
      const sorted = Object.entries(shapMap).sort((a,b) => Math.abs(b[1].shap) - Math.abs(a[1].shap));
      sorted.forEach(([_, info]) => net += info.shap);
      if (badge) badge.innerText = (net >= 0 ? '+' : '') + net.toFixed(2) + ' Net SHAP';

      // Pick top 3 absolute features
      const top3 = sorted.slice(0, 3);
      top3.forEach(([feat, info]) => {{
        const isPos = info.shap >= 0;
        const row = document.createElement('div');
        row.className = 'flex items-center justify-between text-[13px]';
        row.innerHTML = `
          <span class="text-on-surface font-medium">${{feat.replace('DiabetesPedigreeFunction', 'Pedigree')}} (${{info.val}})</span>
          <span class="font-code-num font-bold ${{isPos ? 'text-error' : 'text-secondary'}}">${{isPos ? '+' : ''}}${{info.shap.toFixed(2)}} SHAP</span>
        `;
        container.appendChild(row);
      }});
    }}

    function renderCumulativeStepper(baseValue, shapMap, finalScore, outputSpace = 'probability', marginOutput = 0) {{
      // Retained as clean no-op for backward compatibility after consolidation
      const container = document.getElementById('stepperContainer');
      if (container) container.innerHTML = '';
    }}

    function renderDivergenceChart(shapMap) {{
      const container = document.getElementById('divergenceRows');
      if (!container) return;
      container.innerHTML = '';

      const features = [
        {{ key: 'Glucose', label: 'Glucose', unit: 'mg/dL' }},
        {{ key: 'BMI', label: 'BMI', unit: 'kg/m²' }},
        {{ key: 'Age', label: 'Age', unit: 'yr' }},
        {{ key: 'DiabetesPedigreeFunction', label: 'Pedigree', unit: '' }},
        {{ key: 'Pregnancies', label: 'Pregnancies', unit: '' }},
        {{ key: 'Insulin', label: 'Insulin', unit: 'µU/mL' }},
        {{ key: 'BloodPressure', label: 'Blood Pressure', unit: 'mm Hg' }},
        {{ key: 'SkinThickness', label: 'Skinfold', unit: 'mm' }}
      ];

      // Calculate max absolute SHAP for responsive, proportional scaling
      let maxAbs = 0.05;
      features.forEach(f => {{
        const item = shapMap[f.key] || {{ shap: 0 }};
        if (Math.abs(item.shap) > maxAbs) maxAbs = Math.abs(item.shap);
      }});

      features.forEach(f => {{
        const item = shapMap[f.key] || {{ shap: 0, val: 0 }};
        const shap = item.shap;
        const val = item.val;
        const isPos = shap >= 0;
        const absVal = Math.abs(shap);
        const barWidth = Math.min(Math.round((absVal / maxAbs) * 88), 92);

        const row = document.createElement('div');
        row.className = 'grid grid-cols-12 items-center text-body-sm py-1.5 hover:bg-surface-container/50 px-2 rounded-lg transition-colors';
        
        const labelText = `${{f.label}} (${{val}}${{f.unit ? ' ' + f.unit : ''}})`;

        if (isPos) {{
          row.innerHTML = `
            <div class="col-span-4 font-semibold text-on-surface truncate" title="${{labelText}}">${{labelText}}</div>
            <div class="col-span-3 flex justify-end"></div>
            <div class="col-span-1 flex justify-center">
              <div class="w-0.5 h-6 bg-outline-variant"></div>
            </div>
            <div class="col-span-4 flex items-center gap-2">
              <div class="h-5 bg-error rounded-r-md transition-all duration-500" style="width: ${{barWidth}}%;"></div>
              <span class="font-code-num text-label-sm text-error font-bold">+${{shap.toFixed(2)}}</span>
            </div>
          `;
        }} else {{
          row.innerHTML = `
            <div class="col-span-4 font-semibold text-on-surface truncate" title="${{labelText}}">${{labelText}}</div>
            <div class="col-span-3 flex items-center justify-end gap-2">
              <span class="font-code-num text-label-sm text-secondary font-bold">${{shap.toFixed(2)}}</span>
              <div class="h-5 bg-secondary rounded-l-md transition-all duration-500" style="width: ${{barWidth}}%;"></div>
            </div>
            <div class="col-span-1 flex justify-center">
              <div class="w-0.5 h-6 bg-outline-variant"></div>
            </div>
            <div class="col-span-4"></div>
          `;
        }}

        container.appendChild(row);
      }});
    }}

    function renderFactorCards(shapMap) {{
      const incContainer = document.getElementById('increasingFactorsList');
      const decContainer = document.getElementById('decreasingFactorsList');
      const netPosBadge = document.getElementById('netPosBadge');
      const netNegBadge = document.getElementById('netNegBadge');
      const primaryDriverText = document.getElementById('primaryDriverText');

      if (!incContainer || !decContainer) return;

      incContainer.innerHTML = '';
      decContainer.innerHTML = '';

      let sumPos = 0;
      let sumNeg = 0;
      const posList = [];
      const negList = [];

      const explanations = {{
        Glucose: {{
          pos: "Elevated glycemia is the primary risk-increasing factor.",
          neg: "Normoglycemic glucose level lowers estimated diabetes probability."
        }},
        BMI: {{
          pos: "Body mass index contributes toward elevated metabolic risk.",
          neg: "Healthy BMI level exerts a protective contribution."
        }},
        Age: {{
          pos: "Age profile adds positive attribution to estimated risk.",
          neg: "Younger age bracket contributes toward lower risk."
        }},
        DiabetesPedigreeFunction: {{
          pos: "Familial diabetes history increases model risk output.",
          neg: "Low pedigree score contributes protectively."
        }},
        BloodPressure: {{
          pos: "Elevated blood pressure adds moderate risk attribution.",
          neg: "Diastolic blood pressure reading provides protective contribution."
        }},
        Pregnancies: {{
          pos: "Higher pregnancy parity contributes toward risk.",
          neg: "Lower parity contributes toward lower predicted risk."
        }},
        SkinThickness: {{
          pos: "Triceps skinfold contributes positively to risk attribution.",
          neg: "Skinfold measurement contributes protectively."
        }},
        Insulin: {{
          pos: "Serum insulin reading adds positive risk contribution.",
          neg: "Normal fasting insulin reduces estimated risk probability."
        }}
      }};

      Object.entries(shapMap).forEach(([feat, info]) => {{
        if (info.shap > 0.0001) {{
          sumPos += info.shap;
          posList.push({{ feat, ...info }});
        }} else if (info.shap < -0.0001) {{
          sumNeg += info.shap;
          negList.push({{ feat, ...info }});
        }}
      }});

      posList.sort((a,b) => b.shap - a.shap);
      negList.sort((a,b) => a.shap - b.shap);

      if (netPosBadge) netPosBadge.innerText = '+' + sumPos.toFixed(2) + ' Net';
      if (netNegBadge) netNegBadge.innerText = sumNeg.toFixed(2) + ' Net';

      if (primaryDriverText) {{
        if (posList.length > 0) {{
          const topDriver = posList[0];
          const pct = Math.round((topDriver.shap / (sumPos || 1)) * 100);
          primaryDriverText.innerHTML = `<span class="material-symbols-outlined text-[18px] text-primary">info</span> <span><strong>Primary Risk Driver:</strong> ${{topDriver.feat}} (${{topDriver.val}}) accounts for ~${{pct}}% of positive attribution (+${{topDriver.shap.toFixed(2)}} SHAP).</span>`;
        }} else if (negList.length > 0) {{
          const topProt = negList[0];
          primaryDriverText.innerHTML = `<span class="material-symbols-outlined text-[18px] text-secondary">verified</span> <span><strong>Favorable Profile:</strong> ${{topProt.feat}} (${{topProt.val}}) is the primary protective factor (${{topProt.shap.toFixed(2)}} SHAP).</span>`;
        }} else {{
          primaryDriverText.innerHTML = `<span class="material-symbols-outlined text-[18px] text-outline">tune</span> <span>Neutral risk attribution profile across evaluated biomarkers.</span>`;
        }}
      }}

      // Render top positive drivers (limit to top 2 for compact presentation)
      if (posList.length === 0) {{
        incContainer.innerHTML = '<div class="text-body-sm text-outline italic py-1">No positive risk-increasing contributions.</div>';
      }} else {{
        posList.slice(0, 2).forEach(item => {{
          const el = document.createElement('div');
          el.className = 'flex flex-col gap-1 p-2 rounded-lg bg-surface border border-outline-variant/20';
          const desc = explanations[item.feat]?.pos || 'Contributes toward higher estimated risk.';
          el.innerHTML = `
            <div class="flex items-center justify-between text-label-sm">
              <span class="font-bold text-on-surface">${{item.feat}} <span class="font-code-num font-normal text-on-surface-variant">(${{item.val}})</span></span>
              <span class="font-code-num font-bold text-error">+${{item.shap.toFixed(2)}} SHAP</span>
            </div>
            <span class="text-[12px] text-on-surface-variant leading-snug">${{desc}}</span>
          `;
          incContainer.appendChild(el);
        }});
      }}

      // Render top negative/protective drivers (limit to top 2 for compact presentation)
      if (negList.length === 0) {{
        decContainer.innerHTML = '<div class="text-body-sm text-outline italic py-1">No protective contributions.</div>';
      }} else {{
        negList.slice(0, 2).forEach(item => {{
          const el = document.createElement('div');
          el.className = 'flex flex-col gap-1 p-2 rounded-lg bg-surface border border-outline-variant/20';
          const desc = explanations[item.feat]?.neg || 'Contributes toward lower estimated risk.';
          el.innerHTML = `
            <div class="flex items-center justify-between text-label-sm">
              <span class="font-bold text-on-surface">${{item.feat}} <span class="font-code-num font-normal text-on-surface-variant">(${{item.val}})</span></span>
              <span class="font-code-num font-bold text-secondary">${{item.shap.toFixed(2)}} SHAP</span>
            </div>
            <span class="text-[12px] text-on-surface-variant leading-snug">${{desc}}</span>
          `;
          decContainer.appendChild(el);
        }});
      }}
    }}

    function validateFormInputs() {{
      const fields = [
        {{ id: 'inputGlucose', name: 'Glucose (OGTT)', min: 40, max: 300 }},
        {{ id: 'inputBMI', name: 'Body Mass Index (BMI)', min: 10, max: 70 }},
        {{ id: 'inputAge', name: 'Age', min: 1, max: 120 }},
        {{ id: 'inputBP', name: 'Blood Pressure', min: 30, max: 160 }},
        {{ id: 'inputPregnancies', name: 'Pregnancies', min: 0, max: 20 }},
        {{ id: 'inputInsulin', name: 'Serum Insulin', min: 0, max: 900 }},
        {{ id: 'inputSkin', name: 'Triceps Skinfold', min: 0, max: 99 }},
        {{ id: 'inputPedigree', name: 'Diabetes Pedigree Function', min: 0.01, max: 3.0 }}
      ];

      for (const f of fields) {{
        const el = document.getElementById(f.id);
        if (!el) continue;
        const v = el.value.trim();
        if (v === '') {{
          showToast('Validation Error: ' + f.name + ' is required');
          el.focus();
          return false;
        }}
        const num = parseFloat(v);
        if (isNaN(num)) {{
          showToast('Validation Error: ' + f.name + ' must be a valid number');
          el.focus();
          return false;
        }}
        if (num < f.min || num > f.max) {{
          showToast('Validation Error: ' + f.name + ' must be between ' + f.min + ' and ' + f.max);
          el.focus();
          return false;
        }}
      }}
      return true;
    }}

    async function calculateRisk() {{
      // Prevent duplicate concurrent requests
      if (isCalculating) return;

      // Validate inputs
      if (!validateFormInputs()) {{
        return;
      }}

      dismissErrorBanner();

      const glucVal = parseFloat(document.getElementById('inputGlucose').value);
      const bmiVal = parseFloat(document.getElementById('inputBMI').value);
      const ageVal = parseFloat(document.getElementById('inputAge').value);
      const bpVal = parseFloat(document.getElementById('inputBP').value);
      const pregVal = parseFloat(document.getElementById('inputPregnancies').value);
      const insVal = parseFloat(document.getElementById('inputInsulin').value);
      const skinVal = parseFloat(document.getElementById('inputSkin').value);
      const pedVal = parseFloat(document.getElementById('inputPedigree').value);

      const submittedInputs = {{
        Pregnancies: pregVal,
        Glucose: glucVal,
        BloodPressure: bpVal,
        SkinThickness: skinVal,
        Insulin: insVal,
        BMI: bmiVal,
        DiabetesPedigreeFunction: pedVal,
        Age: ageVal
      }};

      const payload = {{
        model: activeModel,
        features: submittedInputs
      }};

      const btn = document.getElementById('btnCalculate');
      const icon = document.getElementById('calcIcon');
      const text = document.getElementById('calcText');
      const modal = document.getElementById('inferenceTransitionModal');
      const progressBar = document.getElementById('inferenceProgressBar');
      const modelTag = document.getElementById('inferenceModelTag');

      if (modelTag) {{
        modelTag.innerText = activeModel === 'random_forest' ? 'Random Forest (Selected Best)' : 'XGBoost (Tuned)';
      }}

      isCalculating = true;
      const thisRequestId = ++activeRequestId;

      if (btn) btn.disabled = true;
      if (icon) {{
        icon.innerText = 'sync';
        icon.classList.add('animate-spin');
      }}
      if (text) text.innerText = 'Running Model Inference...';

      // Helper to update stages in 4-stage animated transition experience
      function setStage(stageNum, status, isDone = false) {{
        for (let s = 1; s <= 4; s++) {{
          const item = document.getElementById('stageItem' + s);
          const iconEl = document.getElementById('stageIcon' + s);
          const statusEl = document.getElementById('stageStatus' + s);
          if (!item || !iconEl || !statusEl) continue;

          if (s < stageNum || (s === stageNum && isDone)) {{
            item.className = 'flex items-start gap-3 p-3 rounded-xl bg-surface-container border border-secondary/30 transition-all';
            iconEl.className = 'w-7 h-7 rounded-lg bg-secondary text-on-secondary flex items-center justify-center text-label-sm font-bold shrink-0 mt-0.5 shadow-xs';
            iconEl.innerHTML = '<span class="material-symbols-outlined text-[16px]">check_circle</span>';
            statusEl.className = 'text-[11px] font-mono font-bold text-secondary';
            statusEl.innerText = 'Completed';
          }} else if (s === stageNum) {{
            item.className = 'flex items-start gap-3 p-3 rounded-xl bg-surface-container-high border-2 border-primary shadow-sm transition-all';
            iconEl.className = 'w-7 h-7 rounded-lg bg-primary text-on-primary flex items-center justify-center text-label-sm font-bold shrink-0 mt-0.5 shadow-xs';
            iconEl.innerHTML = '<span class="material-symbols-outlined text-[16px] animate-spin">sync</span>';
            statusEl.className = 'text-[11px] font-mono font-bold text-primary';
            statusEl.innerText = status || 'Processing...';
          }} else {{
            item.className = 'flex items-start gap-3 p-3 rounded-xl bg-surface-container-low/50 border border-outline-variant/20 transition-all opacity-60';
            iconEl.className = 'w-7 h-7 rounded-lg bg-surface-container text-outline flex items-center justify-center text-label-sm font-bold shrink-0 mt-0.5';
            iconEl.innerHTML = '<span>' + s + '</span>';
            statusEl.className = 'text-[11px] font-mono font-bold text-outline';
            statusEl.innerText = 'Queued';
          }}
        }}
        if (progressBar) {{
          const pcts = [25, 50, 75, 100];
          progressBar.style.width = pcts[stageNum - 1] + '%';
        }}
      }}

      // Display animated four-stage transition experience
      if (modal) modal.classList.remove('hidden');
      setStage(1, 'Validating biometrics...');

      const port = (window.PREDICTION_API_PORT || 8502);
      const host = (window.location && window.location.hostname && window.location.hostname !== '') ? window.location.hostname : '127.0.0.1';
      const apiUrl = 'http://' + host + ':' + port + '/api/predict';

      const delay = (ms) => new Promise(res => setTimeout(res, ms));

      try {{
        const fetchPromise = fetch(apiUrl, {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});

        await delay(350);
        setStage(2, 'Running ensemble...');

        await delay(350);
        setStage(3, 'Calculating TreeSHAP...');

        const response = await fetchPromise;

        if (!response.ok) {{
          const errData = await response.json().catch(() => ({{}}));
          throw new Error(errData.detail || errData.error || ('HTTP ' + response.status));
        }}

        const data = await response.json();

        // Prevent late responses from replacing newer results
        if (thisRequestId !== activeRequestId) {{
          console.warn("Discarding stale inference response for request #" + thisRequestId);
          return;
        }}

        await delay(350);
        setStage(4, 'Synthesizing interpretation...', true);

        await delay(300);

        // Store result record
        currentResultRecord = {{
          id: 'REC-' + Math.floor(100000 + Math.random() * 900000),
          timestamp: new Date().toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }}),
          model: activeModel,
          modelName: data.model_name || (activeModel === 'random_forest' ? 'Random Forest' : 'XGBoost'),
          inputs: submittedInputs,
          data: data,
          isStale: false
        }};

        // Render dedicated results view
        renderResults(currentResultRecord);

        // Hide transition modal
        if (modal) modal.classList.add('hidden');

        // Navigate to dedicated results page
        switchTab('prediction-results');

        showToast(data.model_name + ' TreeSHAP inference completed (p=' + (data.probability * 100).toFixed(1) + '%)');

      }} catch (err) {{
        console.error('Inference API error:', err);
        if (modal) modal.classList.add('hidden');

        // Remain on assessment page and show useful error and retry action
        showAssessmentError('Inference Service Unreachable', 'Backend Error: ' + err.message + ' (Ensure inference server is running on port ' + port + ')');
        showToast('Backend Error: ' + err.message);
      }} finally {{
        isCalculating = false;
        if (btn) btn.disabled = false;
        if (icon) {{
          icon.classList.remove('animate-spin');
          icon.innerText = 'calculate';
        }}
        if (text) text.innerText = 'Estimate Diabetes Risk';
      }}
    }}

    function exportReport() {{
      const gluc = document.getElementById('inputGlucose').value || '154';
      const bp = document.getElementById('inputBP').value || '78';
      const bmi = document.getElementById('inputBMI').value || '31.4';
      const age = document.getElementById('inputAge').value || '48';
      const preg = document.getElementById('inputPregnancies').value || '2';
      const ins = document.getElementById('inputInsulin').value || '130';
      const skin = document.getElementById('inputSkin').value || '32';
      const ped = document.getElementById('inputPedigree').value || '0.672';
      const risk = document.getElementById('riskGaugeValue').innerText;
      const riskLabel = document.getElementById('riskPillText').innerText;
      const modelDisplayName = activeModel === 'random_forest' ? 'Random Forest Classifier (Optimal Threshold: 0.48)' : 'XGBoost Classifier (Optimal Threshold: 0.42)';

      const reportHtml = `
        <div class="p-4 rounded-xl bg-surface-container-low flex flex-col gap-2">
          <div class="flex justify-between items-center">
            <span class="text-label-md font-bold text-on-surface">Patient Identifier:</span>
            <span class="font-mono text-outline">ANON-${{Math.floor(1000 + Math.random() * 9000)}}</span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-label-md font-bold text-on-surface">Assessment Date:</span>
            <span>${{new Date().toLocaleDateString('en-US', {{ year: 'numeric', month: 'long', day: 'numeric' }})}}</span>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-label-md font-bold text-on-surface">Active Model Architecture:</span>
            <span class="font-bold text-primary">${{modelDisplayName}} • TreeSHAP</span>
          </div>
        </div>

        <div class="border border-outline-variant/40 rounded-xl p-4 flex items-center justify-between">
          <div>
            <span class="text-label-sm text-outline block">Estimated Diabetes Risk</span>
            <span class="text-headline-lg font-bold text-primary">${{risk}}</span>
          </div>
          <span class="px-3 py-1 rounded-full text-label-md font-bold bg-error-container text-on-error-container">${{riskLabel}}</span>
        </div>

        <div>
          <h4 class="text-label-md font-bold text-on-surface mb-2">Recorded Physiological Markers</h4>
          <div class="grid grid-cols-2 gap-2 text-label-sm">
            <div class="p-2 bg-surface rounded">Glucose (OGTT): <strong>${{gluc}} mg/dL</strong></div>
            <div class="p-2 bg-surface rounded">BMI: <strong>${{bmi}} kg/m²</strong></div>
            <div class="p-2 bg-surface rounded">Blood Pressure: <strong>${{bp}} mm Hg</strong></div>
            <div class="p-2 bg-surface rounded">Age: <strong>${{age}} years</strong></div>
            <div class="p-2 bg-surface rounded">Pregnancies: <strong>${{preg}}</strong></div>
            <div class="p-2 bg-surface rounded">Serum Insulin: <strong>${{ins}} µU/mL</strong></div>
            <div class="p-2 bg-surface rounded">Triceps Skinfold: <strong>${{skin}} mm</strong></div>
            <div class="p-2 bg-surface rounded">Pedigree Function: <strong>${{ped}}</strong></div>
          </div>
        </div>

        <div class="p-3 rounded-lg bg-surface-container text-[12px] text-on-surface-variant">
          <strong>Notice:</strong> This model is an educational/research prototype and does not replace professional medical assessment or provide a clinical diagnosis. Final clinical evaluations must be performed by a qualified healthcare professional.
        </div>
      `;

      document.getElementById('reportBody').innerHTML = reportHtml;
      document.getElementById('reportModal').classList.remove('hidden');
      if (window.lenisInstance) window.lenisInstance.stop();
    }}

    function copyPatientJSON() {{
      const payload = {{
        patient_id: "ANON-" + Math.floor(1000 + Math.random() * 9000),
        timestamp: new Date().toISOString(),
        active_model: activeModel === 'random_forest' ? "Random Forest Classifier" : "XGBoost Classifier",
        biometrics: {{
          glucose_ogtt_mg_dl: parseFloat(document.getElementById('inputGlucose').value) || 154,
          blood_pressure_mm_hg: parseFloat(document.getElementById('inputBP').value) || 78,
          bmi_kg_m2: parseFloat(document.getElementById('inputBMI').value) || 31.4,
          age_years: parseInt(document.getElementById('inputAge').value) || 48,
          pregnancies: parseInt(document.getElementById('inputPregnancies').value) || 2,
          insulin_uU_ml: parseFloat(document.getElementById('inputInsulin').value) || 130,
          triceps_skinfold_mm: parseFloat(document.getElementById('inputSkin').value) || 32,
          pedigree_function: parseFloat(document.getElementById('inputPedigree').value) || 0.672
        }},
        model_prediction: {{
          estimated_risk_probability: document.getElementById('riskGaugeValue').innerText,
          risk_classification: document.getElementById('riskPillText').innerText,
          decision_threshold: activeModel === 'random_forest' ? 0.48 : 0.42,
          engine: "TreeSHAP"
        }}
      }};
      navigator.clipboard?.writeText(JSON.stringify(payload, null, 2));
      showToast('Copied Patient Data & SHAP Values to Clipboard');
    }}

    // Butter-smooth momentum scrolling (Lenis)
    try {{
      if (typeof Lenis !== 'undefined') {{
        const lenis = new Lenis({{
          duration: 1.1,
          easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
          smoothWheel: true,
          wheelMultiplier: 1.0,
          touchMultiplier: 1.5,
          infinite: false
        }});
        window.lenisInstance = lenis;

        function raf(time) {{
          lenis.raf(time);
          requestAnimationFrame(raf);
        }}
        requestAnimationFrame(raf);
      }}
    }} catch (e) {{
      console.warn('Lenis smooth scrolling fallback to native CSS:', e);
    }}

    // Initialize application state and event listeners on load
    window.addEventListener('DOMContentLoaded', () => {{
      changeModel('random_forest');
      switchImportanceView('rf');

      // Attach invalidation & change listeners to all 8 form inputs
      ['inputGlucose', 'inputBMI', 'inputAge', 'inputBP', 'inputPregnancies', 'inputInsulin', 'inputSkin', 'inputPedigree'].forEach(id => {{
        const el = document.getElementById(id);
        if (el) {{
          el.addEventListener('input', () => {{
            handleFormInputChange(id);
            invalidateStaleResults('inputs_changed');
          }});
          el.addEventListener('change', () => {{
            handleFormInputChange(id);
            invalidateStaleResults('inputs_changed');
          }});
        }}
      }});

      // Route based on URL hash
      const hash = window.location.hash.replace('#', '');
      if (hash === 'results' || hash === 'prediction-results') {{
        switchTab('prediction-results', false);
      }} else if (hash === 'explainability-xai') {{
        switchTab('explainability-xai', false);
      }} else if (hash === 'model-benchmark-and-insights' || hash === 'benchmark') {{
        switchTab('model-benchmark-and-insights', false);
      }} else {{
        switchTab('risk-prediction', false);
      }}

      syncFrameHeight();
    }});
    window.addEventListener('resize', syncFrameHeight);
  </script>
</body>
</html>
'''

with open('frontend/index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print('Updated frontend/index.html successfully with Random Forest and prominent SHAP features!')
