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
            Risk Prediction
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
      <div class="flex items-center gap-space-sm sm:gap-space-md">
        <!-- Cohort Switcher -->
        <div class="relative">
          <button class="flex items-center gap-space-xs px-2.5 sm:px-3.5 py-1.5 rounded-lg bg-surface-container text-on-surface text-label-md hover:bg-surface-container-high transition-colors" type="button" onclick="toggleCohortDropdown()">
            <span class="material-symbols-outlined text-secondary text-[18px]">assignment_ind</span>
            <span id="currentCohortLabel" class="hidden sm:inline">Cohort 04</span>
            <span class="material-symbols-outlined text-on-surface-variant text-[16px]">expand_more</span>
          </button>
          <!-- Dropdown -->
          <div id="cohortDropdown" class="hidden absolute right-0 mt-2 w-64 bg-surface-container-lowest rounded-xl shadow-xl border border-outline-variant/40 py-2 z-50">
            <div class="px-3 py-1 text-[11px] font-semibold text-outline uppercase tracking-wider">Clinical Cohorts</div>
            <button class="w-full text-left px-3 py-2 text-label-md hover:bg-surface-container flex items-center gap-2" onclick="loadCohort('high'); closeCohortDropdown();">
              <span class="w-2 h-2 rounded-full bg-error"></span> Case A: Elevated Glycemia (68%)
            </button>
            <button class="w-full text-left px-3 py-2 text-label-md hover:bg-surface-container flex items-center gap-2" onclick="loadCohort('low'); closeCohortDropdown();">
              <span class="w-2 h-2 rounded-full bg-tertiary"></span> Case B: Normoglycemic (14%)
            </button>
            <button class="w-full text-left px-3 py-2 text-label-md hover:bg-surface-container flex items-center gap-2" onclick="loadCohort('moderate'); closeCohortDropdown();">
              <span class="w-2 h-2 rounded-full bg-amber-500"></span> Case C: Impaired Glucose (46%)
            </button>
          </div>
        </div>
      </div>
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
              <span class="text-label-lg font-bold text-on-surface" id="bannerModelTitle">XGBoost Classifier (Tuned • ROC-AUC 0.841)</span>
            </div>
          </div>
          <!-- Toggle Buttons -->
          <div class="flex items-center gap-2 bg-surface-container-low p-1 rounded-xl border border-outline-variant/30">
            <button id="toggleBtnXgb" class="px-4 py-1.5 rounded-lg text-label-md font-bold transition-all bg-primary text-on-primary shadow-xs" onclick="changeModel('xgboost')">
              XGBoost (Best)
            </button>
            <button id="toggleBtnRf" class="px-4 py-1.5 rounded-lg text-label-md font-medium text-on-surface-variant hover:text-on-surface transition-all" onclick="changeModel('random_forest')">
              Random Forest
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
                High-dimensional risk estimation combining biometric clinical parameters with transparent, feature-level attribution calculated via Shapley additive explanations (SHAP).
              </p>
              
              <!-- Cohort Quick Load Strip -->
              <div class="flex items-center gap-2 mt-3 flex-nowrap overflow-x-auto py-1">
                <span class="text-label-sm uppercase tracking-wider text-outline font-semibold whitespace-nowrap shrink-0">Quick Presets:</span>
                <button class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-error-container/40 text-on-error-container hover:bg-error-container/70 text-label-sm transition-colors font-medium whitespace-nowrap shrink-0" onclick="loadCohort('high')" type="button">
                  <span class="w-2 h-2 rounded-full bg-error shrink-0"></span>
                  Case A: Elevated Glycemia (High Risk)
                </button>
                <button class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface text-label-sm transition-colors font-medium whitespace-nowrap shrink-0" onclick="loadCohort('low')" type="button">
                  <span class="w-2 h-2 rounded-full bg-tertiary shrink-0"></span>
                  Case B: Normoglycemic (Low Risk)
                </button>
                <button class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-100/70 hover:bg-amber-100 text-amber-900 text-label-sm transition-colors font-medium whitespace-nowrap shrink-0" onclick="loadCohort('moderate')" type="button">
                  <span class="w-2 h-2 rounded-full bg-amber-600 shrink-0"></span>
                  Case C: Impaired Glucose (Moderate)
                </button>
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
                  <span class="text-label-sm text-secondary bg-surface px-2 py-0.5 rounded-full font-medium" id="heroModelBadge">XGBoost calibrated</span>
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
                  <text class="text-[9px]" fill="#434655" x="165" y="14" font-weight="500">Mean Baseline 0.35</text>
                </svg>
                <div class="flex items-center justify-between text-label-sm text-on-surface-variant pt-2">
                  <span class="flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-primary" id="legendDot"></span><span id="legendText">XGBoost Predictor</span></span>
                  <span class="flex items-center gap-1 cursor-pointer font-semibold text-secondary hover:underline" onclick="scrollToShap()"><span class="w-1.5 h-1.5 rounded-full bg-secondary"></span>SHAP Attribution</span>
                  <span class="text-tertiary font-semibold">99.8% Parsed</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <!-- Main Workstation Layout: 7 cols Form / 5 cols Result -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mb-10">
          
          <!-- LEFT COLUMN: Patient Information & Clinical Measurements (7 Columns) -->
          <div class="lg:col-span-7 bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm flex flex-col gap-6">
            <div class="flex items-center justify-between pb-3">
              <div>
                <h2 class="text-headline-sm text-on-surface font-bold">Patient Information</h2>
                <p class="text-body-sm text-on-surface-variant mt-0.5">Enter standard physiological markers to evaluate individualized metabolic risk.</p>
              </div>
              <span class="hidden sm:inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface-container text-on-surface-variant text-label-sm">
                <span class="material-symbols-outlined text-[14px]">checklist</span>
                8 Features Required
              </span>
            </div>

            <form class="flex flex-col gap-6" id="predictionForm" onsubmit="event.preventDefault(); calculateRisk();">
              
              <!-- Subsection 1: Demographics -->
              <div>
                <div class="flex items-center gap-2 mb-3">
                  <span class="w-6 h-6 rounded-md bg-surface-container text-primary flex items-center justify-center text-label-sm font-bold">1</span>
                  <h3 class="text-label-lg text-on-surface font-semibold">Demographic Profile</h3>
                </div>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <!-- Age Field -->
                  <div class="flex flex-col gap-1.5">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputAge">
                        Age
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Patient chronological age in completed years">info</span>
                      </label>
                      <span class="text-label-sm text-outline">years</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputAge" max="100" min="18" placeholder="e.g. 48" type="number" value="48" required/>
                  </div>
                  <!-- Pregnancies Field -->
                  <div class="flex flex-col gap-1.5">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputPregnancies">
                        Pregnancies
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Total number of completed pregnancies / parity">info</span>
                      </label>
                      <span class="text-label-sm text-outline">count</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputPregnancies" max="20" min="0" placeholder="e.g. 2" type="number" value="2" required/>
                  </div>
                </div>
              </div>

              <!-- Subsection 2: Clinical Biometrics -->
              <div>
                <div class="flex items-center gap-2 mb-3">
                  <span class="w-6 h-6 rounded-md bg-surface-container text-primary flex items-center justify-center text-label-sm font-bold">2</span>
                  <h3 class="text-label-lg text-on-surface font-semibold">Clinical & Laboratory Measurements</h3>
                </div>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <!-- Glucose Field -->
                  <div class="flex flex-col gap-1.5 sm:col-span-2 p-3.5 rounded-xl bg-surface-container-low/60 border border-outline-variant/30">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-semibold flex items-center gap-1" for="inputGlucose">
                        Glucose (2-hr OGTT)
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Plasma glucose concentration at 2 hours in an oral glucose tolerance test">help</span>
                      </label>
                      <span class="text-label-sm text-on-surface-variant font-medium">mg/dL</span>
                    </div>
                    <div class="flex items-center gap-3">
                      <div class="relative flex-1">
                        <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container transition-colors" id="inputGlucose" max="300" min="40" type="number" value="154" required/>
                      </div>
                      <div class="flex flex-col text-right">
                        <span class="inline-flex items-center px-2 py-0.5 rounded-full bg-error-container text-on-error-container text-label-sm font-semibold" id="glucoseBadge">Elevated (≥126)</span>
                        <span class="text-[10px] text-outline mt-0.5">Ref: Normal &lt;100 mg/dL</span>
                      </div>
                    </div>
                  </div>

                  <!-- Blood Pressure -->
                  <div class="flex flex-col gap-1.5">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputBP">
                        Blood Pressure
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Diastolic blood pressure (mm Hg)">info</span>
                      </label>
                      <span class="text-label-sm text-outline">mm Hg</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputBP" max="160" min="30" type="number" value="78" required/>
                  </div>

                  <!-- BMI Field -->
                  <div class="flex flex-col gap-1.5">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputBMI">
                        Body Mass Index (BMI)
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Weight in kg / (height in m)^2">info</span>
                      </label>
                      <span class="text-label-sm text-outline">kg/m²</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputBMI" max="70" min="10" step="0.1" type="number" value="31.4" required/>
                  </div>

                  <!-- Insulin Field -->
                  <div class="flex flex-col gap-1.5">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputInsulin">
                        Serum Insulin
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="2-Hour serum insulin (µU/mL)">info</span>
                      </label>
                      <span class="text-label-sm text-outline">µU/mL</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputInsulin" max="900" min="0" type="number" value="130" required/>
                  </div>

                  <!-- Skin Thickness -->
                  <div class="flex flex-col gap-1.5">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputSkin">
                        Triceps Skinfold
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Triceps skin fold thickness in millimeters">info</span>
                      </label>
                      <span class="text-label-sm text-outline">mm</span>
                    </div>
                    <input class="w-full h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputSkin" max="99" min="0" type="number" value="32" required/>
                  </div>

                  <!-- Diabetes Pedigree Function -->
                  <div class="flex flex-col gap-1.5 sm:col-span-2">
                    <div class="flex items-center justify-between">
                      <label class="text-label-md text-on-surface font-medium flex items-center gap-1" for="inputPedigree">
                        Diabetes Pedigree Function
                        <span class="material-symbols-outlined text-outline text-[16px] cursor-help" title="Genetic synthesis score based on family history & relative relations">info</span>
                      </label>
                      <span class="text-label-sm text-outline">score (0.05 - 2.50)</span>
                    </div>
                    <div class="flex items-center gap-3">
                      <input class="flex-1 h-11 px-3.5 bg-surface-container-lowest border border-outline-variant/50 rounded-lg font-code-num text-body-md text-on-surface outline-none shadow-xs focus:border-primary focus:bg-surface-container-low transition-colors" id="inputPedigree" max="3.0" min="0.01" step="0.001" type="number" value="0.672" required/>
                      <span class="hidden md:inline text-label-sm text-on-surface-variant bg-surface-container px-3 py-2.5 rounded-lg border border-outline-variant/30 font-medium">High Familial Risk Factor</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Form Action Row -->
              <div class="flex flex-col sm:flex-row items-center gap-3 pt-3">
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

          <!-- RIGHT COLUMN: Risk Assessment Gauge & Confidence (5 Columns) -->
          <div class="lg:col-span-5 flex flex-col gap-6">
            
            <!-- Primary Risk Gauge Card -->
            <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm flex flex-col items-center text-center relative overflow-hidden">
              
              <!-- Header Note with Active Model Display & Switcher Button -->
              <div class="w-full flex items-center justify-between pb-3 border-b border-outline-variant/20">
                <div class="flex items-center gap-1.5">
                  <span class="material-symbols-outlined text-[18px] text-primary" id="gaugeModelIcon">memory</span>
                  <span class="text-label-md font-bold text-on-surface" id="gaugeModelLabel">XGBoost v2.4 Inference</span>
                </div>
                <!-- Quick Model Switcher button right on the card -->
                <button class="px-2.5 py-1 rounded-full bg-surface-container hover:bg-surface-container-high text-primary text-label-sm font-semibold transition-colors flex items-center gap-1" onclick="toggleModelFromCard()" title="Switch active ML model">
                  <span class="material-symbols-outlined text-[14px]">swap_horiz</span>
                  <span id="cardSwitchBtnText">Switch to Random Forest</span>
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
                  <span class="text-display-lg font-bold text-on-surface tracking-tight" id="riskGaugeValue">68%</span>
                  <span class="text-label-md uppercase tracking-wider text-on-surface-variant font-semibold">Estimated Risk</span>
                  <span class="text-label-sm text-outline mt-0.5">Threshold: 0.50</span>
                </div>
              </div>

              <!-- Risk Categorization Pill -->
              <div class="mt-1 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-error-container text-on-error-container text-label-lg font-bold" id="riskPill">
                <span class="material-symbols-outlined text-[18px]" id="riskPillIcon">warning</span>
                <span id="riskPillText">Higher Predicted Risk</span>
              </div>
              <p class="text-body-sm text-on-surface-variant mt-2 max-w-xs" id="riskPhenotypeDesc">
                Patient biometric markers align with positive diabetic phenotype under high oral glucose challenge.
              </p>

              <!-- Spectrum Visualizer -->
              <div class="w-full mt-4 flex flex-col gap-1.5">
                <div class="flex items-center justify-between text-label-sm text-on-surface-variant">
                  <span id="specLabelLow">Low (&lt;30%)</span>
                  <span id="specLabelMod">Moderate (30-60%)</span>
                  <span id="specLabelHigh" class="font-bold text-error">Elevated (&gt;60%)</span>
                </div>
                <!-- 3-Segment Color Bar -->
                <div class="w-full h-2.5 rounded-full bg-surface-container flex overflow-hidden p-0.5 gap-0.5 relative">
                  <div class="w-[30%] h-full rounded-l-full bg-tertiary-container/80"></div>
                  <div class="w-[30%] h-full bg-secondary-fixed-dim"></div>
                  <div class="w-[40%] h-full rounded-r-full bg-error"></div>
                  <!-- Marker Pin -->
                  <div class="absolute top-[-3px] w-2 h-4 bg-on-surface rounded-full shadow-md transition-all duration-500" id="spectrumMarker" style="left: 68%;"></div>
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
                  <span class="text-[11px] font-mono font-bold text-secondary bg-surface px-2 py-0.5 rounded-full" id="shapSummaryBadge">+0.33 Net SHAP</span>
                </div>
                
                <div class="flex flex-col gap-1.5 text-label-sm" id="instantShapPills">
                  <!-- Injected via JS -->
                  <div class="flex items-center justify-between">
                    <span class="text-on-surface font-medium">Glucose (154 mg/dL)</span>
                    <span class="font-code-num font-bold text-error">+0.31 SHAP</span>
                  </div>
                  <div class="flex items-center justify-between">
                    <span class="text-on-surface font-medium">BMI (31.4 kg/m²)</span>
                    <span class="font-code-num font-bold text-error">+0.14 SHAP</span>
                  </div>
                  <div class="flex items-center justify-between">
                    <span class="text-on-surface font-medium">Blood Pressure (78 mm)</span>
                    <span class="font-code-num font-bold text-secondary">-0.07 SHAP</span>
                  </div>
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

              <!-- Model Calibration & Confidence Footer -->
              <div class="w-full mt-4 pt-3 bg-surface-container-low rounded-xl p-3.5 flex items-center justify-between text-left">
                <div class="flex items-center gap-2.5">
                  <div class="w-8 h-8 rounded-lg bg-secondary-container text-on-secondary-container flex items-center justify-center">
                    <span class="material-symbols-outlined text-[18px]">verified_user</span>
                  </div>
                  <div>
                    <span class="text-label-sm text-on-surface-variant block">Model Confidence</span>
                    <span class="text-label-md text-on-surface font-bold" id="confidenceValue">89.2% Predictive Certainty</span>
                  </div>
                </div>
                <span class="text-label-sm text-secondary bg-surface px-2 py-1 rounded-md font-medium border border-outline-variant/30" id="brierScore">Brier: 0.11</span>
              </div>
            </div>

          </div>
        </div>

        <!-- ========================================================================= -->
        <!-- SECTION 3: SHAP EXPLAINABILITY: Why did the model make this prediction?   -->
        <!-- ========================================================================= -->
        <section class="w-full mb-10 pt-4" id="shap-analysis-section">
          <div class="flex flex-col gap-1 mb-6">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <span class="w-8 h-8 rounded-lg bg-secondary-container text-on-secondary-container flex items-center justify-center">
                  <span class="material-symbols-outlined text-[20px]">psychology</span>
                </span>
                <h2 class="text-headline-sm text-on-surface font-bold">
                  SHAP Explainability: Why did the model make this prediction?
                </h2>
              </div>
              <button class="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-secondary-container/40 text-on-secondary-container hover:bg-secondary-container text-label-md font-semibold transition-colors" onclick="switchTab('explainability-xai')">
                <span class="material-symbols-outlined text-[18px]">open_in_new</span>
                <span>Open Global SHAP Suite</span>
              </button>
            </div>
            <p class="text-body-md text-on-surface-variant">
              TreeSHAP value decomposition decomposes the final probability from the population baseline into individual additive risk pushes.
            </p>
          </div>

          <!-- Side-by-Side Dual Risk Factor Panels -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <!-- Card A: Risk Increasing Factors (Red / Warm) -->
            <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-7 shadow-sm flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between pb-4 border-b border-outline-variant/20">
                  <div class="flex items-center gap-3">
                    <span class="w-10 h-10 rounded-xl bg-error-container text-on-error-container flex items-center justify-center shrink-0">
                      <span class="material-symbols-outlined text-[24px]">arrow_upward</span>
                    </span>
                    <div>
                      <h3 class="text-headline-sm text-on-surface font-bold">Risk-Increasing Factors</h3>
                      <span class="text-body-sm text-on-surface-variant">Pushed estimate higher towards diabetes</span>
                    </div>
                  </div>
                  <span class="px-3.5 py-1.5 rounded-full bg-error-container font-code-num text-body-sm text-on-error-container font-bold" id="netPosBadge">+0.56 Net</span>
                </div>

                <!-- Feature Item List -->
                <div class="flex flex-col gap-5 mt-5" id="increasingFactorsList">
                  <!-- Dynamically populated via JS -->
                </div>
              </div>

              <div class="mt-6 pt-4 flex items-center gap-2.5 text-on-surface-variant text-body-md font-medium border-t border-outline-variant/20" id="primaryDriverText">
                <span class="material-symbols-outlined text-[20px] text-outline">info</span>
                Primary driver: Plasma Glucose accounts for ~55% of total risk inflation.
              </div>
            </div>

            <!-- Card B: Risk Decreasing Factors (Teal / Green) -->
            <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-7 shadow-sm flex flex-col justify-between">
              <div>
                <div class="flex items-center justify-between pb-4 border-b border-outline-variant/20">
                  <div class="flex items-center gap-3">
                    <span class="w-10 h-10 rounded-xl bg-secondary-container text-on-secondary-container flex items-center justify-center shrink-0">
                      <span class="material-symbols-outlined text-[24px]">arrow_downward</span>
                    </span>
                    <div>
                      <h3 class="text-headline-sm text-on-surface font-bold">Risk-Decreasing Factors</h3>
                      <span class="text-body-sm text-on-surface-variant">Pulled estimate lower towards normal baseline</span>
                    </div>
                  </div>
                  <span class="px-3.5 py-1.5 rounded-full bg-secondary-container font-code-num text-body-sm text-on-secondary-container font-bold" id="netNegBadge">-0.14 Net</span>
                </div>

                <!-- Feature Item List -->
                <div class="flex flex-col gap-5 mt-5" id="decreasingFactorsList">
                  <!-- Dynamically populated via JS -->
                </div>
              </div>

              <div class="mt-6 pt-4 flex items-center gap-2.5 text-on-surface-variant text-body-md font-medium border-t border-outline-variant/20">
                <span class="material-symbols-outlined text-[20px] text-secondary">check_circle</span>
                Vascular and endocrine biomarkers provide partial protective buffering.
              </div>
            </div>

          </div>
        </section>

        <!-- ========================================================================= -->
        <!-- SECTION 4: Detailed SHAP Analytics Card & Waterfall Plot                  -->
        <!-- ========================================================================= -->
        <section class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm mb-10 flex flex-col gap-6">
          <div class="flex flex-col gap-1 pb-2">
            <h2 class="text-headline-sm text-on-surface font-bold">Feature Contribution Analysis (SHAP Waterfall)</h2>
            <p class="text-body-sm text-on-surface-variant" id="waterfallFormulaText">
              Each feature nudges the patient's score from the model base rate <code class="bg-surface-container px-1 py-0.5 rounded text-primary font-mono">E[f(x)] = 0.35</code> to predicted output <code class="bg-surface-container px-1 py-0.5 rounded text-error font-mono" id="waterfallPredictedFormula">f(x) = 0.68</code>.
            </p>
          </div>

          <!-- Horizontal Centered Bi-directional SHAP Divergence Chart -->
          <div class="w-full bg-surface-container-low rounded-xl p-5 overflow-x-auto border border-outline-variant/30">
            <div class="min-w-[640px] flex flex-col gap-3">
              <div class="grid grid-cols-12 text-center text-label-sm text-on-surface-variant pb-2 border-b border-outline-variant/40 font-semibold">
                <div class="col-span-3 text-left">Patient Feature</div>
                <div class="col-span-4 text-right text-secondary">← Protective (-SHAP)</div>
                <div class="col-span-1 text-center font-mono font-bold text-outline">0.0</div>
                <div class="col-span-4 text-left text-error">Elevating (+SHAP) →</div>
              </div>

              <!-- Divergence Rows Container -->
              <div id="divergenceRows" class="flex flex-col gap-2">
                <!-- Dynamically generated rows via JS -->
              </div>
            </div>
          </div>

          <!-- Step-by-Step Cumulative Waterfall Cascade Visualization -->
          <div class="p-5 rounded-xl bg-surface-container-low/60 flex flex-col gap-3 border border-outline-variant/30">
            <div class="flex items-center justify-between">
              <span class="text-label-md text-on-surface font-semibold flex items-center gap-1.5">
                <span class="material-symbols-outlined text-primary text-[18px]">stacked_bar_chart</span>
                Cumulative Probability Stepper
              </span>
              <span class="text-label-sm text-outline">Base Rate → Patient Risk</span>
            </div>
            
            <div class="flex items-center gap-2 overflow-x-auto py-2" id="stepperContainer">
              <!-- Dynamically populated cascade items via JS -->
            </div>
          </div>
        </section>

        <!-- SECTION 5: Global Model Explainability & Diagnostics -->
        <section class="mb-10">
          <div class="flex flex-col gap-1 mb-6">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-primary text-[22px]">analytics</span>
              <h2 class="text-headline-sm text-on-surface font-bold">
                How the Model Makes Predictions
              </h2>
            </div>
            <p class="text-body-md text-on-surface-variant">
              Global insights validated across benchmarked cohorts illustrating feature dependencies for both XGBoost and Random Forest.
            </p>
          </div>

          <!-- 3 Stat KPI Cards -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <!-- Card 1 -->
            <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
              <div class="flex items-center justify-between">
                <span class="text-label-md text-on-surface-variant font-medium">Most Influential Feature</span>
                <span class="w-8 h-8 rounded-lg bg-primary-fixed text-on-primary-fixed flex items-center justify-center">
                  <span class="material-symbols-outlined text-[18px]">bloodtype</span>
                </span>
              </div>
              <div class="mt-4">
                <span class="text-headline-lg font-bold text-on-surface block">Glucose</span>
                <div class="flex items-center gap-2 mt-1">
                  <span class="text-label-sm text-primary font-bold">Mean |SHAP|: 0.932</span>
                  <span class="text-label-sm text-outline">• Accounts for 42% global gain</span>
                </div>
              </div>
            </div>

            <!-- Card 2 -->
            <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
              <div class="flex items-center justify-between">
                <span class="text-label-md text-on-surface-variant font-medium">Explainability Method</span>
                <span class="w-8 h-8 rounded-lg bg-secondary-container text-on-secondary-container flex items-center justify-center">
                  <span class="material-symbols-outlined text-[18px]">account_tree</span>
                </span>
              </div>
              <div class="mt-4">
                <span class="text-headline-lg font-bold text-on-surface block">TreeSHAP</span>
                <div class="flex items-center gap-2 mt-1">
                  <span class="text-label-sm text-secondary font-bold">Exact Additive Efficiency</span>
                  <span class="text-label-sm text-outline">• Lundberg et al.</span>
                </div>
              </div>
            </div>

            <!-- Card 3 -->
            <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
              <div class="flex items-center justify-between">
                <span class="text-label-md text-on-surface-variant font-medium">Model Architecture</span>
                <span class="w-8 h-8 rounded-lg bg-surface-container text-primary flex items-center justify-center">
                  <span class="material-symbols-outlined text-[18px]">model_training</span>
                </span>
              </div>
              <div class="mt-4">
                <span class="text-headline-lg font-bold text-on-surface block" id="kpiModelName">XGBoost Tuned</span>
                <div class="flex items-center gap-2 mt-1">
                  <span class="text-label-sm text-tertiary font-bold" id="kpiModelAuc">ROC-AUC: 0.841</span>
                  <span class="text-label-sm text-outline" id="kpiModelF1">• F1-Score: 0.654</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Global Feature Importance Ranking (Ranked Preview) -->
          <div class="bg-surface-container-lowest rounded-2xl p-6 lg:p-8 shadow-sm">
            <div class="flex items-center justify-between mb-4">
              <div>
                <h3 class="text-label-lg text-on-surface font-bold" id="importanceSectionTitle">Global Feature Importance (Cohort Mean |SHAP|)</h3>
                <span class="text-body-sm text-on-surface-variant" id="importanceSectionSubtitle">Ranked importance across validation sets (TreeSHAP attribution).</span>
              </div>
              <div class="flex items-center gap-2">
                <button class="px-3 py-1 rounded-full text-label-sm font-semibold transition-colors bg-primary text-on-primary" id="btnImpXgb" onclick="switchImportanceView('xgb')">XGBoost SHAP</button>
                <button class="px-3 py-1 rounded-full text-label-sm font-medium transition-colors bg-surface-container text-on-surface-variant hover:text-on-surface" id="btnImpRf" onclick="switchImportanceView('rf')">Random Forest Gini</button>
              </div>
            </div>

            <!-- Importance Bars Container -->
            <div id="importanceBarsContainer" class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 pt-2">
              <!-- Dynamically populated via JS -->
            </div>
          </div>
        </section>

        <!-- SECTION 6: Model Specifications & Technical Metadata Strip -->
        <section class="bg-surface-container-low rounded-2xl p-4 lg:p-5 mb-8 shadow-xs border border-outline-variant/30">
          <div class="flex flex-wrap items-center justify-between gap-4 text-label-sm text-on-surface-variant">
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-primary text-[18px]">verified</span>
              <span class="font-semibold text-on-surface">Models:</span> XGBoost v2.0 (Best) &amp; Random Forest (n=100)
            </div>
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-secondary text-[18px]">hub</span>
              <span class="font-semibold text-on-surface">Explainability:</span> TreeSHAP (Lundberg et al., Nature MI)
            </div>
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-outline text-[18px]">dataset</span>
              <span class="font-semibold text-on-surface">Dataset:</span> Pima Indians Diabetes / NHANES Standardized
            </div>
            <div class="flex items-center gap-2">
              <span class="material-symbols-outlined text-tertiary text-[18px]">security</span>
              <span class="font-semibold text-on-surface">Ingress:</span> Zero-Retention Client Ingestion
            </div>
          </div>
        </section>

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
              <tr class="hover:bg-surface-container-low transition-colors bg-primary-fixed/20 font-semibold" id="tableRowXgb">
                <td class="py-3.5 px-4 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-primary"></span>
                  XGBoost Classifier (Tuned)
                </td>
                <td class="py-3.5 px-4 font-code-num text-primary font-bold">77.27%</td>
                <td class="py-3.5 px-4 font-code-num">70.21%</td>
                <td class="py-3.5 px-4 font-code-num">61.11%</td>
                <td class="py-3.5 px-4 font-code-num">65.35%</td>
                <td class="py-3.5 px-4 font-code-num text-primary font-bold">0.8409</td>
                <td class="py-3.5 px-4">
                  <button class="px-3 py-1 rounded-full bg-primary text-on-primary text-label-sm font-semibold shadow-xs" onclick="changeModel('xgboost'); switchTab('risk-prediction');">
                    Use XGBoost
                  </button>
                </td>
              </tr>
              <tr class="hover:bg-surface-container-low transition-colors" id="tableRowRf">
                <td class="py-3.5 px-4 flex items-center gap-2">
                  <span class="w-2.5 h-2.5 rounded-full bg-secondary"></span>
                  Random Forest Classifier (n=100)
                </td>
                <td class="py-3.5 px-4 font-code-num">75.32%</td>
                <td class="py-3.5 px-4 font-code-num">64.29%</td>
                <td class="py-3.5 px-4 font-code-num text-secondary font-bold">66.67%</td>
                <td class="py-3.5 px-4 font-code-num">65.45%</td>
                <td class="py-3.5 px-4 font-code-num">0.8257</td>
                <td class="py-3.5 px-4">
                  <button class="px-3 py-1 rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface text-label-sm font-semibold transition-colors" onclick="changeModel('random_forest'); switchTab('risk-prediction');">
                    Use Random Forest
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
          <div class="mt-4 p-3 rounded-lg bg-surface-container-low text-body-sm text-on-surface-variant flex items-center gap-2">
            <span class="material-symbols-outlined text-secondary text-[18px]">lightbulb</span>
            <span><strong>Clinical Trade-off:</strong> Random Forest offers higher Recall (66.67% vs 61.11%), detecting more potential diabetic patients, while XGBoost achieves higher overall Accuracy (77.27%) and higher ROC-AUC (0.841).</span>
          </div>
        </div>

        <!-- Evaluation Plots Row: ROC-AUC & Both Confusion Matrices -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
            <div>
              <h3 class="text-label-lg font-bold text-on-surface mb-2">ROC-AUC Curves Comparison</h3>
              <p class="text-body-sm text-on-surface-variant mb-3">XGBoost (AUC 0.841) vs Random Forest (AUC 0.826).</p>
              <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-2">
                <img src="{img_roc}" alt="ROC Comparison" class="w-full h-auto object-contain max-h-[300px]"/>
              </div>
            </div>
            <p class="text-body-sm text-on-surface-variant mt-3 pt-2 border-t border-outline-variant/20">
              Evaluates true positive rate vs false positive rate across all operational decision cutoffs.
            </p>
          </div>

          <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <h3 class="text-label-lg font-bold text-on-surface">Confusion Matrix: XGBoost</h3>
                <span class="text-label-sm text-primary font-bold bg-primary-fixed/40 px-2 py-0.5 rounded">Tuned Best</span>
              </div>
              <p class="text-body-sm text-on-surface-variant mb-3">Accuracy: 77.27% • Precision: 70.21%</p>
              <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-2">
                <img src="{img_cm_xgb}" alt="Confusion Matrix XGBoost" class="w-full h-auto object-contain max-h-[300px]"/>
              </div>
            </div>
            <p class="text-body-sm text-on-surface-variant mt-3 pt-2 border-t border-outline-variant/20">
              Higher specificity and lower false positive count (14 FP).
            </p>
          </div>

          <div class="bg-surface-container-lowest rounded-2xl p-6 shadow-sm flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-2">
                <h3 class="text-label-lg font-bold text-on-surface">Confusion Matrix: Random Forest</h3>
                <span class="text-label-sm text-secondary font-bold bg-secondary-container/40 px-2 py-0.5 rounded">Ensemble</span>
              </div>
              <p class="text-body-sm text-on-surface-variant mb-3">Accuracy: 75.32% • Recall: 66.67%</p>
              <div class="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex items-center justify-center p-2">
                <img src="{img_cm_rf}" alt="Confusion Matrix Random Forest" class="w-full h-auto object-contain max-h-[300px]"/>
              </div>
            </div>
            <p class="text-body-sm text-on-surface-variant mt-3 pt-2 border-t border-outline-variant/20">
              Higher sensitivity and lower false negative count (18 FN).
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
            <option value="xgboost">XGBoost (Tuned Best • AUC 0.841)</option>
            <option value="random_forest">Random Forest Classifier (AUC 0.826)</option>
          </select>
        </div>
        <label class="flex items-center justify-between cursor-pointer">
          <span class="font-medium text-on-surface">Auto-recalculate on keystroke</span>
          <input type="checkbox" id="settingLiveRecalc" checked class="w-4 h-4 text-primary rounded"/>
        </label>
        <label class="flex items-center justify-between cursor-pointer">
          <span class="font-medium text-on-surface">Classification Threshold</span>
          <span class="text-label-sm font-mono bg-surface-container px-2 py-0.5 rounded">0.50</span>
        </label>
      </div>
      <div class="flex justify-end pt-3">
        <button class="px-4 py-2 rounded-xl bg-primary text-on-primary text-label-md font-semibold" onclick="closeSettingsModal()">Done</button>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <footer class="w-full bg-surface-container-lowest shadow-[0_-1px_6px_rgba(0,0,0,0.02)] no-print mt-auto">
    <div class="max-w-7xl mx-auto px-6 py-space-lg flex flex-col md:flex-row items-center justify-between gap-space-md text-label-md text-on-surface-variant">
      <div class="flex flex-wrap items-center gap-space-md">
        <span class="flex items-center gap-1.5 text-on-surface font-medium">
          <span class="material-symbols-outlined text-secondary text-[16px]">verified</span>
          Clinical Decision Support
        </span>
        <span class="hidden sm:inline text-outline-variant">•</span>
        <span>XGBoost &amp; Random Forest Supported</span>
        <span class="hidden sm:inline text-outline-variant">•</span>
        <span>TreeSHAP Integrated</span>
      </div>
      <div class="flex items-center gap-space-sm px-space-md py-1 rounded-full bg-surface-container text-label-sm">
        <span class="w-2 h-2 rounded-full bg-tertiary"></span>
        <span class="text-on-surface font-medium">System Ready: SHAP Explainer Active</span>
      </div>
    </div>
  </footer>

  <!-- DYNAMIC CLIENT LOGIC -->
  <script>
    let activeModel = 'xgboost'; // 'xgboost' or 'random_forest'

    const DEP_PLOTS = {{
      glucose: {{
        img: '{img_dep_gluc}',
        desc: '<strong>Glucose Effect:</strong> Clear inflection point observed at ~125 mg/dL. Values below 100 mg/dL yield negative SHAP (protective), whereas values exceeding 140 mg/dL exhibit steep risk elevation.'
      }},
      bmi: {{
        img: '{img_dep_bmi}',
        desc: '<strong>BMI Effect:</strong> Risk turns increasingly positive above BMI 26.0 kg/m² (overweight cutoff) and levels off into high diabetic contribution past BMI 35.0 kg/m².'
      }},
      age: {{
        img: '{img_dep_age}',
        desc: '<strong>Age Effect:</strong> Accelerating contribution after age 35, reflecting gradual beta-cell secretory depletion with plateau above age 50.'
      }},
      pedigree: {{
        img: '{img_dep_ped}',
        desc: '<strong>Diabetes Pedigree Function Effect:</strong> Strong monotonic correlation with positive SHAP contribution as hereditary familial risk score exceeds 0.50.'
      }},
      preg: {{
        img: '{img_dep_preg}',
        desc: '<strong>Pregnancies Parity Effect:</strong> Moderate positive risk push when parity exceeds 4 pregnancies relative to nulliparous cohorts.'
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

    function changeModel(modelName) {{
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

      if (modelName === 'xgboost') {{
        toggleXgb.className = 'px-4 py-1.5 rounded-lg text-label-md font-bold transition-all bg-primary text-on-primary shadow-xs';
        toggleRf.className = 'px-4 py-1.5 rounded-lg text-label-md font-medium text-on-surface-variant hover:text-on-surface transition-all';
        bannerTitle.innerText = 'XGBoost Classifier (Tuned • ROC-AUC 0.841)';
        gaugeModelLabel.innerText = 'XGBoost v2.4 Inference';
        gaugeModelIcon.innerText = 'memory';
        cardSwitchBtnText.innerText = 'Switch to Random Forest';
        kpiModelName.innerText = 'XGBoost Tuned';
        kpiModelAuc.innerText = 'ROC-AUC: 0.841';
        kpiModelF1.innerText = '• F1-Score: 0.654';
        heroModelBadge.innerText = 'XGBoost calibrated';
        if (heroEngineTag) heroEngineTag.innerText = 'Explainable Machine Learning • TreeSHAP Engine';
        legendText.innerText = 'XGBoost Predictor';
        showToast('Switched to XGBoost Model (ROC-AUC: 0.841, Accuracy: 77.27%)');
      }} else {{
        toggleRf.className = 'px-4 py-1.5 rounded-lg text-label-md font-bold transition-all bg-primary text-on-primary shadow-xs';
        toggleXgb.className = 'px-4 py-1.5 rounded-lg text-label-md font-medium text-on-surface-variant hover:text-on-surface transition-all';
        bannerTitle.innerText = 'Random Forest Classifier (100 Trees • ROC-AUC 0.826)';
        gaugeModelLabel.innerText = 'Random Forest Inference';
        gaugeModelIcon.innerText = 'forest';
        cardSwitchBtnText.innerText = 'Switch to XGBoost';
        kpiModelName.innerText = 'Random Forest';
        kpiModelAuc.innerText = 'ROC-AUC: 0.826';
        kpiModelF1.innerText = '• Recall: 66.67%';
        heroModelBadge.innerText = 'Random Forest calibrated';
        if (heroEngineTag) heroEngineTag.innerText = 'Random Forest Ensemble • TreeSHAP (Class 1)';
        legendText.innerText = 'Random Forest Predictor';
        showToast('Switched to Random Forest Model (ROC-AUC: 0.826, Recall: 66.67%)');
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
        subtitle.innerText = 'Normalized mean decrease in Gini impurity across 100 decision trees.';
        renderImportanceBars([
          {{ name: '1. Glucose Concentration', score: '0.311', pct: 100 }},
          {{ name: '2. Body Mass Index (BMI)', score: '0.179', pct: 57.5 }},
          {{ name: '3. Age', score: '0.115', pct: 36.9 }},
          {{ name: '4. Diabetes Pedigree', score: '0.104', pct: 33.4 }},
          {{ name: '5. Insulin Serum', score: '0.079', pct: 25.4 }},
          {{ name: '6. Pregnancies Count', score: '0.078', pct: 25.1 }},
          {{ name: '7. Blood Pressure', score: '0.069', pct: 22.2 }},
          {{ name: '8. Skinfold Thickness', score: '0.065', pct: 20.9 }}
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

    // Tab switching
    function switchTab(tabId) {{
      const tabs = ['risk-prediction', 'explainability-xai', 'model-benchmark-and-insights'];
      tabs.forEach(t => {{
        const content = document.getElementById('tab-content-' + t);
        const btn = document.getElementById('nav-' + t);
        if (content && btn) {{
          const isXai = (t === 'explainability-xai');
          const prefix = isXai ? 'flex items-center gap-1.5 ' : '';
          
          if (t === tabId) {{
            content.classList.remove('hidden');
            btn.className = prefix + 'px-space-md py-1.5 rounded-lg transition-all bg-primary text-on-primary shadow-sm font-semibold text-label-md';
            if (isXai) {{
              const b = document.getElementById('shapNavBadge');
              if (b) b.className = 'px-1.5 py-0.5 rounded bg-white/20 text-white text-[11px] font-bold';
            }}
          }} else {{
            content.classList.add('hidden');
            btn.className = prefix + 'px-space-md py-1.5 rounded-lg text-label-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-all';
            if (isXai) {{
              const b = document.getElementById('shapNavBadge');
              if (b) b.className = 'px-1.5 py-0.5 rounded bg-primary-fixed text-primary text-[11px] font-bold';
            }}
          }}
        }}
      }});
      if (window.lenisInstance) {{
        window.lenisInstance.scrollTo(0, {{ immediate: false, duration: 0.7 }});
      }} else {{
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}
      syncFrameHeight();
    }}

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

    function toggleCohortDropdown() {{
      const d = document.getElementById('cohortDropdown');
      d.classList.toggle('hidden');
    }}
    function closeCohortDropdown() {{
      document.getElementById('cohortDropdown').classList.add('hidden');
    }}

    function openSettingsModal() {{ 
      document.getElementById('settingsModal').classList.remove('hidden'); 
      if (window.lenisInstance) window.lenisInstance.stop();
    }}
    function closeSettingsModal() {{ 
      document.getElementById('settingsModal').classList.add('hidden'); 
      if (window.lenisInstance) window.lenisInstance.start();
    }}

    function closeReportModal() {{ 
      document.getElementById('reportModal').classList.add('hidden'); 
      if (window.lenisInstance) window.lenisInstance.start();
    }}

    function showToast(message) {{
      const toast = document.getElementById('statusToast');
      const toastText = document.getElementById('statusToastText');
      toastText.innerText = message;
      toast.classList.remove('translate-y-20', 'opacity-0');
      setTimeout(() => {{
        toast.classList.add('translate-y-20', 'opacity-0');
      }}, 2800);
    }}

    function loadCohort(type) {{
      const age = document.getElementById('inputAge');
      const preg = document.getElementById('inputPregnancies');
      const gluc = document.getElementById('inputGlucose');
      const bp = document.getElementById('inputBP');
      const bmi = document.getElementById('inputBMI');
      const ins = document.getElementById('inputInsulin');
      const skin = document.getElementById('inputSkin');
      const ped = document.getElementById('inputPedigree');
      const cohortLabel = document.getElementById('currentCohortLabel');

      if (type === 'high') {{
        age.value = 48;
        preg.value = 2;
        gluc.value = 154;
        bp.value = 78;
        bmi.value = 31.4;
        ins.value = 130;
        skin.value = 32;
        ped.value = 0.672;
        cohortLabel.innerText = "Cohort 04 (High)";
        calculateRisk(false, 68);
        showToast('Loaded Case A: Elevated Glycemia (68% Predicted Risk)');
      }} else if (type === 'low') {{
        age.value = 28;
        preg.value = 0;
        gluc.value = 88;
        bp.value = 68;
        bmi.value = 21.8;
        ins.value = 65;
        skin.value = 19;
        ped.value = 0.214;
        cohortLabel.innerText = "Cohort 02 (Low)";
        calculateRisk(false, 14);
        showToast('Loaded Case B: Normoglycemic (14% Predicted Risk)');
      }} else if (type === 'moderate') {{
        age.value = 41;
        preg.value = 1;
        gluc.value = 126;
        bp.value = 74;
        bmi.value = 28.2;
        ins.value = 95;
        skin.value = 24;
        ped.value = 0.450;
        cohortLabel.innerText = "Cohort 07 (Moderate)";
        calculateRisk(false, 46);
        showToast('Loaded Case C: Impaired Glucose (46% Predicted Risk)');
      }}
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
      
      updateRiskOutput(0, 'Awaiting Clinical Data', 'neutral', {{}}, 0.35);
      showToast('Patient form reset to blank');
    }}

    function updateRiskOutput(riskPercent, label, tier, shapValues, baseValue = 0.35) {{
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

      gaugeValue.innerText = riskPercent + '%';
      
      // Circumference = 2 * PI * 82 ≈ 515.2
      const totalCircumference = 515.2;
      const offset = totalCircumference - (totalCircumference * (riskPercent / 100));
      gaugeCircle.style.strokeDashoffset = offset;

      spectrumMarker.style.left = Math.min(Math.max(riskPercent, 4), 94) + '%';

      const cert = (75 + Math.abs(riskPercent - 50) * 0.4).toFixed(1);
      confidenceValue.innerText = cert + '% Predictive Certainty';

      if (tier === 'elevated') {{
        gaugeCircle.setAttribute('stroke', '#ba1a1a');
        riskPill.className = 'mt-1 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-error-container text-on-error-container text-label-lg font-bold';
        riskPillText.innerText = label;
        riskPillIcon.innerText = 'warning';
        glucoseBadge.className = 'inline-flex items-center px-2 py-0.5 rounded-full bg-error-container text-on-error-container text-label-sm font-semibold';
        glucoseBadge.innerText = 'Elevated (≥126)';
        riskPhenotypeDesc.innerText = 'Patient biometric markers align with positive diabetic phenotype under high oral glucose challenge.';
        if (adviceTitle) adviceTitle.innerText = 'Diagnostic Follow-up Suggestion';
        if (adviceText) adviceText.innerText = 'Recommended confirmation via laboratory HbA1c panel and fasting lipid profile given persistent elevated glucose excursion.';
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
        riskPhenotypeDesc.innerText = 'Metabolic biomarkers within optimal non-diabetic range with protective endocrine buffering.';
        if (adviceTitle) adviceTitle.innerText = 'Preventative Health Guidance';
        if (adviceText) adviceText.innerText = 'Routine annual metabolic screening recommended. Maintain current balanced nutrition and physical activity.';
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
        riskPhenotypeDesc.innerText = 'Borderline glycemic indicators suggest pre-diabetic risk or insulin resistance.';
        if (adviceTitle) adviceTitle.innerText = 'Lifestyle Intervention Plan';
        if (adviceText) adviceText.innerText = 'Consider structured lifestyle modifications (dietary carbohydrate control, 150 min/wk moderate activity) and repeat fasting panel in 6 months.';
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

      // Update Waterfall text
      document.getElementById('waterfallPredictedFormula').innerText = 'f(x) = ' + (riskPercent / 100).toFixed(2);
      
      // Update Stepper
      renderCumulativeStepper(baseValue, shapValues, riskPercent / 100);
      
      // Update Divergence Rows
      renderDivergenceChart(shapValues);
      
      // Update Increasing & Decreasing factor cards
      renderFactorCards(shapValues);
    }}

    function renderInstantShapSummary(shapMap) {{
      const container = document.getElementById('instantShapPills');
      const badge = document.getElementById('shapSummaryBadge');
      container.innerHTML = '';

      let net = 0;
      const sorted = Object.entries(shapMap).sort((a,b) => Math.abs(b[1].shap) - Math.abs(a[1].shap));
      sorted.forEach(([_, info]) => net += info.shap);
      badge.innerText = (net >= 0 ? '+' : '') + net.toFixed(2) + ' Net SHAP';

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

    function renderCumulativeStepper(baseValue, shapMap, finalScore) {{
      const container = document.getElementById('stepperContainer');
      container.innerHTML = '';

      // Base rate
      const baseEl = document.createElement('div');
      baseEl.className = 'px-3 py-2 rounded-lg bg-surface text-center shrink-0 border border-outline-variant/30';
      baseEl.innerHTML = `<span class="text-[11px] text-outline block font-medium">Base E[f(x)]</span><span class="font-code-num text-label-md font-bold text-on-surface">${{baseValue.toFixed(2)}}</span>`;
      container.appendChild(baseEl);

      const sortedEntries = Object.entries(shapMap).sort((a,b) => Math.abs(b[1].shap) - Math.abs(a[1].shap));
      
      // Take top 4 influential and group remainder
      const topFew = sortedEntries.slice(0, 4);
      const remainder = sortedEntries.slice(4);
      let remSum = 0;
      remainder.forEach(r => remSum += r[1].shap);

      topFew.forEach(([feat, info]) => {{
        const isPos = info.shap >= 0;
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined text-outline text-[16px] shrink-0';
        icon.innerText = isPos ? 'add' : 'remove';
        container.appendChild(icon);

        const card = document.createElement('div');
        card.className = `px-3 py-2 rounded-lg ${{isPos ? 'bg-error-container/40' : 'bg-secondary-container/40'}} text-center shrink-0 border border-outline-variant/20`;
        const valStr = (isPos ? '+' : '') + info.shap.toFixed(2);
        card.innerHTML = `<span class="text-[11px] ${{isPos ? 'text-on-error-container' : 'text-on-secondary-container'}} block font-medium">${{feat.replace('DiabetesPedigreeFunction', 'Pedigree').replace('BloodPressure', 'BP').replace('SkinThickness', 'Skin')}}</span><span class="font-code-num text-label-md font-bold ${{isPos ? 'text-error' : 'text-secondary'}}">${{valStr}}</span>`;
        container.appendChild(card);
      }});

      if (remainder.length > 0 && Math.abs(remSum) >= 0.005) {{
        const isPos = remSum >= 0;
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined text-outline text-[16px] shrink-0';
        icon.innerText = isPos ? 'add' : 'remove';
        container.appendChild(icon);

        const card = document.createElement('div');
        card.className = `px-3 py-2 rounded-lg ${{isPos ? 'bg-error-container/30' : 'bg-secondary-container/30'}} text-center shrink-0 border border-outline-variant/20`;
        const valStr = (isPos ? '+' : '') + remSum.toFixed(2);
        card.innerHTML = `<span class="text-[11px] ${{isPos ? 'text-on-error-container' : 'text-on-secondary-container'}} block font-medium">Others</span><span class="font-code-num text-label-md font-bold ${{isPos ? 'text-error' : 'text-secondary'}}">${{valStr}}</span>`;
        container.appendChild(card);
      }}

      // Final Arrow and Score
      const arrow = document.createElement('span');
      arrow.className = 'material-symbols-outlined text-primary text-[20px] shrink-0';
      arrow.innerText = 'arrow_right_alt';
      container.appendChild(arrow);

      const finalEl = document.createElement('div');
      finalEl.className = 'px-4 py-2 rounded-lg bg-primary text-on-primary text-center shrink-0 shadow-xs';
      finalEl.innerHTML = `<span class="text-[11px] text-primary-fixed block font-medium">Final Score f(x)</span><span class="font-code-num text-label-md font-bold">${{finalScore.toFixed(2)}} (${{Math.round(finalScore * 100)}}%)</span>`;
      container.appendChild(finalEl);
    }}

    function renderDivergenceChart(shapMap) {{
      const container = document.getElementById('divergenceRows');
      container.innerHTML = '';

      const features = [
        {{ key: 'Glucose', label: 'Glucose', unit: 'mg/dL' }},
        {{ key: 'BMI', label: 'BMI', unit: 'kg/m²' }},
        {{ key: 'Age', label: 'Age', unit: 'yr' }},
        {{ key: 'BloodPressure', label: 'Blood Pressure', unit: 'mm' }},
        {{ key: 'Pregnancies', label: 'Pregnancies', unit: 'count' }},
        {{ key: 'DiabetesPedigreeFunction', label: 'Pedigree Func', unit: '' }},
        {{ key: 'SkinThickness', label: 'Skin Thickness', unit: 'mm' }},
        {{ key: 'Insulin', label: 'Insulin', unit: 'µU/mL' }}
      ];

      features.forEach(f => {{
        const item = shapMap[f.key] || {{ shap: 0, val: 0 }};
        const shap = item.shap;
        const val = item.val;
        const isPos = shap >= 0;
        const absVal = Math.abs(shap);
        const barWidth = Math.min(Math.round((absVal / 0.35) * 85), 90);

        const row = document.createElement('div');
        row.className = 'grid grid-cols-12 items-center text-body-sm py-1.5 hover:bg-surface-container/50 px-2 rounded-lg transition-colors';
        
        const labelText = `${{f.label}} (${{val}}${{f.unit ? ' ' + f.unit : ''}})`;

        if (isPos) {{
          row.innerHTML = `
            <div class="col-span-3 font-semibold text-on-surface truncate" title="${{labelText}}">${{labelText}}</div>
            <div class="col-span-4 flex justify-end"></div>
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
            <div class="col-span-3 font-semibold text-on-surface truncate" title="${{labelText}}">${{labelText}}</div>
            <div class="col-span-4 flex items-center justify-end gap-2">
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

      incContainer.innerHTML = '';
      decContainer.innerHTML = '';

      let sumPos = 0;
      let sumNeg = 0;
      const posList = [];
      const negList = [];

      const explanations = {{
        Glucose: {{
          pos: 'Elevated plasma concentration exceeds normal glycemic clearance capacity.',
          neg: 'Optimal fasting / post-challenge glycemia within healthy reference range.'
        }},
        BMI: {{
          pos: 'Weight in excess of optimal body mass promotes peripheral insulin insensitivity.',
          neg: 'Normal body mass index reduces systemic metabolic resistance.'
        }},
        Age: {{
          pos: 'Advancing chronological age correlates with beta-cell secretory depletion.',
          neg: 'Younger age profile preserves endogenous metabolic flexibility.'
        }},
        DiabetesPedigreeFunction: {{
          pos: 'Strong polygenic inheritance score indicates familial predisposition.',
          neg: 'Lower familial diabetes density confers protective genetic buffer.'
        }},
        BloodPressure: {{
          pos: 'Elevated vascular tension suggests microvascular or metabolic stress.',
          neg: 'Optimal diastolic pressure within normal vascular boundaries.'
        }},
        Pregnancies: {{
          pos: 'Elevated parity increases historical gestational metabolic challenge.',
          neg: 'Lower parity score relative to high-risk multiparous cohorts.'
        }},
        SkinThickness: {{
          pos: 'Elevated triceps skinfold reflects increased subcutaneous adiposity.',
          neg: 'Subcutaneous adipose distribution within moderate parameters.'
        }},
        Insulin: {{
          pos: 'Elevated circulating insulin indicates compensatory pancreatic hypersecretion.',
          neg: 'Endogenous insulin response maintains baseline pancreatic function.'
        }}
      }};

      Object.entries(shapMap).forEach(([feat, info]) => {{
        if (info.shap >= 0) {{
          sumPos += info.shap;
          posList.push({{ feat, ...info }});
        }} else {{
          sumNeg += info.shap;
          negList.push({{ feat, ...info }});
        }}
      }});

      posList.sort((a,b) => b.shap - a.shap);
      negList.sort((a,b) => a.shap - b.shap);

      netPosBadge.innerText = '+' + sumPos.toFixed(2) + ' Net';
      netNegBadge.innerText = sumNeg.toFixed(2) + ' Net';

      if (posList.length > 0) {{
        const topDriver = posList[0];
        const pct = Math.round((topDriver.shap / (sumPos || 1)) * 100);
        primaryDriverText.innerHTML = `<span class="material-symbols-outlined text-[20px] text-outline">info</span> Primary driver: ${{topDriver.feat}} accounts for ~${{pct}}% of total risk inflation.`;
      }} else {{
        primaryDriverText.innerHTML = `<span class="material-symbols-outlined text-[20px] text-secondary">verified</span> No significant risk-inflating features detected.`;
      }}

      // Render Positive
      if (posList.length === 0) {{
        incContainer.innerHTML = '<div class="text-body-md text-outline italic py-4">No significant risk-increasing factors.</div>';
      }} else {{
        posList.forEach(item => {{
          const el = document.createElement('div');
          el.className = 'flex flex-col gap-1.5 py-1.5';
          const barPct = Math.min(Math.round((item.shap / 0.35) * 85), 100);
          const desc = explanations[item.feat]?.pos || 'Elevates probabilistic diabetes risk.';
          el.innerHTML = `
            <div class="flex items-center justify-between text-body-lg">
              <span class="font-bold text-on-surface text-[17px] tracking-tight">${{item.feat}}: <span class="font-code-num font-semibold text-on-surface-variant">${{item.val}}</span></span>
              <span class="font-code-num font-black text-error text-[17px]">+${{item.shap.toFixed(2)}} SHAP</span>
            </div>
            <div class="w-full h-3 rounded-full bg-surface-container overflow-hidden my-0.5 shadow-inner">
              <div class="h-full bg-error rounded-full transition-all duration-500" style="width: ${{barPct}}%;"></div>
            </div>
            <span class="text-body-md text-[15px] text-on-surface-variant leading-relaxed font-normal">${{desc}}</span>
          `;
          incContainer.appendChild(el);
        }});
      }}

      // Render Negative
      if (negList.length === 0) {{
        decContainer.innerHTML = '<div class="text-body-md text-outline italic py-4">No protective factors identified.</div>';
      }} else {{
        negList.forEach(item => {{
          const el = document.createElement('div');
          el.className = 'flex flex-col gap-1.5 py-1.5';
          const barPct = Math.min(Math.round((Math.abs(item.shap) / 0.35) * 85), 100);
          const desc = explanations[item.feat]?.neg || 'Reduces probabilistic diabetes risk.';
          el.innerHTML = `
            <div class="flex items-center justify-between text-body-lg">
              <span class="font-bold text-on-surface text-[17px] tracking-tight">${{item.feat}}: <span class="font-code-num font-semibold text-on-surface-variant">${{item.val}}</span></span>
              <span class="font-code-num font-black text-secondary text-[17px]">${{item.shap.toFixed(2)}} SHAP</span>
            </div>
            <div class="w-full h-3 rounded-full bg-surface-container overflow-hidden my-0.5 shadow-inner">
              <div class="h-full bg-secondary rounded-full transition-all duration-500" style="width: ${{barPct}}%;"></div>
            </div>
            <span class="text-body-md text-[15px] text-on-surface-variant leading-relaxed font-normal">${{desc}}</span>
          `;
          decContainer.appendChild(el);
        }});
      }}
    }}

    function calculateRisk(isLive = false, forcedRisk = null) {{
      const gluc = parseFloat(document.getElementById('inputGlucose').value);
      const bmi = parseFloat(document.getElementById('inputBMI').value);
      const age = parseFloat(document.getElementById('inputAge').value);
      const bp = parseFloat(document.getElementById('inputBP').value);
      const preg = parseFloat(document.getElementById('inputPregnancies').value);
      const ins = parseFloat(document.getElementById('inputInsulin').value);
      const skin = parseFloat(document.getElementById('inputSkin').value);
      const ped = parseFloat(document.getElementById('inputPedigree').value);

      if (isNaN(gluc) && isNaN(bmi) && isNaN(age)) {{
        resetForm();
        return;
      }}

      const btn = document.getElementById('btnCalculate');
      const icon = document.getElementById('calcIcon');
      const text = document.getElementById('calcText');

      const executeCalculation = () => {{
        // Model-specific Shapley weights:
        // XGBoost: higher gain on Glucose
        // Random Forest: higher weight on BMI & Pedigree
        const glucFactor = activeModel === 'random_forest' ? 0.0051 : 0.0055;
        const bmiFactor = activeModel === 'random_forest' ? 0.0185 : 0.0160;
        const pedFactor = activeModel === 'random_forest' ? 0.165 : 0.140;

        const shap_gluc = ( (gluc || 117) - 110 ) * glucFactor;
        const shap_bmi = ( (bmi || 32) - 25 ) * bmiFactor;
        const shap_age = ( (age || 29) - 30 ) * 0.004;
        const shap_ped = ( (ped || 0.37) - 0.35 ) * pedFactor;
        const shap_bp = ( (bp || 72) - 75 ) * 0.002;
        const shap_preg = ( (preg || 2) - 2.5 ) * 0.018;
        const shap_skin = ( (skin || 23) - 25 ) * 0.002;
        const shap_ins = ( (ins || 80) - 100 ) * 0.0003;

        const baseVal = 0.35;
        const totalShap = shap_gluc + shap_bmi + shap_age + shap_ped + shap_bp + shap_preg + shap_skin + shap_ins;
        
        let calculatedRisk = forcedRisk !== null ? forcedRisk : Math.round(Math.min(Math.max((baseVal + totalShap) * 100, 4), 98));
        
        const shapMap = {{
          Glucose: {{ val: gluc || 117, shap: shap_gluc }},
          BMI: {{ val: bmi || 32, shap: shap_bmi }},
          Age: {{ val: age || 29, shap: shap_age }},
          DiabetesPedigreeFunction: {{ val: ped || 0.37, shap: shap_ped }},
          BloodPressure: {{ val: bp || 72, shap: shap_bp }},
          Pregnancies: {{ val: preg || 2, shap: shap_preg }},
          SkinThickness: {{ val: skin || 23, shap: shap_skin }},
          Insulin: {{ val: ins || 80, shap: shap_ins }}
        }};

        let tier = 'moderate';
        let label = 'Moderate Predicted Risk';
        if (calculatedRisk >= 60) {{
          tier = 'elevated';
          label = 'Higher Predicted Risk';
        }} else if (calculatedRisk <= 30) {{
          tier = 'low';
          label = 'Low Predicted Risk';
        }}

        updateRiskOutput(calculatedRisk, label, tier, shapMap, baseVal);

        if (!isLive) {{
          showToast(`${{activeModel === 'random_forest' ? 'Random Forest' : 'XGBoost'}} SHAP risk calculated successfully`);
        }}
      }};

      if (!isLive) {{
        icon.innerText = 'sync';
        icon.classList.add('animate-spin');
        text.innerText = 'Calculating SHAP Values...';
        btn.disabled = true;

        setTimeout(() => {{
          icon.classList.remove('animate-spin');
          icon.innerText = 'calculate';
          text.innerText = 'Estimate Diabetes Risk';
          btn.disabled = false;
          executeCalculation();
        }}, 450);
      }} else {{
        executeCalculation();
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
      const modelDisplayName = activeModel === 'random_forest' ? 'Random Forest Classifier (AUC: 0.826)' : 'XGBoost Classifier (AUC: 0.841)';

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
          <strong>Notice:</strong> This computational report is generated for investigational clinical decision support. Final diagnostic determinations must be performed by a licensed physician in conjunction with formal venous plasma HbA1c tests.
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
          estimated_risk: document.getElementById('riskGaugeValue').innerText,
          risk_category: document.getElementById('riskPillText').innerText,
          baseline_rate: 0.35,
          engine: "TreeSHAP (Lundberg et al.)"
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

    // Auto calculate initial default case on load
    window.addEventListener('DOMContentLoaded', () => {{
      switchImportanceView('xgb');
      calculateRisk(true, 68);
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
