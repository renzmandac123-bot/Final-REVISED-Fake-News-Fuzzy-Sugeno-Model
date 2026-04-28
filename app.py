import streamlit as st
import re
import time
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, unquote
from PIL import Image
import io
import requests

st.set_page_config(
    page_title="Filipino Fake News Detector",
    page_icon="🔬",
    layout="wide",
)

# load model code once
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import importlib.util, sys, types

    # load spaCy
    import spacy
    try:
        nlp_en = spacy.load("en_core_web_sm")
    except Exception:
        nlp_en = spacy.blank("en")
    try:
        nlp_xx = spacy.load("xx_ent_wiki_sm")
    except Exception:
        nlp_xx = spacy.blank("xx")

    # load langdetect
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0

    # try cv2
    try:
        import cv2
        CV2_OK = True
    except Exception:
        cv2 = None
        CV2_OK = False

    # try BeautifulSoup
    from bs4 import BeautifulSoup

    # build a module namespace that contains all the dependencies
    # the model code expects as globals
    g = {
        "nlp_en": nlp_en,
        "nlp_xx": nlp_xx,
        "detect": detect,
        "cv2": cv2,
        "CV2_OK": CV2_OK,
        "BeautifulSoup": BeautifulSoup,
        "np": np,
        "re": re,
        "Path": Path,
        "requests": requests,
        "REQUESTS_OK": True,
        "IMAGE_CACHE_DIR": Path("/tmp/image_cache"),
    }
    Path("/tmp/image_cache").mkdir(parents=True, exist_ok=True)

    # execute model.py inside this namespace
    with open("model.py", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, "model.py", "exec"), g)

    return g

m = load_model()

def run_pipeline(text, img_array=None, img_desc="", use_images=False):
    result = m["evaluate_article"](
        text,
        image_path=img_array,
        use_images=use_images,
    )
    result["img_desc"] = img_desc
    return result

def label_color(label):
    colors = {
        "Real":          "#2ecc71",
        "Mostly Real":   "#82e060",
        "Half Real":     "#f0c040",
        "Mostly Fake":   "#f08020",
        "Fake":          "#e04040",
        "Pants on Fire": "#c0185a",
    }
    return colors.get(label, "#888888")

def score_bar(value, label=""):
    pct = max(0.0, min(1.0, float(value))) * 100
    r = int(40 + 200 * (pct / 100))
    g2 = int(200 - 160 * (pct / 100))
    bar_html = f"""
    <div style='display:flex;align-items:center;gap:10px;margin:4px 0;'>
      <div style='width:220px;background:#1a1a2e;border-radius:4px;
                  height:14px;overflow:hidden;flex-shrink:0;'>
        <div style='width:{pct:.1f}%;background:rgb({r},{g2},60);
                    height:100%;border-radius:4px;'></div>
      </div>
      <code>{value:.4f}</code>
      <small style='color:#888;'>{label}</small>
    </div>"""
    return bar_html

def mf_bars(fuzz_dict):
    colors = {"low": "#4caf93", "medium": "#f0b429", "high": "#e05252"}
    html = ""
    for level in ("low", "medium", "high"):
        v = fuzz_dict.get(level, 0.0)
        pct = v * 100
        c = colors[level]
        html += f"""<div style='display:flex;align-items:center;gap:6px;margin:2px 0;'>
          <span style='width:56px;font-size:11px;color:#aaa;'>{level}</span>
          <div style='width:160px;background:#1a1a2e;border-radius:3px;
                      height:10px;overflow:hidden;'>
            <div style='width:{pct:.1f}%;background:{c};height:100%;border-radius:3px;'></div>
          </div>
          <code style='font-size:11px;color:{c};'>{v:.4f}</code>
        </div>"""
    return html

def threshold_ruler(F):
    labels = ["PoF","Fake","Mostly Fake","Half Real","Mostly Real","Real"]
    seg_colors = ["#c0185a","#e04040","#f08020","#f0c040","#82e060","#2ecc71"]
    segs = ""
    w = 100/6
    for i, (lbl, col) in enumerate(zip(labels, seg_colors)):
        segs += f"<div title='{lbl}' style='width:{w:.2f}%;background:{col};height:100%;display:inline-block;'></div>"
    marker = F * 100
    return f"""
    <div style='position:relative;height:20px;border-radius:4px;
                overflow:hidden;background:#111;margin:8px 0;'>
      {segs}
      <div style='position:absolute;top:0;left:{marker:.2f}%;
                  transform:translateX(-50%);height:100%;width:3px;
                  background:#fff;box-shadow:0 0 6px #fff;'></div>
    </div>
    <div style='display:flex;justify-content:space-between;
                font-size:10px;color:#666;'>
      {''.join(f"<span>{l}</span>" for l in labels)}
    </div>"""

def show_results(result, img_array=None, img_desc=""):
    raw        = result["raw"]
    fuzzified  = result["fuzzified"]
    F          = result["CS"]
    label      = result["multi_label"]
    binary     = result["binary_label"]
    use_images = img_array is not None

    st.markdown("---")

    # Feature scores
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Feature Scores")
        html = ""
        descs = {
            "ET": "0 = calm, 1 = emotional",
            "GQ": "0 = clean, 1 = noisy",
            "CS": "0 = credible, 1 = unsupported",
            "IEP": "0 = neutral image, 1 = provocative",
            "PTC": "0 = consistent, 1 = inconsistent",
        }
        for k, v in raw.items():
            html += f"<b>{k}</b>: {score_bar(v, descs.get(k,''))}"
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        st.subheader("Fuzzification")
        html = ""
        for k, fd in fuzzified.items():
            html += f"<div style='margin:8px 0;'><b style='color:#7ab4ff;'>{k} = {raw[k]:.4f}</b>"
            html += mf_bars(fd)
            html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # Rule firing
    st.subheader("Sugeno Rule Evaluation")
    active = m["RULES"] if use_images else [
        (a, z) for a, z in m["RULES"]
        if all(f not in ("IEP","PTC") for f, _ in a)
    ]
    fired = []
    total_num, total_den = 0.0, 0.0
    for ants, z in active:
        w = 1.0
        for feat, lab in ants:
            w = min(w, fuzzified.get(feat, {}).get(lab, 0.0))
        if w > 0.001:
            fired.append((ants, z, w, w*z))
        total_num += w * z
        total_den += w
    fired.sort(key=lambda x: -x[3])

    if fired:
        table_html = """
        <table style='width:100%;border-collapse:collapse;
                      font-family:monospace;font-size:12px;color:#cdd;
                      background:#0d0d1f;'>
        <thead><tr style='background:#0d0d1f;color:#7ab4ff;
                          border-bottom:2px solid #2a2a5a;'>
          <th style='padding:5px 8px;text-align:left;'>Antecedent</th>
          <th style='padding:5px 8px;'>z</th>
          <th style='padding:5px 8px;'>weight w</th>
          <th style='padding:5px 8px;'>w × z</th>
        </tr></thead><tbody>"""
        for i, (ants, z, w, wz) in enumerate(fired[:15]):
            bg = "#0d0d1f" if i % 2 == 0 else "#111128"
            ant_str = " AND ".join(f"<b>{f}</b> is <b>{l}</b>" for f,l in ants)
            table_html += f"""<tr style='background:{bg};'>
              <td style='padding:4px 8px;color:#aacfff;'>IF {ant_str}</td>
              <td style='padding:4px 8px;text-align:center;color:#ffd980;'>{z:.2f}</td>
              <td style='padding:4px 8px;text-align:center;'>{w:.4f}</td>
              <td style='padding:4px 8px;text-align:right;color:#90ee90;'>{wz:.5f}</td>
            </tr>"""
        table_html += "</tbody></table>"
        st.markdown(
            f"<div style='background:#0d0d1f;border:1px solid #2a2a4a;"
            f"border-radius:6px;overflow:hidden;'>{table_html}</div>"
            f"<small style='color:#666;'>Showing top {min(15,len(fired))} of "
            f"{len(fired)} fired rules</small>",
            unsafe_allow_html=True
        )

    # Final output
    st.markdown("---")
    st.subheader("Final Classification")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Sugeno Score", f"{F:.4f}")
        st.markdown(threshold_ruler(F), unsafe_allow_html=True)
    with col_b:
        col = label_color(label)
        st.markdown(
            f"<div style='background:{col};color:#fff;padding:12px 20px;"
            f"border-radius:20px;text-align:center;font-size:18px;"
            f"font-weight:bold;'>{label}</div>",
            unsafe_allow_html=True
        )
        st.caption("6-class label")
    with col_c:
        bin_col = "#2ecc71" if binary == "True" else "#e04040"
        st.markdown(
            f"<div style='background:{bin_col};color:#fff;padding:12px 20px;"
            f"border-radius:20px;text-align:center;font-size:18px;"
            f"font-weight:bold;'>{binary}</div>",
            unsafe_allow_html=True
        )
        st.caption("Binary: True = credible")

    # Explanation
    st.markdown("---")
    st.subheader("Why this classification?")
    try:
        expl_html = m["generate_explanation"](raw, fuzzified, F, label, use_images)
        # Strip HTML for display in st.info
        plain = re.sub(r'<[^>]+>', ' ', expl_html)
        plain = plain.replace('&nbsp;', ' ').replace('&ldquo;', '"').replace('&rdquo;', '"')
        plain = plain.replace('&ndash;', '-').replace('WARNING:', 'Note:')
        plain = re.sub(r'\s{2,}', ' ', plain).strip()
        st.info(plain)
    except Exception as ex:
        st.warning(f"Explanation not available: {ex}")

# ── URL extraction helper ─────────────────────────────────────────────────────
def fetch_from_url(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    text, title, img_url, img_arr, img_desc = "", "", "", None, ""
    log = []

    try:
        from newspaper import Article as _NpArticle
        art = _NpArticle(url, headers=headers, fetch_images=False)
        art.download()
        art.parse()
        text      = art.text or ""
        title     = art.title or ""
        img_url   = art.top_image or ""
        log.append(f"newspaper4k: {len(text)} chars")
    except Exception as e:
        log.append(f"newspaper4k failed: {e}")

    if len(text) < 100:
        try:
            from bs4 import BeautifulSoup
            resp = requests.get(url, headers=headers, timeout=12)
            soup = BeautifulSoup(resp.text, "html.parser")
            og_t = soup.find("meta", property="og:title")
            title = og_t["content"].strip() if og_t and og_t.get("content") else ""
            og_i = soup.find("meta", property="og:image")
            img_url = og_i["content"].strip() if og_i and og_i.get("content") else ""
            body = (soup.find("article") or soup.find("main") or soup.body)
            if body:
                for tag in body.find_all(["nav","aside","footer","script","style","noscript"]):
                    tag.decompose()
                text = re.sub(r'\n{3,}', '\n\n', body.get_text(separator="\n")).strip()
                log.append(f"BS4 fallback: {len(text)} chars")
        except Exception as e:
            log.append(f"BS4 fallback failed: {e}")

    # strip bylines
    text = re.sub(
        r'^(?:by\s+[\w\s,.\-]+(?:\||–|-).*\n|(?:published|updated|posted)\s.*\n)',
        '', text, flags=re.IGNORECASE|re.MULTILINE, count=3
    ).strip()

    # get alt-text/caption
    if img_url:
        try:
            resp_html = requests.get(url, headers=headers, timeout=10).text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp_html, "html.parser")
            og_alt = soup.find("meta", property="og:image:alt")
            if og_alt and len((og_alt.get("content") or "").strip()) > 5:
                img_desc = og_alt["content"].strip()
            else:
                img_fname = unquote(Path(urlparse(img_url).path).name).lower()
                for tag in soup.find_all("img"):
                    for attr in ("src","data-src","data-lazy-src"):
                        if img_fname and img_fname in (tag.get(attr,"") or "").lower():
                            alt = (tag.get("alt","") or "").strip()
                            if len(alt) > 3:
                                img_desc = alt
                            fig = tag.find_parent("figure")
                            if fig:
                                cap = fig.find("figcaption")
                                if cap:
                                    img_desc += " " + cap.get_text().strip()
                            break
        except Exception:
            pass

    # download image
    if img_url:
        try:
            import cv2
            resp_img = requests.get(img_url, headers=headers, timeout=10)
            arr = np.frombuffer(resp_img.content, dtype=np.uint8)
            img_arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            img_arr = None

    return text, title, img_url, img_arr, img_desc, log

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("🔬 Filipino Fake News Detector")
st.caption("Type 1 Takagi-Sugeno Zero-Order Fuzzy Inference System")

tab1, tab2 = st.tabs(["📄 Manual Input", "🌐 URL Input"])

# ── TAB 1: Manual Input ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Manual Text and Image Input")
    st.caption("Paste the article text below. Optionally upload the headline image.")

    text_input = st.text_area(
        "Article text",
        height=200,
        placeholder="Paste the article text here..."
    )
    img_file = st.file_uploader(
        "Headline image (optional)",
        type=["jpg","jpeg","png","gif","webp","bmp"]
    )

    if st.button("Analyse", key="btn_manual", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some article text before analysing.")
        else:
            with st.spinner("Running fuzzy analysis..."):
                img_array = None
                img_desc  = ""
                use_images = False

                if img_file is not None:
                    try:
                        import cv2
                        pil_img = Image.open(img_file).convert("RGB")
                        img_array = np.array(pil_img)[:, :, ::-1]
                        img_desc  = img_file.name
                        use_images = True
                    except Exception as e:
                        st.warning(f"Image could not be loaded: {e}")

                result = run_pipeline(text_input, img_array, img_desc, use_images)

            if img_array is not None:
                st.image(
                    Image.fromarray(img_array[:, :, ::-1]),
                    caption="Uploaded image",
                    width=300
                )

            show_results(result, img_array, img_desc)

# ── TAB 2: URL Input ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("URL-Based Article Input")
    st.caption("Enter a news article URL. The system will extract the text and headline image automatically.")

    url_input = st.text_input(
        "Article URL",
        placeholder="https://www.rappler.com/..."
    )

    if st.button("Fetch and Analyse", key="btn_url", type="primary"):
        if not url_input.strip() or not url_input.startswith("http"):
            st.warning("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Fetching article..."):
                text, title, img_url, img_arr, img_desc, log = fetch_from_url(url_input)

            if len(text) < 50:
                st.error(
                    "Could not extract enough text from that URL. "
                    "The page may be paywalled, JavaScript-only, or blocking automated access."
                )
                with st.expander("Extraction log"):
                    for entry in log:
                        st.text(entry)
            else:
                st.success(f"Extracted {len(text):,} characters from: {title or url_input}")

                with st.expander("Extracted article text (first 800 chars)"):
                    st.text(text[:800] + ("..." if len(text) > 800 else ""))

                use_images = img_arr is not None

                if img_arr is not None:
                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        st.image(
                            Image.fromarray(img_arr[:, :, ::-1]),
                            caption="Headline image",
                            width=280
                        )
                    with col_info:
                        ptc_src = img_desc if img_desc else img_url
                        st.caption(
                            f"Image source for PTC: "
                            f"{'alt-text/caption' if img_desc else 'filename fallback'}"
                        )
                        if img_desc:
                            st.caption(f"Description: {img_desc[:120]}")
                else:
                    st.info("No headline image could be loaded. Running in text-only mode.")
                    use_images = False

                ptc_input = img_desc if img_desc else img_url

                with st.spinner("Running fuzzy analysis..."):
                    result = run_pipeline(text, img_arr, ptc_input, use_images)

                show_results(result, img_arr, ptc_input)

                with st.expander("Extraction log"):
                    for entry in log:
                        st.text(entry)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Fuzzy Fake News Detection System for Filipino Online News | "
    "Takagi-Sugeno Type 1 FIS | "
    "Features: ET, GQ, CS, IEP, PTC"
)
