import base64
import hmac
import io
import os
import re
import ssl
import time
from pathlib import Path

import fitz
import httpx
import streamlit as st
import streamlit.components.v1 as components
import truststore
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
OCR_MODEL = "openweight-ocr"
DEFAULT_MODEL = os.getenv("ALBERT_TRANSLATION_MODEL", "openweight-small")
MAX_FILE_MB = 30
MAX_AUDIO_MB = 20
MAX_PAGES = 40
CHUNK_SIZE = 4500
MIN_REQUEST_INTERVAL = 12.0

LANGUAGES = {
    "Allemand": {"code": "de", "prompt": "allemand naturel"},
    "Anglais": {"code": "en", "prompt": "anglais naturel"},
    "Arabe algérien": {"code": "ar-DZ", "prompt": "arabe algérien parlé (darja algérienne), en alphabet arabe, sans revenir à l'arabe standard sauf nécessité"},
    "Arabe standard moderne": {"code": "ar", "prompt": "arabe standard moderne clair"},
    "Arabe tunisien": {"code": "ar-TN", "prompt": "arabe tunisien parlé (derja tunisienne), en alphabet arabe, sans revenir à l'arabe standard sauf nécessité"},
    "Espagnol": {"code": "es", "prompt": "espagnol naturel"},
    "Français": {"code": "fr", "prompt": "français clair et naturel"},
    "Italien": {"code": "it", "prompt": "italien naturel"},
    "Néerlandais": {"code": "nl", "prompt": "néerlandais naturel"},
    "Polonais": {"code": "pl", "prompt": "polonais naturel"},
    "Portugais": {"code": "pt", "prompt": "portugais européen naturel"},
    "Roumain": {"code": "ro", "prompt": "roumain naturel"},
    "Russe": {"code": "ru", "prompt": "russe naturel"},
    "Turc": {"code": "tr", "prompt": "turc standard naturel"},
    "Ukrainien": {"code": "uk", "prompt": "ukrainien naturel"},
    "Wolof": {"code": "wo", "prompt": "wolof naturel utilisé au Sénégal, sans mélange inutile avec le français"},
}
STYLES = {
    "Naturel et fidèle": "Utilise un style naturel, fidèle et fluide.",
    "Administratif clair": "Utilise un style administratif clair, précis et accessible au public.",
    "Langage simplifié": "Utilise des phrases simples et un vocabulaire accessible, sans supprimer ni résumer aucune information.",
}

st.set_page_config(page_title="Traducteur PDF Albert", page_icon="🌍", layout="wide")


def make_client(api_key: str) -> OpenAI:
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    http_client = httpx.Client(verify=context, timeout=httpx.Timeout(connect=30, read=600, write=300, pool=30), trust_env=True)
    return OpenAI(base_url=BASE_URL, api_key=api_key, http_client=http_client, max_retries=0)


@st.cache_data(ttl=300, show_spinner=False)
def list_translation_models(api_key: str) -> list[str]:
    client = make_client(api_key)
    try:
        ignored = ("ocr", "whisper", "embedding", "rerank", "speech")
        result = []
        for item in client.models.list().data:
            model_id = str(getattr(item, "id", "") or "")
            model_type = str(getattr(item, "type", "") or "").lower()
            if model_id and not any(word in f"{model_id} {model_type}".lower() for word in ignored):
                result.append(model_id)
        return sorted(set(result))
    finally:
        client.close()


@st.cache_data(ttl=300, show_spinner=False)
def list_asr_models(api_key: str) -> list[str]:
    client = make_client(api_key)
    try:
        return sorted({
            str(item.id) for item in client.models.list().data
            if getattr(item, "type", None) == "automatic-speech-recognition"
        })
    finally:
        client.close()


def wait_before_request(state: dict) -> None:
    elapsed = time.monotonic() - state.get("last_request", 0.0)
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)


def completion_with_retry(client: OpenAI, state: dict, **kwargs):
    delays = (15, 30, 60)
    for attempt in range(len(delays) + 1):
        wait_before_request(state)
        try:
            state["last_request"] = time.monotonic()
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            if attempt == len(delays):
                raise
            retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
            try:
                delay = max(delays[attempt], float(retry_after))
            except (TypeError, ValueError):
                delay = delays[attempt]
            st.warning(f"Limite Albert atteinte. Nouvelle tentative dans {int(delay)} secondes.")
            time.sleep(min(delay, 90))


def normalize_extracted_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    output, paragraph = [], []
    for line in lines:
        if not line:
            if paragraph:
                output.append(" ".join(paragraph)); paragraph = []
            continue
        is_heading = len(line) < 100 and (line.isupper() or line.endswith(":"))
        is_bullet = bool(re.match(r"^(?:[-•▪◦]|\d+[.)])\s+", line))
        if is_heading or is_bullet:
            if paragraph:
                output.append(" ".join(paragraph)); paragraph = []
            output.append(line)
        else:
            paragraph.append(line)
    if paragraph:
        output.append(" ".join(paragraph))
    return "\n\n".join(output).strip()


def render_page_jpeg(page: fitz.Page, dpi: int = 170) -> bytes:
    return page.get_pixmap(dpi=dpi, alpha=False).tobytes("jpeg", jpg_quality=88)


def ocr_page(client: OpenAI, state: dict, page: fitz.Page) -> str:
    image = base64.b64encode(render_page_jpeg(page)).decode("ascii")
    response = completion_with_retry(client, state, model=OCR_MODEL, messages=[{
        "role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]
    }], temperature=0.0, max_tokens=4096)
    return normalize_extracted_text(response.choices[0].message.content or "")


def extract_pdf(pdf_bytes: bytes, client: OpenAI, state: dict, progress) -> tuple[list[dict], int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count > MAX_PAGES:
        count = doc.page_count; doc.close()
        raise ValueError(f"Le PDF contient {count} pages. La limite est de {MAX_PAGES} pages.")
    pages, ocr_count = [], 0
    try:
        for index, page in enumerate(doc):
            progress.progress(index / max(1, doc.page_count), text=f"Extraction de la page {index + 1}/{doc.page_count}")
            text = normalize_extracted_text(page.get_text("text", sort=True))
            if len(re.sub(r"\s+", "", text)) < 40:
                text = ocr_page(client, state, page); ocr_count += 1
            pages.append({"page": index + 1, "text": text})
        return pages, ocr_count
    finally:
        doc.close()


def split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    if len(paragraph) <= limit:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
    parts, current = [], ""
    for sentence in sentences:
        words = sentence.split() if len(sentence) > limit else [sentence]
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > limit:
                parts.append(current); current = word
            else:
                current = candidate
    if current:
        parts.append(current)
    return parts


def make_chunks(pages: list[dict], limit: int = CHUNK_SIZE) -> list[str]:
    units = []
    for page in pages:
        units.append(f"[[PAGE {page['page']}]]")
        for paragraph in [p.strip() for p in re.split(r"\n\s*\n", page["text"]) if p.strip()]:
            units.extend(split_long_paragraph(paragraph, limit - 100))
    chunks, current = [], ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip()
        if current and len(candidate) > limit:
            chunks.append(current); current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def clean_translation(text: str) -> str:
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return re.sub(r"^(?:Voici|Ci-dessous)\s+(?:la\s+)?traduction\s*:?\s*", "", text, flags=re.I).strip()


def translate_chunk(client, state, model, chunk, source, target, style, glossary, previous_tail):
    continuity = f"\nContexte terminologique de la fin du bloc précédent, à ne pas répéter :\n{previous_tail}\n" if previous_tail else ""
    glossary_instruction = f"\nGlossaire obligatoire :\n{glossary.strip()}\n" if glossary.strip() else ""
    system = f"""Tu es un traducteur professionnel de documents administratifs et d'information au public.
Traduis du {LANGUAGES[source]['prompt']} vers le {LANGUAGES[target]['prompt']}.
{STYLES[style]}
Règles impératives :
- renvoie uniquement la traduction ;
- ne résume rien et ne supprime aucune information ;
- n'ajoute aucune explication ni note ;
- conserve les marqueurs [[PAGE n]] exactement, sans les traduire ;
- conserve les paragraphes, listes, noms propres, sigles, nombres, dates, montants et références ;
- reste cohérent dans le choix des termes sur tout le document ;
- si un terme administratif n'a pas d'équivalent fiable, conserve le terme français entre parenthèses ;
- pour une variété arabe dialectale, respecte précisément la variété demandée.
{glossary_instruction}{continuity}"""
    response = completion_with_retry(client, state, model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": chunk}], temperature=0.0, max_tokens=8192)
    translated = clean_translation(response.choices[0].message.content or "")
    if not translated:
        raise ValueError("Le modèle n'a renvoyé aucun texte.")
    return translated


def translate_document(client, model, chunks, source, target, style, glossary, progress, state):
    results = []
    for index, chunk in enumerate(chunks):
        progress.progress(index / max(1, len(chunks)), text=f"Traduction du bloc {index + 1}/{len(chunks)}")
        results.append(translate_chunk(client, state, model, chunk, source, target, style, glossary, results[-1][-1200:] if results else ""))
    progress.progress(1.0, text="Traduction terminée")
    return "\n\n".join(results)


def translate_audio_text(client, state, model, text, source, target) -> str:
    system = f"""Tu es un traducteur professionnel.
Traduis fidèlement du {LANGUAGES[source]['prompt']} vers le {LANGUAGES[target]['prompt']}.
Renvoie uniquement la traduction, sans commentaire, variante ni Markdown.
Préserve les noms propres, dates, nombres, références et retours à la ligne. Ne résume pas."""
    response = completion_with_retry(client, state, model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": text}], temperature=0.0, max_tokens=8192)
    result = clean_translation(response.choices[0].message.content or "")
    if not result:
        raise ValueError("Le modèle n'a renvoyé aucune traduction.")
    return result


def transcribe_audio(client: OpenAI, uploaded, asr_model: str, source: str) -> str:
    raw = uploaded.getvalue()
    if not raw:
        raise ValueError("Le fichier audio est vide.")
    if len(raw) > MAX_AUDIO_MB * 1024 * 1024:
        raise ValueError(f"Le fichier audio dépasse {MAX_AUDIO_MB} Mo.")
    audio = io.BytesIO(raw)
    audio.name = getattr(uploaded, "name", None) or "enregistrement.wav"
    response = client.audio.transcriptions.create(model=asr_model, file=audio, language=LANGUAGES[source]["code"], response_format="json")
    text = getattr(response, "text", None) or (response if isinstance(response, str) else "")
    if not text:
        raise ValueError("Le modèle de transcription n'a renvoyé aucun texte.")
    return text.strip()


def make_docx(text: str, title: str, source: str, target: str) -> bytes:
    document = Document(); document.add_heading(title, level=0); document.add_paragraph(f"Traduction : {source} → {target}")
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block: continue
        match = re.fullmatch(r"\[\[PAGE (\d+)\]\]", block)
        if match:
            document.add_heading(f"Page source {match.group(1)}", level=1); continue
        paragraph = document.add_paragraph(block)
        if LANGUAGES[target]["code"].startswith("ar"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    buffer = io.BytesIO(); document.save(buffer); return buffer.getvalue()


def make_pdf(text: str, title: str, source: str, target: str) -> bytes:
    from html import escape
    direction = "rtl" if LANGUAGES[target]["code"].startswith("ar") else "ltr"
    alignment = "right" if direction == "rtl" else "left"
    blocks = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block: continue
        match = re.fullmatch(r"\[\[PAGE (\d+)\]\]", block)
        blocks.append(f"<h2>Page source {match.group(1)}</h2>" if match else f"<p>{escape(block).replace(chr(10), '<br>')}</p>")
    html = f'<article dir="{direction}"><h1>{escape(title)}</h1><p class="meta">Traduction : {escape(source)} vers {escape(target)}</p>{"".join(blocks)}</article>'
    css = f'@page {{size:A4;margin:18mm 17mm}} body {{font-family:sans-serif;font-size:10.5pt;line-height:1.45;color:#172033}} article {{direction:{direction};text-align:{alignment}}} h1 {{font-size:20pt;color:#17365d}} h2 {{font-size:13pt;color:#245b8a;page-break-after:avoid}} p {{margin:0 0 4mm}} .meta {{color:#526579;margin-bottom:8mm}}'
    output = io.BytesIO(); writer = fitz.DocumentWriter(output); story = fitz.Story(html=html, user_css=css)
    mediabox = fitz.paper_rect("a4"); rect = fitz.Rect(48, 48, mediabox.width - 48, mediabox.height - 48); more = True
    while more:
        device = writer.begin_page(mediabox); more, _ = story.place(rect); story.draw(device); writer.end_page()
    writer.close(); return output.getvalue()


def preview_pdf(pdf_bytes: bytes, height: int = 720) -> None:
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    components.html(f'<div style="width:100%;height:{height}px;border:1px solid #d8dee8;border-radius:8px;overflow:hidden"><iframe src="data:application/pdf;base64,{encoded}#toolbar=1&navpanes=0&view=FitH" title="Prévisualisation du PDF importé" style="width:100%;height:100%;border:0"></iframe></div>', height=height + 8, scrolling=False)


def get_secret(name: str, default: str = "") -> str:
    """Lit un secret Streamlit, puis une variable d'environnement en secours."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name, default)
    return str(value or "").strip()


def require_access_password() -> None:
    """Protège l'application avec APP_PASSWORD, sans exposer ce secret au navigateur."""
    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.error("Le mot de passe d'accès APP_PASSWORD n'est pas configuré côté serveur.")
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.subheader("Accès protégé")
    with st.form("access_form", clear_on_submit=True):
        supplied = st.text_input("Mot de passe", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Se connecter", type="primary")

    if submitted:
        if hmac.compare_digest(supplied, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()


st.title("Traduction avec Albert")
st.caption("Traduction de PDF avec OCR, ou transcription et traduction d'un enregistrement vocal.")
require_access_password()
api_key = get_secret("ALBERT_API_KEY")
if not api_key:
    st.error("La clé ALBERT_API_KEY n'est pas configurée dans les secrets Streamlit.")
    st.stop()
try:
    models = list_translation_models(api_key.strip())
except Exception as exc:
    models, model_error = [], str(exc)
with st.sidebar:
    st.subheader("Modèle de traduction")
    st.caption("La variable ALBERT_TRANSLATION_MODEL définit uniquement le choix initial. Vous pouvez changer de modèle ici pour chaque test, sans redémarrer l'application.")
    if models:
        preferred_model = DEFAULT_MODEL if DEFAULT_MODEL in models else ("openweight-small" if "openweight-small" in models else models[0])
        model = st.selectbox("Modèle Albert utilisé pour la traduction", models, index=models.index(preferred_model), key="translation_model", help="La liste est chargée depuis Albert API. Le modèle sélectionné sera utilisé pour la prochaine traduction. Le modèle OCR reste indépendant.")
        st.info(f"Modèle sélectionné : {model}")
        if st.button("Actualiser la liste des modèles", use_container_width=True):
            list_translation_models.clear(); list_asr_models.clear(); st.rerun()
    else:
        st.warning("Impossible de charger les modèles automatiquement.")
        model = st.text_input("Identifiant du modèle de traduction", value=DEFAULT_MODEL, key="translation_model_manual", help="Saisissez exactement un identifiant accepté par votre instance Albert API.")
        with st.expander("Détail de l'erreur"):
            st.code(model_error if "model_error" in locals() else "Liste vide")

document_tab, audio_tab = st.tabs(["📄 Document PDF", "🎙️ Voix et audio"])

with document_tab:
    col1, col2 = st.columns(2)
    source = col1.selectbox("Langue source", list(LANGUAGES), index=list(LANGUAGES).index("Français"), key="doc_source")
    targets = [name for name in LANGUAGES if name != source]
    target = col2.selectbox("Langue cible", targets, index=targets.index("Anglais") if "Anglais" in targets else 0, key="doc_target")
    style = st.selectbox("Style", list(STYLES), index=0)
    glossary = st.text_area("Glossaire facultatif", placeholder="Une ligne par terme, par exemple :\nrelevé de carrière = ...\npension de retraite = ...", height=90)
    uploaded = st.file_uploader("PDF à traduire", type=["pdf"], key="document_file")
    if uploaded:
        raw = uploaded.getvalue()
        if len(raw) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"Le fichier dépasse {MAX_FILE_MB} Mo."); st.stop()
        try:
            preview = fitz.open(stream=raw, filetype="pdf"); page_count = preview.page_count; preview.close()
            st.caption(f"{uploaded.name} · {page_count} page(s) · {len(raw) / 1024 / 1024:.1f} Mo")
            with st.expander("Prévisualiser le document importé", expanded=True):
                preview_pdf(raw)
        except Exception:
            st.error("Le fichier PDF ne peut pas être lu."); st.stop()
        if st.button("Traduire le PDF", type="primary", use_container_width=True):
            client = make_client(api_key.strip()); state = {"last_request": 0.0}; progress = st.progress(0.0, text="Préparation")
            try:
                pages, ocr_count = extract_pdf(raw, client, state, progress)
                empty_pages = [p["page"] for p in pages if not p["text"].strip()]
                if empty_pages: raise ValueError("Aucun texte n'a été extrait des pages : " + ", ".join(map(str, empty_pages)))
                chunks = make_chunks(pages)
                st.info(f"{page_count} page(s), {ocr_count} page(s) OCR, {len(chunks)} bloc(s) de traduction · modèle : {model}")
                translation = translate_document(client, model.strip(), chunks, source, target, style, glossary, progress, state)
                title = f"Traduction de {Path(uploaded.name).stem}"
                st.session_state["translation_result"] = {"text": translation, "docx": make_docx(translation, title, source, target), "pdf": make_pdf(translation, title, source, target), "stem": Path(uploaded.name).stem, "target_code": LANGUAGES[target]["code"], "model": model.strip()}
            except APIStatusError as exc:
                st.error(f"Albert API a répondu avec l'erreur HTTP {exc.status_code}."); st.code(str(exc), language="text")
            except APIConnectionError as exc:
                st.error("Connexion à Albert API impossible."); st.code(str(exc), language="text")
            except Exception as exc:
                st.error(f"Échec : {type(exc).__name__}: {exc}")
            finally:
                client.close(); progress.empty()
    result = st.session_state.get("translation_result")
    if result:
        st.subheader("Traduction"); st.caption(f"Modèle utilisé : {result.get('model', 'non renseigné')}"); st.text_area("Résultat", result["text"], height=500)
        pdf_col, docx_col = st.columns(2)
        pdf_col.download_button("Télécharger en PDF", result["pdf"], file_name=f"{result['stem']}_{result['target_code']}.pdf", mime="application/pdf", use_container_width=True)
        docx_col.download_button("Télécharger en Word", result["docx"], file_name=f"{result['stem']}_{result['target_code']}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)

with audio_tab:
    st.subheader("Transcription audio suivie d'une traduction")
    try:
        asr_models = list_asr_models(api_key.strip()); asr_error = ""
    except Exception as exc:
        asr_models, asr_error = [], str(exc)
    if asr_models:
        asr_model = st.selectbox("Modèle de transcription", asr_models)
    else:
        st.warning("Impossible de charger automatiquement les modèles de transcription.")
        asr_model = st.text_input("Identifiant du modèle de transcription", placeholder="automatic-speech-recognition")
        if asr_error:
            with st.expander("Détail de l'erreur ASR"): st.code(asr_error)
    a1, a2 = st.columns(2)
    audio_source = a1.selectbox("Langue parlée", list(LANGUAGES), index=list(LANGUAGES).index("Français"), key="audio_source")
    audio_targets = [name for name in LANGUAGES if name != audio_source]
    audio_target = a2.selectbox("Langue de traduction", audio_targets, index=audio_targets.index("Anglais") if "Anglais" in audio_targets else 0, key="audio_target")
    audio_mode = st.radio("Source audio", ["Enregistrer avec le microphone", "Importer un fichier"], horizontal=True)
    if audio_mode == "Enregistrer avec le microphone":
        audio_file = st.audio_input("Enregistrer maintenant", sample_rate=16000, help="Autorisez le navigateur à utiliser le microphone, puis démarrez l'enregistrement.")
    else:
        audio_file = st.file_uploader("Fichier MP3 ou WAV", type=["mp3", "wav"], key="audio_file")
    if audio_file:
        audio_bytes = audio_file.getvalue(); st.audio(audio_bytes, format=getattr(audio_file, "type", None) or "audio/wav")
        st.caption(f"{getattr(audio_file, 'name', 'enregistrement.wav')} · {len(audio_bytes) / 1024 / 1024:.2f} Mo · modèle de traduction : {model}")
        if st.button("Transcrire puis traduire", type="primary", use_container_width=True, disabled=not asr_model.strip()):
            client = make_client(api_key.strip()); state = {"last_request": 0.0}
            try:
                with st.status("Traitement en cours", expanded=True) as status:
                    st.write("1/2 Transcription"); transcript = transcribe_audio(client, audio_file, asr_model.strip(), audio_source)
                    st.write("2/2 Traduction"); audio_translation = translate_audio_text(client, state, model.strip(), transcript, audio_source, audio_target)
                    st.session_state["audio_result"] = {"transcript": transcript, "translation": audio_translation, "source": audio_source, "target": audio_target, "model": model.strip()}
                    status.update(label="Traitement terminé", state="complete")
            except Exception as exc:
                st.error(f"Échec du traitement audio : {type(exc).__name__}: {exc}")
            finally:
                client.close()
    audio_result = st.session_state.get("audio_result")
    if audio_result:
        t1, t2 = st.tabs([f"Transcription en {audio_result['source']}", f"Traduction en {audio_result['target']}"])
        with t1: st.text_area("Texte transcrit", audio_result["transcript"], height=300)
        with t2:
            st.caption(f"Modèle utilisé : {audio_result['model']}")
            st.text_area("Texte traduit", audio_result["translation"], height=300)
