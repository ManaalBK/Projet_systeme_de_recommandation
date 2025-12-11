# app1.py
# -*- coding: utf-8 -*-
"""
Système de recommandation multimodal (image + texte) pour le dataset Chanel.
Usage:
    streamlit run app.py
Requirements (pip):
    pip install streamlit pandas numpy matplotlib seaborn pillow requests tqdm scikit-learn torchvision torch \
                sentence-transformers gensim scikit-image faiss-cpu datasets umap-learn
Notes:
 - Le script télécharge le dataset depuis HuggingFace (DBQ/Chanel.Product.prices.Germany).
"""

import os
import io
import time
from typing import List, Tuple, Optional, Dict

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

# ML / CV / NLP libs
import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray

# FAISS for fast nearest neighbor (CPU)
import faiss

# ---- Config ----
DATA_DIR = "data"
PROCESSED_IMG_DIR = os.path.join(DATA_DIR, "processed_images")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Streamlit page config
st.set_page_config(page_title="Recommandation multimodale", layout="wide")

# ---- Helpers: Dataset loading & basic cleaning ----
@st.cache_data(show_spinner=False)
def load_dataset_hf() -> pd.DataFrame:
    """Télécharge dataset (HuggingFace DBQ/Chanel.Product.prices.Germany) et renvoie un DataFrame.
    Si internet indisponible, l'utilisateur devra fournir un CSV local."""
    try:
        from datasets import load_dataset
        ds = load_dataset("DBQ/Chanel.Product.prices.Germany")
        df = ds["train"].to_pandas()
        return df
    except Exception as e:
        st.warning("Impossible de télécharger le dataset depuis HuggingFace automatiquement.\n"
                   "Tu peux uploader un CSV via l'interface plus bas.")
        return pd.DataFrame()  # vide: l'utilisateur devra uploader

def clean_text_column(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    df = df.copy()
    # Remplacer valeurs "n.a." etc. par NaN
    df[col] = df[col].replace(['n.a.', 'N/A', '', 'na', 'None'], np.nan)
    # Dropna pour la colonne
    df = df.dropna(subset=[col])
    # Lowercase
    df[col] = df[col].astype(str).str.lower()
    # Strip
    df[col] = df[col].str.strip()
    # Remove very short titles (<3 chars)
    df = df[df[col].str.len() >= 3]
    # Remove rows where title has <2 words (optionnel)
    df = df[df[col].str.split().apply(len) >= 5]
    return df

# ---- Image download & preprocessing ----
# Image transform used for feature extraction (ResNet)
resnet_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

# Augmentation set (applied only for dataset augmentation if needed)
augmentation_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])

def download_image_to_pil(url: str, timeout: float = 6.0) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception:
        return None

def save_processed_image(img: Image.Image, out_path: str, resize=(224, 224)) -> bool:
    try:
        img_resized = img.resize(resize, Image.LANCZOS)
        img_resized.save(out_path, format="JPEG", quality=90)
        return True
    except Exception:
        return False

# ---- Visual descriptors extraction ----
# 1) LBP descriptor (texture)
def extract_lbp_from_pil(img: Image.Image, radius=3, n_points=24) -> np.ndarray:
    arr = np.array(img.convert("L"))  # grayscale
    lbp = local_binary_pattern(arr, P=n_points, R=radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist

# 2) HOG descriptor (shape)
def extract_hog_from_pil(img: Image.Image, resize=(128,128)) -> np.ndarray:
    gray = rgb2gray(np.array(img.resize(resize)))
    feature_vec = hog(gray, orientations=9, pixels_per_cell=(8,8),
                      cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
    return feature_vec.astype("float32")

# 3) ResNet50 embeddings (deep learning)
@st.cache_resource(show_spinner=False)
def load_resnet50_feature_extractor():
    model = models.resnet50(pretrained=True)
    model.eval()
    # Remove last fc layer (avgpool stays, output shape 2048x1x1)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(DEVICE)
    return feature_extractor

def extract_resnet_from_pil(img: Image.Image) -> np.ndarray:
    img_t = resnet_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = load_resnet50_feature_extractor()(img_t)  # shape (1, 2048, 1, 1)
    feat = feat.squeeze().cpu().numpy().reshape(-1)  # shape (2048,)
    # l2-normalize for cosine usage
    norm = np.linalg.norm(feat) + 1e-10
    return (feat / norm).astype("float32")

# ---- Text descriptors ----
@st.cache_resource(show_spinner=False)
def load_sentence_transformer(name: str = "all-MiniLM-L6-v2"):
    # small and fast; change if you need bigger model
    model = SentenceTransformer(name, device=DEVICE)
    return model

def compute_sentencetransformer_embedding(model, text: str) -> np.ndarray:
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")

# Word2Vec training (lightweight)
def train_word2vec(sentences: List[List[str]], vector_size: int = 100) -> Word2Vec:
    model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)
    return model

def title_to_w2v_embedding(w2v_model: Word2Vec, title: str) -> np.ndarray:
    words = title.split()
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if not vectors:
        return np.zeros(w2v_model.vector_size, dtype="float32")
    emb = np.mean(vectors, axis=0)
    # normalize
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.astype("float32")

# TF-IDF vectorizer helper
@st.cache_resource(show_spinner=False)
def create_tfidf_vectorizer(corpus: List[str], max_features: int = 3000):
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
    vect.fit(corpus)
    return vect

# ---- FAISS helpers ----
def build_faiss_index(embeddings: np.ndarray, n_list: int = 100) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using inner product (works with normalized embeddings for cosine).
    For simplicity we use IndexFlatIP (exact). For large datasets use IndexIVFFlat + train step.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    # ensure float32 contiguous
    embeddings = embeddings.astype('float32')
    index.add(embeddings)
    return index

# ---- Utility: safe path creation & existence filter ----
def ensure_image_paths(df: pd.DataFrame, processed_dir: str = PROCESSED_IMG_DIR) -> pd.DataFrame:
    """Create 'image_path' column mapping index -> processed image path if file exists."""
    df = df.copy()
    df['image_path'] = [os.path.join(processed_dir, f"{idx}.jpg") for idx in df.index]
    # keep rows that actually exist
    df = df[df['image_path'].apply(os.path.exists)]
    return df

# ---- Pipeline: compute embeddings for small sample or full set ----
def compute_embeddings_for_df(df: pd.DataFrame,
                              max_items: Optional[int] = 500,
                              force_recompute: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute and return dict of embeddings:
      - 'resnet': (N, D_vis)
      - 'lbp': (N, D_lbp)
      - 'hog': (N, D_hog)
      - 'text_sentbert': (N, D_txt)
      - 'text_w2v': (N, D_w2v)
      - 'tfidf': scipy sparse matrix or dense (N, D_tfidf)
    """
    # Limit size to speed up experimentation
    if max_items is not None:
        df_proc = df.iloc[:max_items].copy()
    else:
        df_proc = df.copy()

    n = len(df_proc)
    st.info(f"Extraction d'embeddings pour {n} items (max_items={max_items}) ...")
    # Prepare containers
    resnet_embs = []
    lbp_embs = []
    hog_embs = []
    titles = df_proc['title'].astype(str).tolist()

    # Initialize SentenceTransformer
    sent_model = load_sentence_transformer()

    # Word2Vec training on tokenized sentences
    tokenized = [t.split() for t in titles]
    w2v_model = train_word2vec(tokenized, vector_size=100)

    # TF-IDF vectorizer
    tfidf_vect = create_tfidf_vectorizer(titles, max_features=3000)
    tfidf_matrix = tfidf_vect.transform(titles)  # sparse

    # loop
    progress = st.progress(0)
    for i, (_, row) in enumerate(df_proc.iterrows()):
        # open image
        img_path = row.get('image_path', None)
        if img_path and os.path.exists(img_path):
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                pil = None
        else:
            pil = None
        # If no local processed image, try downloading original url on-the-fly
        if pil is None and row.get('imageurl', None):
            pil = download_image_to_pil(row['imageurl'])
            # save as processed for future runs
            if pil is not None:
                save_processed_image(pil, os.path.join(PROCESSED_IMG_DIR, f"{row.name}.jpg"))

        if pil is None:
            # fallback: zero vectors
            resnet_embs.append(np.zeros(2048, dtype='float32'))
            lbp_embs.append(np.zeros(27, dtype='float32'))  # approx bins used by LBP with n_points=24
            hog_embs.append(np.zeros(4608, dtype='float32'))  # depends on HOG params; placeholder
        else:
            # ResNet
            resnet_embs.append(extract_resnet_from_pil(pil))
            # LBP
            lbp_embs.append(extract_lbp_from_pil(pil))
            # HOG
            hog_embs.append(extract_hog_from_pil(pil))
        progress.progress(int((i+1)/n * 100))

    # Convert lists to arrays (pad HOG dims if necessary)
    resnet_embs = np.vstack(resnet_embs).astype('float32')  # (N, 2048)
    lbp_embs = np.vstack(lbp_embs).astype('float32')
    # HOG arrays may vary; stack carefully
    try:
        hog_embs = np.vstack(hog_embs).astype('float32')
    except ValueError:
        # If HOG sizes differ, pad/truncate to max length
        maxlen = max(len(h) for h in hog_embs)
        hog_embs_fixed = np.zeros((len(hog_embs), maxlen), dtype='float32')
        for i, h in enumerate(hog_embs):
            L = len(h)
            hog_embs_fixed[i, :L] = h[:maxlen]
        hog_embs = hog_embs_fixed

    # Text embeddings
    txt_sent = np.vstack([compute_sentencetransformer_embedding(sent_model, t) for t in titles]).astype('float32')
    txt_w2v = np.vstack([title_to_w2v_embedding(w2v_model, t) for t in titles]).astype('float32')
    # tfidf_matrix remains sparse (scipy), convert to dense for small n
    tfidf_dense = tfidf_matrix.toarray().astype('float32')

    out = {
        "resnet": resnet_embs,
        "lbp": lbp_embs,
        "hog": hog_embs,
        "text_sentbert": txt_sent,
        "text_w2v": txt_w2v,
        "tfidf": tfidf_dense,
        "tfidf_vectorizer": tfidf_vect,
        "w2v_model": w2v_model
    }
    st.success("Extraction terminée.")
    return out

# ---- Search functions ----
def knn_search_faiss(index, query_emb: np.ndarray, topk: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    # FAISS expects shape (1, d) float32
    q = query_emb.reshape(1, -1).astype('float32')
    scores, indices = index.search(q, topk)
    # For IndexFlatIP, scores are inner products -> since vectors normalized, higher = closer.
    return indices[0], scores[0]

def brute_knn_cosine(embeddings: np.ndarray, query_emb: np.ndarray, topk: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    sims = cosine_similarity(query_emb.reshape(1,-1), embeddings).flatten()
    idx = np.argsort(-sims)[:topk]
    return idx, sims[idx]

# ---- UI: Streamlit App ----
def main():
    st.title("Système de recommandation multimodale (Image + Texte)")
    st.markdown("**Instructions** :\n- Charge le dataset automatique ou upload un CSV.\n- Prétraiter (download images) puis extraire embeddings (ResNet, LBP, HOG, TF-IDF, Word2Vec, SentenceTransformer).\n- Utilise la recherche par image / texte / combinée.")

    # Data load section
    st.header("1) Chargement du dataset")
    with st.expander("Options de chargement"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Charger dataset depuis HuggingFace (DBQ/Chanel...)")
            if st.button("Télécharger dataset (HuggingFace)"):
                df_local = load_dataset_hf()
                if df_local.empty:
                    st.warning("Dataset vide: uploader un CSV manuellement.")
                else:
                    st.session_state['df_raw'] = df_local
                    st.success(f"Dataset chargé: {len(df_local)} lignes")
        with col2:
            uploaded = st.file_uploader("Ou uploader un fichier CSV local (col includes imageurl, title, price, category2_code)", type=["csv"])
            if uploaded is not None:
                df_local = pd.read_csv(uploaded)
                st.session_state['df_raw'] = df_local
                st.success(f"CSV uploadé: {len(df_local)} lignes")

    # If dataset present in session, show preview
    df_raw = st.session_state.get('df_raw', None)
    if df_raw is None:
        st.info("Aucun dataset chargé. Charge via HuggingFace ou upload un CSV.")
        # allow exit early
        return

    st.write("Aperçu du dataset (colonnes) :")
    st.write(df_raw.columns.tolist())
    st.dataframe(df_raw.head(5))

    # Select important columns (allow user to map)
    st.header("2) Préparation & nettoyage")
    st.markdown("Sélectionner les colonnes correspondantes dans ton dataset.")
    col_image = st.selectbox("Colonne contenant l'URL de l'image", options=df_raw.columns.tolist(), index=df_raw.columns.tolist().index('imageurl') if 'imageurl' in df_raw.columns else 0)
    col_title = st.selectbox("Colonne contenant le titre/description", options=df_raw.columns.tolist(), index=df_raw.columns.tolist().index('title') if 'title' in df_raw.columns else 1)
    col_category = st.selectbox("Colonne contenant la catégorie (optionnel)", options=[None] + df_raw.columns.tolist(), index=0)
    col_price = st.selectbox("Colonne prix (optionnel)", options=[None] + df_raw.columns.tolist(), index=0)

    # Build df_important
    df_imp = df_raw[[col_image, col_title]].copy()
    df_imp.columns = ['imageurl', 'title']
    if col_category:
        df_imp['category2_code'] = df_raw[col_category]
    else:
        df_imp['category2_code'] = "unknown"
    if col_price:
        df_imp['price'] = df_raw[col_price]
    else:
        df_imp['price'] = np.nan

    st.write("Avant nettoyage :", df_imp.shape)
    if st.button("Nettoyer et filtrer (supprimer doublons, valeurs manquantes, titres courts)"):
        df_imp = df_imp.drop_duplicates(subset=['title', 'imageurl', 'category2_code'])
        df_imp = clean_text_column(df_imp, col='title')
        # Replace common missing strings in imageurl
        df_imp['imageurl'] = df_imp['imageurl'].replace(['n.a.', 'N/A', ''], np.nan)
        df_imp = df_imp.dropna(subset=['imageurl', 'title'])
        # Save to session
        st.session_state['df_cleaned'] = df_imp
        st.success(f"Nettoyage terminé : {len(df_imp)} items")

    df_cleaned = st.session_state.get('df_cleaned', None)
    if df_cleaned is None:
        st.info("Nettoyage non réalisé. Clique sur le bouton 'Nettoyer et filtrer' pour continuer.")
        return

    st.write(f"Taille du dataset nettoyé : {len(df_cleaned)}")
    st.dataframe(df_cleaned.head(5))

    # Image download / preprocessing step
    st.header("3) Téléchargement et prétraitement des images")
    st.markdown("Télécharger les images (échantillon conseillé) et les sauvegarder dans `data/processed_images`.")
    sample_size = st.number_input("Nombre d'images à prétraiter (0 = tout)", min_value=0, max_value=10000, value=200, step=50)
    run_download = st.button("Télécharger & prétraiter les images (sample)")

    if run_download:
        # choose subset
        if sample_size == 0 or sample_size >= len(df_cleaned):
            to_proc = df_cleaned.copy()
        else:
            to_proc = df_cleaned.sample(n=sample_size, random_state=42).copy()
        # process and save with index mapping
        pbar = st.progress(0)
        for i, (idx, row) in enumerate(to_proc.iterrows()):
            url = row['imageurl']
            out_path = os.path.join(PROCESSED_IMG_DIR, f"{idx}.jpg")
            if os.path.exists(out_path):
                # skip
                pbar.progress(int((i+1)/len(to_proc) * 100))
                continue
            img = download_image_to_pil(url)
            if img is None:
                # save nothing; could mark row
                pbar.progress(int((i+1)/len(to_proc) * 100))
                continue
            save_processed_image(img, out_path)
            pbar.progress(int((i+1)/len(to_proc) * 100))
        st.success("Téléchargement terminé. Vérifie le dossier data/processed_images")

    # Ensure image_path column
    if 'image_path' not in df_cleaned.columns:
        df_cleaned['image_path'] = [os.path.join(PROCESSED_IMG_DIR, f"{idx}.jpg") for idx in df_cleaned.index]

    # Keep only rows with downloaded images (optionnel)
    keep_only_with_images = st.checkbox("Utiliser seulement les items dont les images ont été téléchargées", value=True)
    if keep_only_with_images:
        df_use = df_cleaned[df_cleaned['image_path'].apply(os.path.exists)].copy()
    else:
        df_use = df_cleaned.copy()

    st.write(f"Items disponibles pour extraction: {len(df_use)}")

    # ---- Embeddings extraction ----
    st.header("4) Extraction des embeddings (visuel & textuel)")
    max_items = st.number_input("Max items pour extraction (0 = tout)", min_value=0, max_value=len(df_use), value=min(500, len(df_use)), step=50)
    if max_items == 0:
        max_items = None

    extract_btn = st.button("Extraire embeddings (ResNet, LBP, HOG, TF-IDF, Word2Vec, SentBERT)")

    if extract_btn:
        with st.spinner("Extraction en cours (cela peut prendre du temps) ..."):
            emb_dict = compute_embeddings_for_df(df_use, max_items=max_items)
            # Save embeddings in session for quick access
            st.session_state['embeddings'] = emb_dict
            # store a working df subset used for embeddings
            st.session_state['df_emb'] = df_use.iloc[: (max_items if max_items else len(df_use)) ].copy()
        st.success("Embeddings extraits et sauvegardés en session.")

    # Get embeddings from session if present
    embeddings = st.session_state.get('embeddings', None)
    df_emb = st.session_state.get('df_emb', None)

    if embeddings is None or df_emb is None:
        st.info("Extraire d'abord les embeddings pour utiliser la recherche.")
        return

    st.write(f"Embeddings shapes: ResNet {embeddings['resnet'].shape}, text {embeddings['text_sentbert'].shape}")

    # Build FAISS indices (visual and text)
    if 'faiss_vis' not in st.session_state:
        # ensure normalization (already normalized in extract_resnet_from_pil and sentencetransformer)
        faiss_idx_vis = build_faiss_index(embeddings['resnet'])
        faiss_idx_txt = build_faiss_index(embeddings['text_sentbert'])
        st.session_state['faiss_vis'] = faiss_idx_vis
        st.session_state['faiss_txt'] = faiss_idx_txt
        st.session_state['df_emb_idx_to_df_idx'] = df_emb.index.to_list()
        st.success("Index FAISS construit (visuel & textuel).")

    # ---- UI Search ----
    st.header("5) Interface de recherche (image / texte / combinée)")
    mode = st.radio("Mode de recherche", ("Recherche par image", "Recherche par texte", "Recherche combinée"))

    top_k = st.slider("Nombre de recommandations", min_value=1, max_value=20, value=10)
    alpha = st.slider("Pondération image vs texte (pour recherche combinée) — alpha=image", 0.0, 1.0, 0.5, 0.05)

    # Helper to display results
    def display_results(indices: List[int], scores: List[float], df_index_map: List[int], title="Résultats"):
        st.subheader(title)
        cols = st.columns(5)
        for rank, idx in enumerate(indices):
            actual_idx = df_index_map[idx]
            row = df_emb.loc[actual_idx]
            # Show image
            try:
                img = Image.open(row['image_path'])
            except Exception:
                img = None
            with cols[rank % 5]:
                if img is not None:
                    st.image(img, use_column_width=True)
                st.markdown(f"**{row['title'][:120]}**")
                st.markdown(f"Catégorie: {row.get('category2_code', 'NA')}")
                st.markdown(f"Score: {float(scores[rank]):.4f}")

    if mode == "Recherche par image":
        uploaded_file = st.file_uploader("Téléversez une image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            query_img = Image.open(uploaded_file).convert("RGB")
            query_emb = extract_resnet_from_pil(query_img)
            idxs, scores = knn_search_faiss(st.session_state['faiss_vis'], query_emb, topk=top_k)
            display_results(idxs.tolist(), scores.tolist(), st.session_state['df_emb_idx_to_df_idx'], title="Top similaires (visuel)")
    elif mode == "Recherche par texte":
        qtxt = st.text_input("Texte / description")
        if qtxt:
            txt_model = load_sentence_transformer()
            q_emb = compute_sentencetransformer_embedding(txt_model, qtxt)
            idxs, scores = knn_search_faiss(st.session_state['faiss_txt'], q_emb, topk=top_k)
            display_results(idxs.tolist(), scores.tolist(), st.session_state['df_emb_idx_to_df_idx'], title="Top similaires (textuel)")
    else:
        uploaded_file = st.file_uploader("Téléversez une image (optionnel)", type=["jpg", "jpeg", "png"])
        qtxt = st.text_input("Texte / description (optionnel)")
        if (uploaded_file is not None) or (qtxt and len(qtxt.strip())>0):
            # compute emb img/text if provided
            if uploaded_file is not None:
                qimg = Image.open(uploaded_file).convert("RGB")
                qvis = extract_resnet_from_pil(qimg)
            else:
                qvis = None
            if qtxt and len(qtxt.strip())>0:
                txt_model = load_sentence_transformer()
                qtxt_emb = compute_sentencetransformer_embedding(txt_model, qtxt)
            else:
                qtxt_emb = None

            # Compute combined score
            # For speed we'll compute inner product scores via FAISS for each modality and combine
            n = top_k*5  # retrieve some candidates from each modality and then re-rank
            candidates = {}
            if qvis is not None:
                ids_v, scores_v = knn_search_faiss(st.session_state['faiss_vis'], qvis, topk=n)
                for i, s in zip(ids_v, scores_v):
                    candidates[int(i)] = candidates.get(int(i), 0.0) + alpha * float(s)
            if qtxt_emb is not None:
                ids_t, scores_t = knn_search_faiss(st.session_state['faiss_txt'], qtxt_emb, topk=n)
                for i, s in zip(ids_t, scores_t):
                    candidates[int(i)] = candidates.get(int(i), 0.0) + (1.0 - alpha) * float(s)
            # if only one modality provided, candidates contains only that modality scores
            # sort candidates by combined score
            cand_items = sorted(candidates.items(), key=lambda x: -x[1])[:top_k]
            if not cand_items:
                st.warning("Aucune correspondance trouvée.")
            else:
                idxs = [c[0] for c in cand_items]
                scores = [c[1] for c in cand_items]
                display_results(idxs, scores, st.session_state['df_emb_idx_to_df_idx'], title="Top multimodal (combiné)")

    st.markdown("---")

if __name__ == "__main__":
    main()
