# 👜 Multimodal Recommendation System (Image + Text)  
### Chanel Product Dataset

Ce projet implémente une **application Streamlit** permettant de rechercher et recommander des produits Chanel à partir :

- 🖼️ **d’une image** (similarité visuelle)
- 📝 **d’un texte ou d’une description** (similarité sémantique)
- 🔗 **d’une combinaison image + texte** (fusion multimodale)

Il s’agit d’un système complet de **Computer Vision**, **NLP**, **extraction d’embeddings** et **recherche par similarité**, utilisant **FAISS**, **ResNet50**, **LBP/HOG**, **Word2Vec**, **TF-IDF** et **Sentence-BERT**.

---

## 🚀 Fonctionnalités principales

### 🔹 1. Chargement du dataset

- Téléchargement automatique depuis **HuggingFace** :  
  `DBQ/Chanel.Product.prices.Germany`
- Possibilité d’**uploader un fichier CSV**
- Sélection manuelle des colonnes :
  - `imageurl`
  - `title`
  - catégories
  - prix

---

### 🔹 2. Nettoyage intelligent des données

- Suppression des titres vides ou trop courts
- Suppression des doublons
- Mise en minuscules
- Filtrage des lignes sans image ou description valide

---

### 🔹 3. Téléchargement et prétraitement des images

- Téléchargement des images du dataset
- Sauvegarde locale dans `data/processed_images/`
- Redimensionnement uniforme en **224 × 224**
- Possibilité de travailler sur un **échantillon** du dataset

---

### 🔹 4. Extraction des embeddings

#### 🖼️ Embeddings visuels
- **ResNet50** — 2048 dimensions
- **LBP** — descripteurs de texture
- **HOG** — descripteurs de forme

#### 📝 Embeddings textuels
- **Sentence-BERT** (`all-MiniLM-L6-v2`)
- **Word2Vec** (entraîné sur le dataset)
- **TF-IDF**

---

### 🔹 5. Recherche et recommandation

- Recherche par **image**
- Recherche par **texte**
- Recherche par **fusion multimodale** (Image + Texte)
- Pondération configurable entre image et texte
- Visualisation interactive des résultats dans Streamlit

---

## 🏗 Architecture du projet

	📁 project/
	│
	├── app1.py                   # Application Streamlit complète
	├── README.md                 # Documentation du projet
	│
	├── data/
	│   ├── processed_images/     # Images prétraitées
	│   └── embeddings/           # Embeddings sauvegardés (optionnel)
	│
	└── requirements.txt          # Liste des dépendances (optionnel)

## 🛠 Technologies utilisées

### 🔹 Computer Vision
- ResNet50 (torchvision)
- HOG, LBP (scikit-image)
- PIL

### 🔹 NLP
- Sentence-BERT (sentence-transformers)
- Word2Vec (gensim)
- TF-IDF (scikit-learn)

### 🔹 Similarité & Indexation
- FAISS (`IndexFlatIP`)
- Cosine Similarity

### 🔹 Interface utilisateur
- **Streamlit**

---

## 📦 Installation

```bash
pip install -r requirements.txt

	- Streamlit
