👜 Multimodal Recommendation System (Image + Text) — Chanel Product Dataset

Ce projet implémente une application Streamlit permettant de rechercher et recommander des produits Chanel à partir :

	* d’une image (similarité visuelle),

	* d’un texte ou description (similarité sémantique),

	* ou d’une combinaison image + texte (fusion multimodale).

Il s’agit d’un système complet de traitement d’images, NLP, extraction d’embeddings, et recherche par similarité utilisant FAISS, ResNet50, LBP/HOG, Word2Vec, TF-IDF, et Sentence-BERT.

🚀 Fonctionnalités principales
🔹 1. Chargement dataset

Téléchargement automatique du dataset depuis HuggingFace : DBQ/Chanel.Product.prices.Germany

Ou upload d’un fichier CSV

Sélection manuelle des colonnes (imageurl, title, catégories, prix)

🔹 2. Nettoyage intelligent

Suppression des titres vides/courts

Suppression des doublons

Mise en minuscules

Filtrage des lignes sans images ou descriptions valides

🔹 3. Téléchargement et prétraitement d’images

Téléchargement des images du dataset

Sauvegarde en local (data/processed_images/)

Redimensionnement uniforme (224x224)

Option d’échantillonnage

🔹 4. Extraction des embeddings

Embeddings visuels :

ResNet50 (2048D)

LBP (texture)

HOG (shape)

Embeddings textuels :

Sentence-BERT (all-MiniLM-L6-v2)

Word2Vec (self-trained)

TF-IDF

🔹 5. Recherche et recommandation

Recherche par image

Recherche par texte

Recherche multimodale (late fusion)

KNN (FAISS ou cosine brute)

Visualisation des résultats

🏗 Architecture du projet
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

🛠 Technologies utilisées
🔹 Computer Vision

ResNet50 (torchvision)

HOG, LBP (scikit-image)

PIL

🔹 NLP

Sentence-BERT (sentence-transformers)

Word2Vec (gensim)

TF-IDF (scikit-learn)

🔹 Similarité & Indexation

FAISS (IndexFlatIP)

Cosine Similarity

🔹 Interfaces

Streamlit