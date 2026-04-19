#  Projet : Classification de Genres Musicaux par Traitement du Signal Audio

> **Analyse complète du signal audio sur le dataset GTZAN — de l'extraction de features jusqu'au Deep Learning**

---

##  Table des matières

1. [Présentation du projet](#présentation-du-projet)
2. [Dataset utilisé](#dataset-utilisé)
3. [Structure du projet](#structure-du-projet)
4. [Technologies et bibliothèques](#technologies-et-bibliothèques)
5. [Étape 1 — Chargement et visualisation du signal](#étape-1--chargement-et-visualisation-du-signal)
6. [Étape 2 — Extraction des features audio](#étape-2--extraction-des-features-audio)
7. [Étape 3 — Préparation des données pour le Machine Learning](#étape-3--préparation-des-données-pour-le-machine-learning)
8. [Étape 4 — Modélisation Machine Learning (KNN)](#étape-4--modélisation-machine-learning-knn)
9. [Étape 5 — Amélioration du modèle](#étape-5--amélioration-du-modèle)
10. [Étape 6 — Deep Learning (CNN sur Spectrogrammes Mel)](#étape-6--deep-learning-cnn-sur-spectrogrammes-mel)
11. [Résultats et performances](#résultats-et-performances)
12. [Bugs identifiés et corrections](#bugs-identifiés-et-corrections)
13. [Recommandations et pistes d'amélioration](#recommandations-et-pistes-damélioration)
14. [Comment exécuter le projet](#comment-exécuter-le-projet)

---

## Présentation du projet

Ce projet a pour objectif de **classifier automatiquement des morceaux de musique dans l'un des 10 genres musicaux** à partir de leurs caractéristiques acoustiques. Il s'inscrit dans le domaine du **traitement automatique du signal audio (DSP)** et du **Machine Learning appliqué à la musique**.

L'approche adoptée suit un pipeline classique en 3 grandes phases :

**Phase 1 — Analyse du signal**
On charge les fichiers audio et on en extrait des représentations visuelles et numériques : forme d'onde (waveform), fréquence fondamentale, énergie, taux de passage par zéro, et diverses caractéristiques spectrales.

**Phase 2 — Ingénierie des features**
Les caractéristiques extraites par frame sont agrégées en statistiques (moyenne et variance) pour produire un vecteur de taille fixe par morceau, utilisable par des algorithmes de Machine Learning.

**Phase 3 — Modélisation**
Deux approches de classification sont testées : un classifieur classique KNN optimisé avec GridSearchCV, et un réseau de neurones convolutif (CNN) qui traite les spectrogrammes Mel comme des images.

---

## Dataset utilisé

**Nom** : GTZAN Music Genre Classification  
**Source** : [Kaggle — andradaolteanu](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
**Contenu** : 1 000 extraits audio de 30 secondes chacun  
**Format** : fichiers `.wav`, taux d'échantillonnage 22 050 Hz  
**Organisation** : 10 genres × 100 fichiers par genre

| Genre | Description musicale |
|---|---|
|  Classical | Musique orchestrale, grande dynamique, instruments acoustiques |
|  Blues | Gamme blues, feeling mélancolique, guitare solo |
|  Country | Guitare acoustique, voix nasale, structures simples |
|  Disco | Tempo régulier ~120 BPM, ligne de basse groovy |
|  Hip-Hop | Beats samplés, voix rappée, basses lourdes |
| Jazz | Improvisation, harmonies complexes, cuivres |
|  Metal | Distorsion saturée, tempo rapide, batterie agressive |
|  Pop | Structures couplet-refrain, production lisse |
|  Reggae | Off-beat marqué (temps 2 et 4), basse proéminente |
|  Rock | Guitares électriques, batterie puissante, énergie |

---

## Structure du projet

```
projet-audio/
│
├── PROJET_Audio_Processing.ipynb     ← Notebook principal (analyse complète)
│
├── gtzan/                            ← Dataset GTZAN (téléchargé via Kaggle)
│   └── Data/
│       └── genres_original/
│           ├── classical/            ← 100 fichiers .wav
│           ├── blues/
│           ├── country/
│           ├── disco/
│           ├── hiphop/
│           ├── jazz/
│           ├── metal/
│           ├── pop/
│           ├── reggae/
│           └── rock/
│
├── feature_matrix_audio_classical.csv   ← Features MFCC par genre
├── feature_matrix_audio_reggae.csv
├── feature_matrix_audio_jazz.csv
├── feature_matrix_audio_blues.csv
├── feature_matrix_audio_country.csv
├── feature_matrix_audio_disco.csv
├── feature_matrix_audio_hiphop.csv
├── feature_matrix_audio_metal.csv
├── feature_matrix_audio_pop.csv
├── feature_matrix_audio_rock.csv
│
└── feature_matrix_audio_fusionner.csv   ← Dataset ML final (tous genres)
```

---

## Technologies et bibliothèques

| Bibliothèque | Version recommandée | Utilisation |
|---|---|---|
| `Python` | 3.9+ | Langage principal |
| `librosa` | 0.10+ | Analyse audio (MFCC, ZCR, spectrogrammes...) |
| `numpy` | 1.24+ | Calculs numériques sur tableaux |
| `pandas` | 2.0+ | Gestion des DataFrames et CSV |
| `matplotlib` | 3.7+ | Visualisation des signaux et graphiques |
| `seaborn` | 0.12+ | Heatmaps (matrice de confusion) |
| `scikit-learn` | 1.3+ | KNN, StandardScaler, GridSearchCV |
| `tensorflow / keras` | 2.13+ | CNN Deep Learning |
| `soundfile` | 0.12+ | Lecture des fichiers audio |
| `scipy` | 1.11+ | Traitement du signal bas niveau |

---

## Étape 1 — Chargement et visualisation du signal

### Chargement audio avec `librosa.load()`

Chaque fichier audio est chargé en mémoire sous forme d'un **tableau numpy** représentant les échantillons du signal dans le temps.

```python
y, sr = librosa.load("fichier.wav", sr=None, mono=True)
```

- `y` : tableau des amplitudes (forme d'onde), typiquement ~660 000 valeurs pour 30 secondes
- `sr` : taux d'échantillonnage (22 050 Hz dans GTZAN = 22 050 mesures par seconde)
- `sr=None` : conserve le taux d'origine sans rééchantillonnage
- `mono=True` : convertit stéréo → mono par moyenne des canaux

### Visualisation : Waveform (forme d'onde)

```python
librosa.display.waveshow(y, sr=sr)
```

La waveform montre comment l'amplitude du son évolue dans le temps. C'est la représentation la plus directe du signal audio.

**Ce qu'on observe par genre :**
- **Classical** : grandes variations dynamiques, passages doux et forts alternés, silences présents
- **Reggae** : amplitude relativement stable, accents rythmiques réguliers visibles
- **Jazz** : très irrégulier, solos expressifs créant des pics d'amplitude imprévisibles

> **Limitation** : la waveform seule ne révèle aucune information sur les fréquences présentes. Elle donne l'énergie globale mais pas le timbre ni la tonalité.

---

## Étape 2 — Extraction des features audio

Cette étape constitue le cœur du projet. On extrait 6 types de caractéristiques acoustiques, chacune capturant un aspect différent du son.

---

### 2.1 Fréquence Fondamentale (F0) — `librosa.pyin()`

**Concept** : La fréquence fondamentale (ou "pitch") est la hauteur perçue d'un son — ce qui distingue un Do grave d'un Do aigu.

**Algorithme pYIN** (Probabilistic YIN) : amélioration de l'algorithme YIN, il estime la F0 frame par frame en cherchant les périodes de répétition dans le signal.

```python
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=1000, sr=sr)
f0_clean = np.nan_to_num(f0)  # NaN → 0 pour les zones sans note (bruit, silence)
```

- `fmin=80 Hz` : fréquence minimale analysée (environ Do grave d'une contrebasse)
- `fmax=1000 Hz` : fréquence maximale (couvre voix, guitare, violon)
- Les zones sans pitch défini (percussions, bruit) retournent `NaN`

**Interprétation par genre :**
- Classical : large plage de F0 (contrebasse ~80 Hz → violon ~800 Hz)
- Reggae : F0 vocal stable et répétitif, ligne de basse constante
- Jazz : F0 très variable à cause de l'improvisation et du vibrato

---

### 2.2 Amplitude et Intensité — RMS + dB

**Trois mesures complémentaires** pour caractériser l'énergie du signal :

**Amplitude crête** : valeur maximale absolue du signal — donne le pic d'énergie mais peu représentative de l'énergie globale.

```python
amplitude = np.max(np.abs(y))
```

**RMS (Root Mean Square)** : énergie moyenne par fenêtre temporelle — correspond mieux à l'énergie sonore réellement perçue.

```python
rms = librosa.feature.rms(y=y)[0]
```

**Intensité en dB** : conversion en échelle logarithmique, plus proche de la perception humaine (l'oreille perçoit les sons de façon logarithmique).

```python
intensity_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
```

**Interprétation par genre :**
- Classical : grande dynamique, RMS très variable (passages pianissimo vs. fortissimo)
- Reggae : RMS stable (compression audio typique du reggae moderne)
- Jazz : RMS variable (solos forts, accompagnements doux)

---

### 2.3 Zero-Crossing Rate (ZCR) — `librosa.feature.zero_crossing_rate()`

**Concept** : Le ZCR compte combien de fois par seconde le signal change de signe (passe de positif à négatif ou inversement). Un son pur (sinusoïde) a un ZCR proportionnel à sa fréquence. Un son bruité (cymbale, souffle) a un ZCR très élevé.

```python
zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
```

- `frame_length=2048` : fenêtre d'analyse de ~93 ms
- `hop_length=512` : décalage de ~23 ms entre deux fenêtres consécutives

**Graphique** : le signal brut (bleu) et la courbe ZCR (rouge) sont superposés pour montrer la corrélation entre l'agitation du signal et son taux de passage par zéro.

**Valeurs typiques :**

| Genre | ZCR moyen | Raison |
|---|---|---|
| Classical | ~0.04–0.06 | Instruments harmoniques purs (cordes) |
| Reggae | ~0.07–0.09 | Mix voix + guitare + percussions |
| Jazz | ~0.08–0.11 | Cuivres + improvisations complexes |
| Metal | ~0.12–0.18 | Distorsion saturée, cymbales omniprésentes |

**Utilité en classification** : le ZCR permet de distinguer les genres "harmoniques" (classical) des genres "bruités/percussifs" (metal, hiphop).

---

### 2.4 Centroïde Spectral — `librosa.feature.spectral_centroid()`

**Concept** : Le centroïde spectral est le "centre de gravité" du spectre fréquentiel. Il indique autour de quelle fréquence l'énergie sonore est concentrée à un instant donné. Un son "brillant" (cymbale, violon aigu) a un centroïde élevé ; un son "sombre" (contrebasse, orgue grave) a un centroïde bas.

```python
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
```

**Graphique** : spectrogramme STFT (axe temps × fréquence, couleur = énergie en dB) avec la courbe du centroïde spectral superposée en cyan. Cela permet de voir si le centroïde suit bien les zones d'énergie du spectrogramme.

**Valeurs typiques :**

| Genre | SC moyen | Interprétation |
|---|---|---|
| Classical | ~1 500–2 500 Hz | Énergie centrée sur médiums-aigus |
| Reggae | ~1 800–2 800 Hz | Guitare rythmique et voix dominent |
| Jazz | ~2 000–3 500 Hz | Cuivres apportent des harmoniques aigus |
| Metal | ~3 000–5 000 Hz | Distorsion + cymbales = beaucoup d'aigus |

---

### 2.5 Largeur de Bande Spectrale — `librosa.feature.spectral_bandwidth()`

**Concept** : La largeur de bande mesure l'étalement de l'énergie autour du centroïde spectral. C'est l'écart-type pondéré du spectre. Un son riche en harmoniques (orchestre complet, guitare distordue) a une large bande ; un son tonal pur (flûte, synthé) a une bande étroite.

```python
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
```

**Graphique** : spectrogramme (colormap `Reds`) + courbe de bandwidth (bleue). La courbe indique à chaque instant l'étalement spectral du son.

- **Bande basse** ≈ centroïde − bandwidth
- **Bande haute** ≈ centroïde + bandwidth

Ces deux valeurs définissent la "zone active" du spectre sonore.

**Interprétation par genre :**
- Classical : bandwidth variable selon les instruments (violon seul = étroit, orchestre complet = large)
- Jazz : large (cuivres aux harmoniques complexes)
- Reggae : modérée (mix équilibré, peu de saturation)

---

### 2.6 Fréquence de Coupure Spectrale (Rolloff) — `librosa.feature.spectral_rolloff()`

**Concept** : Le rolloff spectral est la fréquence en dessous de laquelle se trouve 85% de l'énergie spectrale totale. C'est un indicateur de la "brillance" du son.

```python
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
```

- `roll_percent=0.85` : seuil fixé à 85% de l'énergie cumulée

**Graphique** : spectrogramme + courbe rolloff (bleue). Quand la courbe est haute, le son est brillant ; quand elle est basse, le son est sombre.

**Valeurs typiques :**

| Genre | Rolloff moyen | Caractéristique |
|---|---|---|
| Classical | ~3 000–5 000 Hz | Large spectre orchestral |
| Reggae | ~1 500–3 000 Hz | Énergie concentrée dans les basses |
| Jazz | ~3 000–6 000 Hz | Cuivres apportent les hautes fréquences |
| Metal | ~5 000–8 000 Hz | Distorsion + cymbales = beaucoup d'aigus |

**Utilité en classification** : Le rolloff est l'une des features les plus discriminantes entre genres "brillants" (metal, classical) et genres "sombres" (reggae, blues).

---

### 2.7 MFCC — Mel-Frequency Cepstral Coefficients — `librosa.feature.mfcc()`

**Concept** : Les MFCC sont les features les plus importantes en reconnaissance et classification audio. Ils modélisent le **timbre** du son, c'est-à-dire la "couleur" sonore qui distingue un violon d'une flûte même à la même note.

**Pipeline de calcul en 5 étapes :**

```
Signal brut
    → Découpage en frames (fenêtres de ~93ms)
    → FFT (transformée de Fourier) → spectre de puissance
    → Application du banc de filtres Mel (échelle perceptuelle)
    → Logarithme (compression dynamique)
    → DCT (décorrélation) → 13 coefficients
```

```python
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# Résultat : matrice de forme (13, T) où T = nombre de frames
```

**Signification de chaque coefficient :**
- **MFCC-0** : énergie globale du signal
- **MFCC-1 à 4** : structure spectrale basse fréquence (timbre grave, résonance)
- **MFCC-5 à 12** : détail haute fréquence (texture, brillance)

**Graphique MFCC** : heatmap temps × coefficient (couleur = valeur). On peut lire la stabilité ou la variabilité du timbre au fil du temps.

**Interprétation par genre :**
- Classical : MFCC-0 très variable (grande dynamique), MFCC supérieurs stables
- Reggae : MFCC-0 constant (compression), MFCC-1/2 négatifs (basses dominantes)
- Jazz : tous les MFCC très variables (improvisation = timbre changeant constamment)
- Metal : MFCC-0 élevé et constant (saturation permanente)

**Pourquoi 13 coefficients ?** C'est le nombre optimal pour capturer l'essentiel du timbre sans surcharger le modèle. Les 13 coefficients couvrent la plage auditive utile pour la discrimination des genres.

---

## Étape 3 — Préparation des données pour le Machine Learning

### 3.1 Agrégation statistique (Moyenne + Variance)

Les MFCC produisent une matrice `(13 × T)` où T dépend de la durée du fichier. Pour utiliser un modèle ML classique, chaque morceau doit être représenté par un **vecteur de taille fixe**.

Solution : résumer la matrice temporelle par des statistiques.

```python
mfcc_mean = np.mean(mfccs, axis=1)  # → vecteur (13,) : timbre moyen
mfcc_var  = np.var(mfccs, axis=1)   # → vecteur (13,) : variabilité du timbre
```

**Résultat** : chaque morceau est décrit par **26 valeurs** (13 moyennes + 13 variances).

### 3.2 Construction du dataset par genre

Pour chaque genre (10 genres × 50 fichiers), on extrait les features et on les stocke dans un DataFrame pandas, puis on sauvegarde en CSV.

### 3.3 Fusion des 10 genres

```python
df_fusionne = pd.concat([df_classical, df_reggae, df_jazz, df_blues,
                          df_hiphop, df_country, df_disco, df_pop,
                          df_rock, df_metal], ignore_index=True)
```

**Dataset final** : ~500 lignes × 27 colonnes (26 features + 1 colonne label)

### 3.4 Normalisation — StandardScaler

```python
from sklearn.preprocessing import StandardScaler

X = df.drop('label', axis=1)  # features
y = df['label']               # cible

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Pourquoi normaliser ?** L'algorithme KNN calcule des distances euclidiennes. Sans normalisation, les features avec de grandes valeurs (MFCC-0 ~ 200) dominent celles avec de petites valeurs (MFCC-12 ~ 2), rendant ces dernières inutiles. La standardisation (moyenne=0, écart-type=1) met toutes les features sur un pied d'égalité.

---

## Étape 4 — Modélisation Machine Learning (KNN)

### Algorithme K-Nearest Neighbors

Le KNN classe un nouveau point en identifiant ses K voisins les plus proches dans l'espace des features et en votant pour la classe majoritaire parmi ces voisins.

```python
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

**Split 80/20** : 80% des données pour l'entraînement, 20% pour le test.

### Graphique 1 — Matrice de Confusion

La matrice de confusion (heatmap seaborn) montre quels genres sont bien classifiés et lesquels sont confondus.

- **Diagonale** = bonnes prédictions
- **Hors-diagonale** = erreurs de classification

Les confusions les plus fréquentes attendues :
- Blues ↔ Jazz (structures musicales proches)
- Pop ↔ Disco (tempo et production similaires)
- Country ↔ Rock (instrumentation partagée)

### Graphique 2 — Courbe Accuracy vs K

Ce graphique montre l'accuracy du modèle en fonction du nombre de voisins K (de 1 à 10).

- **K=1** : surapprentissage (memorise les exemples d'entraînement)
- **K grand** : sous-apprentissage (trop peu discriminant)
- **K optimal** : généralement entre 3 et 7 sur GTZAN

**Performance attendue** : 60–70% d'accuracy sur 10 classes (à comparer avec 10% aléatoire).

---

## Étape 5 — Amélioration du modèle

### 5.1 Optimisation des hyperparamètres — GridSearchCV + Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': list(range(1, 21)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
```

**80 combinaisons** testées (20 × 2 × 2) par validation croisée à 5 plis.

L'utilisation d'un `Pipeline` est la **bonne pratique** car elle applique le scaling à l'intérieur de chaque fold de la cross-validation, évitant le **data leakage** (fuite d'information du test vers l'entraînement).

### 5.2 Features enrichies (MFCC + Δ + Δ² + Chroma)

En plus des 13 MFCC de base, on ajoute :

- **Δ-MFCC (delta)** : dérivée première → vitesse de variation du timbre
- **Δ²-MFCC (delta-delta)** : dérivée seconde → accélération du timbre
- **Chroma (12 demi-tons)** : profil harmonique, indique les notes et accords présents

```python
mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
delta  = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
```

**Total : (13+13+13+12) × 2 statistiques = 102 features par morceau**

**Performance attendue avec features enrichies** : 72–82% d'accuracy.

---

## Étape 6 — Deep Learning (CNN sur Spectrogrammes Mel)

### Concept : traiter l'audio comme une image

Au lieu d'extraire des features manuellement, on transforme chaque morceau en une **image 2D** (spectrogramme Mel) et on entraîne un CNN pour reconnaître les patterns visuels caractéristiques de chaque genre.

### Génération du Spectrogramme Mel

```python
mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)
```

- `n_mels=128` : 128 bandes de fréquences sur échelle Mel
- Taille finale de chaque image : `(128, T, 1)` où T ≈ 1 292 frames pour 30 secondes

### Architecture CNN

```
Input (128 × 1292 × 1)
    → Conv2D(32 filtres, 3×3) + ReLU + MaxPooling(2×2)
    → Conv2D(64 filtres, 3×3) + ReLU + MaxPooling(2×2)
    → Conv2D(128 filtres, 3×3) + ReLU + MaxPooling(2×2)
    → Flatten
    → Dense(128) + ReLU + Dropout(30%)
    → Dense(10) + Softmax
    → Prédiction du genre
```

| Couche | Rôle |
|---|---|
| Conv2D + ReLU | Détecte des patterns locaux (textures, formes) dans le spectrogramme |
| MaxPooling | Réduit la taille spatiale, conserve les patterns importants |
| Dropout(0.3) | Désactive 30% des neurones aléatoirement → évite le surapprentissage |
| Softmax | Convertit les scores en probabilités (somme = 1) sur 10 genres |

**Compilation** :
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Graphiques d'entraînement (20 epochs)

**Courbe Accuracy (train vs validation)** : montre la progression de l'apprentissage.
- Si les deux courbes convergent → bon modèle généralisable
- Si `train >> val` → overfitting (trop mémorisé les données d'entraînement)

**Courbe Loss (train vs validation)** : mesure l'erreur de prédiction.
- Loss décroissante = apprentissage correct
- Val loss qui remonte = signal d'overfitting → augmenter le Dropout ou réduire les epochs

**Performance attendue** : 80–88% d'accuracy — meilleure que le KNN classique.

---

## Résultats et performances

| Modèle | Features | Accuracy estimée | Remarque |
|---|---|---|---|
| KNN (K=5) | 26 features MFCC | 60–70% | Baseline, rapide |
| KNN optimisé (GridSearchCV) | 26 features | 65–75% | Meilleur K automatiquement trouvé |
| KNN + features enrichies | 102 features | 72–82% | MFCC + Δ + Δ² + Chroma |
| CNN (20 epochs) | Mel Spectrogram | 80–88% | Apprentissage automatique des features |
| CNN amélioré (50 epochs + augmentation) | Mel Spectrogram | 85–92% | Meilleur résultat possible |

> **Base aléatoire** : 10% (10 classes équilibrées) — tous les modèles améliorent significativement ce score.

---

## Bugs identifiés et corrections

| Cellule | Bug | Code incorrect | Correction |
|---|---|---|---|
| ZCR Classical | Variable `y` non définie | `num=len(y)` | `num=len(y_classical)` |
| ZCR Classical | Mauvaise variable dans la condition | `if y_reggae is None` | `if y_classical is None` |
| Spectral Centroid Classical | Durée calculée sur le mauvais signal | `len(y_reggae) / sr` | `len(y_classical) / sr` |
| F0 (tous genres) | Variable `f0_clean` écrasée | `f0_clean = ...` (3×) | Nommer `f0_clean_classical`, `f0_clean_reggae`, `f0_clean_jazz` |
| Normalisation | Scaler fitté avant le split | `scaler.fit_transform(X)` sur tout X | Faire le split d'abord, puis `fit` sur X_train uniquement |

---

## Recommandations et pistes d'amélioration

### Refactorisation du code

Le code actuel répète le même traitement pour chaque genre (classical, reggae, jazz...), ce qui génère ~70% de redondance. Une refactorisation avec des fonctions génériques réduirait drastiquement la taille du notebook.

```python
def extraire_mfcc_features(genre, max_files=100):
    dossier = f"gtzan/Data/genres_original/{genre}"
    features = []
    for fichier in os.listdir(dossier)[:max_files]:
        y, sr = librosa.load(os.path.join(dossier, fichier), sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        row = list(mfcc.mean(axis=1)) + list(mfcc.var(axis=1)) + [genre]
        features.append(row)
    return pd.DataFrame(features)

# Traitement de tous les genres en 3 lignes :
genres = ['classical','reggae','jazz','blues','country','disco','hiphop','metal','pop','rock']
df_final = pd.concat([extraire_mfcc_features(g) for g in genres], ignore_index=True)
```

### Enrichir les features

```python
# Ajouter Spectral Contrast (7 bandes) — discriminant pour metal vs classical
contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

# Ajouter Tonnetz (6 dim) — capture les relations harmoniques
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

# Tempo et beat (BPM) — très discriminant reggae/disco/hiphop
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
```

### Tester d'autres classifieurs

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# SVM (souvent le meilleur sur les features audio)
svm = SVC(kernel='rbf', C=10, gamma='scale')

# Random Forest (robuste, peu de tuning nécessaire)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
```

### Améliorer le CNN

```python
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks pour éviter l'overfitting
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)

# Augmenter les epochs avec arrêt automatique
history = model.fit(X_train, y_train, epochs=100, callbacks=[early_stop, reduce_lr])
```

### Utiliser tous les fichiers

Le code actuel limite à 50 fichiers par genre (`MAX_FILES=50`). GTZAN en contient 100 par genre. Passer à 100 peut améliorer l'accuracy de 5 à 10%.

---

## Comment exécuter le projet

### Prérequis

```bash
pip install librosa scipy matplotlib seaborn pandas numpy scikit-learn tensorflow soundfile kaggle
```

### Configuration Kaggle

1. Créer un compte sur [kaggle.com](https://www.kaggle.com)
2. Générer une clé API : `Account → API → Create New Token` → télécharge `kaggle.json`
3. Uploader `kaggle.json` dans le notebook (cellule de configuration)

### Exécution dans Google Colab (recommandé)

1. Ouvrir Google Colab : [colab.research.google.com](https://colab.research.google.com)
2. Uploader le notebook `.ipynb`
3. Exécuter les cellules dans l'ordre (Ctrl+F9 pour tout exécuter)
4. La première cellule installe les dépendances, la seconde télécharge le dataset

### Ordre d'exécution

```
1. Installation des packages
2. Configuration Kaggle + téléchargement dataset
3. Visualisation des waveforms (3 genres)
4. Extraction des features (F0, RMS, ZCR, Centroid, Bandwidth, Rolloff, MFCC)
5. Agrégation + construction des CSV par genre
6. Fusion + normalisation
7. Entraînement KNN → évaluation
8. KNN optimisé (GridSearchCV) → évaluation
9. CNN sur Mel Spectrogrammes → évaluation
```

---
