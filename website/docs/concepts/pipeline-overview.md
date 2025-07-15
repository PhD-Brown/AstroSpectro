---
id: pipeline-overview
title: pipeline overview
sidebar_label: Pipeline overview
---

<!-- To be ompleted  -->

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faProjectDiagram } from '@fortawesome/free-solid-svg-icons'

# <FontAwesomeIcon icon={faProjectDiagram} /> Vue d'Ensemble du Pipeline

Cette page présente le flux de travail complet du projet **AstroSpectro**, de l'acquisition des données brutes à la génération des résultats de classification. Le pipeline est conçu comme une série d'étapes séquentielles et modulaires.

### Diagramme du Flux de Travail

> Le schéma ci-dessous illustre les grandes phases du pipeline. Chaque phase est mise en œuvre par un ou plusieurs modules spécifiques dans le code source.


<div style={{textAlign: 'center', backgroundColor: 'var(--ifm-background-color-secondary)', padding: '1rem', borderRadius: 'var(--ifm-card-border-radius)'}}>

```mermaid
---
config:
  layout: fixed
---
flowchart TD
 subgraph subGraph0["**Phase 1 : Acquisition**"]
        DL["**Étape 1** : Téléchargement<br><i>(dr5_downloader.py)</i>"]
        LOGS["Journal des téléchargements<br><i>(downloaded_plans.csv)</i>"]
  end
 subgraph subGraph1["**Phase 2 : Sélection du Lot**"]
        SEL@{ label: "**Étape 2** : Sélection d'un lot<br><i>(Dataset_Builder)</i>" }
        SELLOG["Journal des spectres utilisés<br><i>(trained_spectra.csv)</i>"]
  end
 subgraph subGraph2["**Phase 3 : Préparation**"]
        CATALOG["**Étape 3** : Génération du Catalogue<br><i>(generate_catalog_from_fits.py)</i>"]
        PREPROC["**Étape 4** : Prétraitement du lot<br><i>(Preprocessor)</i>"]
  end
 subgraph subGraph3["**Phase 4 : Extraction & Apprentissage**"]
        FEAT>"**Étape 5** : Extraction des Features<br><i>(PeakDetector &amp; FeatureEngineer)</i>"]
        TRAIN["**Étape 6** : Entraînement &amp; Validation<br><i>(Classifier)</i>"]
  end
 subgraph subGraph4["**Phase 5 : Finalisation**"]
        UPDLOG["**Étape 7** : Mise à jour du journal<br><i>(DatasetBuilder)</i>"]
        REPORTS["**Étape 8** : Sauvegarde des artefacts<br><i>(Rapports JSON, Modèles PKL)</i>"]
  end
    DL L_DL_LOGS_0@--> LOGS & SEL
    SEL L_SEL_SELLOG_0@--> SELLOG & CATALOG
    CATALOG L_CATALOG_PREPROC_0@--> PREPROC
    FEAT L_FEAT_TRAIN_0@--> TRAIN
    TRAIN L_TRAIN_UPDLOG_0@--> UPDLOG & REPORTS
    UPDLOG L_UPDLOG_SELLOG_0@--> SELLOG
    PREPROC L_PREPROC_subGraph3_0@--> subGraph3
    DL@{ shape: db}
    LOGS@{ shape: card}
    SEL@{ shape: rounded}
    SELLOG@{ shape: card}
    CATALOG@{ shape: procs}
    PREPROC@{ shape: h-cyl}
    TRAIN@{ shape: internal-storage}
    UPDLOG@{ shape: card}
    REPORTS@{ shape: rounded}
     DL:::actionStyle
     DL:::Peach
     DL:::Sky
     LOGS:::Ash
     SEL:::gray
     SELLOG:::Ash
     CATALOG:::gray
     PREPROC:::gray
     FEAT:::gray
     TRAIN:::gray
     UPDLOG:::gray
     REPORTS:::gray
    classDef actionStyle fill:#21262d,stroke:#3d8bff,color:#c9d1d9,rx:5,ry:5
    classDef Peach stroke-width:1px, stroke-dasharray:none, stroke:#FBB35A, fill:#FFEFDB, color:#8F632D
    classDef Sky stroke-width:1px, stroke-dasharray:none, stroke:#374D7C, fill:#E2EBFF, color:#374D7C
    classDef Ash stroke-width:1px, stroke-dasharray:none, stroke:#999999, fill:#EEEEEE, color:#000000
    classDef blue fill:#A4B8C4, stroke:#506070, color:#223040
    classDef gray fill:#6E8387, stroke:#49575a, color:#ffffff
    click DL "https://github.com/PhD-Brown/AstroSpectro/blob/main/src/tools/dr5_downloader.py"
    click SEL "https://github.com/PhD-Brown/AstroSpectro/blob/main/src/tools/dataset_builder.py"
    click CATALOG "https://github.com/PhD-Brown/AstroSpectro/blob/main/src/tools/generate_catalog_from_fits.py"
    click PREPROC "https://github.com/PhD-Brown/AstroSpectro/blob/main/src/pipeline/preprocessor.py"
    click FEAT "https://github.com/PhD-Brown/AstroSpectro/blob/main/src/pipeline/peak_detector.py"
    click TRAIN "https://github.com/PhD-Brown/AstroSpectro/blob/main/src/pipeline/classifier.py"
    linkStyle 1 stroke:#000000,fill:none
    L_DL_LOGS_0@{ animation: slow } 
    L_DL_SEL_0@{ animation: slow } 
    L_SEL_SELLOG_0@{ animation: slow } 
    L_SEL_CATALOG_0@{ animation: slow } 
    L_CATALOG_PREPROC_0@{ animation: slow } 
    L_FEAT_TRAIN_0@{ animation: slow } 
    L_TRAIN_UPDLOG_0@{ animation: slow } 
    L_TRAIN_REPORTS_0@{ animation: slow } 
    L_UPDLOG_SELLOG_0@{ animation: slow } 
    L_PREPROC_subGraph3_0@{ animation: slow }

```

</div>

### Description des Phases

<div className="card-demo">
<div className="card margin-bottom--md">
<div className="card__header"><h4>1. Acquisition</h4></div>
<div className="card__body">
Le point de départ. Le pipeline récupère automatiquement les spectres bruts depuis une source externe (LAMOST DR5) et les stocke localement. Il journalise les plans d'observation complétés pour éviter les téléchargements redondants.
</div>
</div>
<div className="card margin-bottom--md">
<div className="card__header"><h4>2. Sélection & Préparation</h4></div>
<div className="card__body">
À chaque exécution, un nouveau lot de travail est sélectionné parmi les spectres disponibles qui n'ont jamais été traités. Ses métadonnées sont extraites des en-têtes FITS et les spectres sont nettoyés (normalisation...). C'est une étape cruciale pour garantir la qualité des données.
</div>
</div>
<div className="card margin-bottom--md">
<div className="card__header"><h4>3. Modélisation</h4></div>
<div className="card__body">
Le cœur de l'analyse. Les spectres sont analysés pour en extraire des features physiquement pertinentes (présence de raies...). Ces features sont ensuite utilisées pour entraîner un modèle de Machine Learning supervisé.
</div>
</div>
<div className="card margin-bottom--md">
<div className="card__header"><h4>4. Finalisation</h4></div>
<div className="card__body">
Une fois le modèle entraîné, le pipeline conclut la session : il met à jour le journal des spectres traités, et génère un rapport de session ainsi que d'autres artefacts (modèle sauvegardé...) pour assurer la traçabilité.
</div>
</div>
</div>