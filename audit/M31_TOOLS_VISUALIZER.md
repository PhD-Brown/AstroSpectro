# M3.1 — AstroVisualizer : contrats scientifiques et techniques

Refs #43. Audit et tests seulement, révision après première revue Claude.
Voir [M31_REVIEW.md](M31_REVIEW.md) pour la seconde livraison et ses preuves.
Production inchangée ; aucune décision de correction ou de convention physique
n'est prise par cette PR. Aucun travail M3.2–M3.4 ou nouvel audit M2/M4.

## Provenance et méthode

- Base vérifiée après `git fetch origin main` :
  `b4018074269011e5fbec232df068b8d3326a4e8d` (merge humain #117).
- Blob `src/tools/visualizer.py` : `f70aee91b15a4ad242667b2717c0100be494e807`.
- Worktree : `C:/Users/alexb/Documents/Projets/AstroSpectro-m31` ; branche
  `audit/43-tools-visualizer`. Principal propre, laissé sur `db513004` ; aucun
  transfert de fichiers ni changement des autres worktrees.
- `AGENTS.md` v1.3 sur main, corps complet de #43, commentaires récents dont
  [passation 6898324768](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6898324768),
  #117, #21/#25/#27 et commentaire #27 lus. Le rapport historique déclare ce module
  non audité. Les indices et le SHAP sont de nouvelles manifestations dans ce module ;
  la similarité avec #27 dans master/notebook ne prouve ni antériorité ni cause commune.
  Le choix d'un rattachement correctif reste humain.
- Lecture intégrale du module (3 627 lignes), inventaire AST de ses **35 méthodes**,
  signatures, appels internes, handlers, appels I/O et effets de bord dans le
  champ `inventory` du [manifeste](m31_experiments.json).
- Interfaces lues : prétraitement FITS/normalisation, détection/association,
  extraction locale FeatureEngineer, construction `colfilter/prep/fs/pca/sampler/clf`,
  schéma du producteur de catalogues, ledger `file_path` de DatasetBuilder,
  `safe_sigma_from_invvar` et `latest_file`. Aucun de ces producteurs n'est modifié.
- Tests M2.8, helpers de revue et `tests/conftest.py` inspectés ; ils ciblent
  DimRedVisualizer. Les nouveaux oracles ne reprennent pas leur campagne KDE.

`AP` signifie preuve acquise uniquement sur le contrat borné indiqué ; `PA`
preuve partielle ; `NA` non validé ; `B` bloqué. Une ligne contenant une preuve
rouge reste PA pour la validation du comportement : elle démontre un défaut,
elle ne valide pas la méthode. Aucun taux de couverture de lignes n'est mesuré.

## Matrice des contrats

Tous les tests verts sont dans [tests/test_tools_visualizer.py](../tests/test_tools_visualizer.py),
les fixtures/oracles dans [m31_visualizer_contracts.py](../tests/m31_visualizer_contracts.py),
les violations dans [proofs_m31_visualizer.py](proofs_m31_visualizer.py).
Les identifiants ci-dessous sont des suffixes de noms `test_` exacts ou des
préfixes non ambigus ; les noms complets et résultats sont dans les manifestes.

| Famille / méthodes | Contrat, risque initial, type | Oracle et fixture | Preuve / état / limite |
|---|---|---|---|
| FITS : `__init__`, `_scan_for_spectra`, `refresh_available_spectra` | Inventaire trié et relatif ; modèle chargé à la demande ; identité élevée (U/I/S) | Arborescence temporaire avec deux basenames identiques, fichier ignoré | `scan_refresh_and_labels_preserve_nested_identity`, AP sur ces chemins ; symlinks/permissions NA |
| `_load_labels_catalog` | Classe attachée au chemin complet, pas au basename ; collisions élevées (U/I/S) | CSV `|`, clés Windows/POSIX, schéma absent, catalogue vide, doublon contradictoire | Tests `labels_invalid_schema`, `scan_refresh...` verts ; preuve `duplicate_conflicting_catalog...` rouge ; PA |
| `_format_header_line`, `_display_formatted_header`, `_display_formatted_header_for_streamlit`, `_display_header_for_streamlit` | Métadonnées, unité et erreur explicite (U/I) | Header synthétique unique, clé absente, fichier absent ; collecteur d'affichage | `header_units_identity_and_explicit_error`, AP borné ; rendu visuel UI et HTML hostile NA |
| `_plot_spectrum_analysis`, `_plot_peak_detection_notebook`, `_plot_spectrum_analysis_plotly` | Paires pixel/flux/λ, pics et raies appariées, axes/labels (I/P/S), élevé | Progression géométrique indépendante, gaussienne absorbante centrée au pixel 200 ; artistes rouges/verts et traces Plotly | `finite_gaussian_peak_identity_all_renderers` vert sur 3 routes ; `nonfinite_prefix...` rouge sur 3 routes ; PA |
| Même famille, erreurs | Pas de fausse détection pour flux constant/tout invalide ; absence/échec explicite (U/I) | FITS 1D/une seule ligne, gzip corrompu, aucun minimum, tout NaN | `no_peaks_and_all_invalid_plotly...`, `invalid_fits_dimensions...` verts ; tout-NaN encore tracé comme courbe, sans mesure ; PA |
| Notebook : métriques et exports | EW/FWHM/incertitudes, unités Å ; élevé (I/P) | EW analytique `A σ sqrt(2π)` ; FWHM `2 sqrt(2 ln2) σ` ; flux ×6, invvar=4 | EW/pics aux continuums1/6, exports CSV/PNG et présence de tabular LaTeX verts ; FWHM à3largeurs×2fenêtres rouge ; sigma latent P2 ; PA ; continuum incliné/émission générale non validés (#25) |
| `_plot_spectra_comparison`, `plot_normalization_comparison` | Médiane, invariance d'échelle, grilles propres, dédoublonnage, offsets (I/P/S) | `[2,4,6,8,10]` et ×3 ; grilles distinctes, fichier illisible entre deux valides | `comparison_scale_invariance_offsets_and_skip_rank`, `normalization_comparator...` verts ; offset = indice dans sélection dédoublonnée (trous admis et documentés) ; échec de normalisation et légendes ambiguës rouges ; PA |
| `analyze_feature_zeros` | Dénominateur = lignes traitées, zéros ≠ NaN, constantes, stats/export (U/I/P/S), élevé | Trois vecteurs contrôlés + un fichier absent ; moyenne 3, écart-type population sqrt(6) | `zero_statistics...` vert ; calcul de features remplacé seulement pour cet oracle statistique ; extraction réelle testée séparément ; RNG rouge ; AP borné/PA global |
| `_generate_position_catalog` | Domaines RA/Dec, comptes par plan, centroïdes et cache (I/P/S), élevé | RA30/50 →40 sur arc court ; RA390→30 ; Dec91 et RA NaN rejetés ; RA359/1 ; ajout antidaté et suppression | `position_catalog_validity_counts_cache_and_force` vert ; direction près de coupure et cache périmé rouges ; convention et gravité à arbitrer ; PA |
| `plot_sky_coverage` | Mollweide, sens RA, heures, Dec, taille/couleur (I/P/S), élevé | RA0/90/180 ; Dec -30/0/60 ; comptes1/2/4 ; points réellement tracés et ticks correspondants | `sky_artists...` vert sur Dec/comptes/tailles ; `sky_ra_tick...` rouge ; miroir6h–18h ; centre/sens libres si cohérents ; PA ; fond image non testé |
| `plot_subclass_distribution` | Comptes et dénominateur annoncé (I/P/S), élevé | Effectifs5/4/3/2/1 top3 et11/7/4/3/2 top2 ; labels manquants | `subclass_raw_counts...` vert ; pourcentages sans rôle de dénominateur et label manquant rouges ; total ou conditionnel déclaré admis, aucune politique ratifiée ; PA |
| `display_dataset_dashboard`, `_resolve_trained_log_path` | KPI/identités/sous-classes/plans (U/I/S), élevé | Ledger réel mono-colonne `file_path`, doublons Windows/POSIX, fichier absent de la collection | `dashboard_single_column_producer_ledger...` vert ; le fallback mono-colonne fonctionne malgré l'absence de `file_path` dans la liste de colonnes. Pas de bug KPI déclaré sur ce format ; ledger enrichi/chemins absolus PA |
| `_prepare_for_model` | Ordre, imputation/scaling/sélection appris, noms exacts (I/P/S), élevé | Vrais CollinearityFilter, prep et sélection optionnelle ; valeurs, noms, ordre et entrée à l'inférence réelle | `prepare_flat_pipeline...` vert ; `actual_classifier_prep...`, `selected_feature_names_derive...` rouges ; nouveau chemin apparenté à #27, rattachement ouvert ; PA ; sampler/PCA/calibration complets non validés |
| `_get_features_sample` | Source, cible exclue, ordre, échantillon reproductible, valeurs manquantes (I/P/S), élevé | CSV synthétique associé gamma=7alpha ; vrais FITS et FeatureEngineer | `features_csv_sampling...`, `real_raw_feature_sample...` verts ; feature manquante convertie en zéro rouge ; PA : aucune preuve que CSV choisi est un holdout |
| `_build_shap_explainer` | Modes tree/linear/kernel/auto et erreur inconnue (U/I) | Vrais RF/LogisticRegression et vrais explainers sur4 lignes/2features | `explainer_factory_real_small_models` vert ; dimensions/valeurs finies, pas validation complète d'additivité ; permutation et kmeans>1000 PA |
| `_run_shap_analysis` | Réduction sur échantillons/classes, pas sur features, noms, nombre d'appels, mode (I/P/S), élevé | Vraie `shap.Explanation` contrôlée 4×3 et4×3×2 ; références par classe ; chargement modèle remplacé | `binary_shap_mean...` vert ; axes3D/double appel/mode ignoré rouges ; nouvelles manifestations, similarité avec #27 sans cause commune établie ; PA |
| `_analyze_saved_model` | Ordre des classes, association importance/nom après sélection (U/I/S), élevé | Wrapper synthétique connu, importances.2/.8, classes M/F/G | `model_inspector_selected_names...` vert ; sans métadonnées sélection, préfixe inventé rouge ; nouvelle manifestation apparentée à #27, rattachement humain ; PA ; aucun pickle externe chargé |
| `plot_shap_summary_bar`, `_plot_feature_importances_bar` | Valeurs/noms associés, topN et exports (U/I/P) | Importances non interchangeables9/4/1 et.7/.2/.1, inspection barres et ticks | `binary_shap_mean...`, `feature_importance_bar...` verts ; export basename rouge ; PA |
| `plot_shap_beeswarm` | Explication réelle et fallbacks explicites (I) | Vraie Explanation2D puis erreurs contrôlées des deux sorties SHAP | `beeswarm_real_explanation_and_explicit_fallback` vert ; multisorties non validées (amont rouge), PA |
| `feature_explorer` | EDA/corrélation, RF, permutation, exports (I/P/S), élevé |24lignes, constante, NaN ; Pearson via sommes centrées indépendantes et médiane12 ; vraies bibliothèques | `feature_explorer_eda_real_rf_and_exports`, `feature_explorer_no_labels...` verts ; Inf et chevauchement d'objets entre partitions rouges ; PA ; pas de performance astrophysique ni split par objet validés |
| Effets de bord | Pas de pollution style/RNG de l'appelant (U/I) | Snapshot rcParams avant/après plot ; état random avant deux méthodes | `peak_plot_restores...` et `sampling_preserves...` rouges ; effets propres AstroVisualizer mesurés, distincts de M2.9 ; PA |
| `interactive_header_explorer`, `interactive_spectrum_analyzer`, `interactive_peak_tuner`, `interactive_model_inspector`, `interactive_spectra_comparator`, `interactive_shap_explainer`, `app` | Dispatch/états/callbacks | Lecture AST et code ; sorties métiers appelées directement dans les tests | PA statique ; cycles UI, rapport HTML via bouton et session Streamlit NA, réservés M3.2/M3.3. Pas de claim d'exécution de ces interfaces |

## Référentiels, unités et responsabilités

La [description officielle FITS LAMOST DR5 v3](https://dr5.lamost.org/v3/doc/data-production-description)
donne le sens de `COEFF0/COEFF1`, `VACUUM` et `HELIO`. Les tests démontrent la
reconstruction en Å à partir des coefficients ; **aucune correction repos** n'est
prouvée. Les headers synthétiques HELIO/VACUUM n'autorisent pas à appeler les
positions des transitions « au repos ». Les listes de référence et la propagation
des masques LAMOST restent liées à #30/#34 ; validité astrophysique NA.

La [documentation specutils gaussian_fwhm](https://docs.astropy.org/projects/specutils/en/latest/api/specutils.analysis.gaussian_fwhm.html)
demande une soustraction du continuum. Ici le notebook lui passe le flux avec
continuum : la FWHM rouge vient de ce chemin du visualiseur, alors que la mesure
FeatureEngineer du même profil utilise un profil d'absorption et passe le test.
L'EW unitaire passe ; cela ne résout pas #25 sur continuum local incliné et signe.
La métrique de centroïde sur flux avec continuum n'est pas validée pour des raies
asymétriques (PA). Le sigma brut est aussi associé au flux normalisé : violation
de propagation d'unités, latente P2 proposée : le test
`notebook_point_metric_csv_is_independent_of_latent_sigma` confirme des CSV identiques
pour sigma×1/6 et×100. Une absence d'incertitude attachée est admise si aucune n'est publiée.
Aucun impact sur un run réel n'est établi.

La [documentation TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
situe les sorties sur le dernier axe. L'oracle3D accepte des moyennes sur les classes,
des sommes sur les classes (moyennées sur les échantillons), ou des valeurs par classe,
lorsque la statistique et les classes sont déclarées. Les références indépendantes
sont les moyennes par classe [4,25,250] et [6,50,500]. La réduction de l'axe des features,
les permutations de noms et une normalisation non déclarée sont rejetées ; aucune
politique d'agrégation finale n'est ratifiée. Le contrôle de somme déclare son mode
dans les métadonnées de la table ; sa propagation dans les légendes/CSV UI reste PA.
Le mode2D ne prouve pas qu'un classifieur binaire fournit toujours une sortie2D.

Les wrappers de chargement de modèles/affichage sont remplacés aux frontières,
pas les courbes, statistiques, FITS, specutils, RF, preprocessing ou SHAP utilisés
dans les intégrations réelles. Les tests de réduction SHAP contrôlent ses valeurs
pour distinguer une erreur du visualiseur d'une erreur de bibliothèque.

## Validation et contre-essais

Résultats actuels de seconde livraison : [M31_REVIEW.md](M31_REVIEW.md),
[m31_review_experiments.json](m31_review_experiments.json),
[m31_review_adversarial.json](m31_review_adversarial.json).
Les références qui suivent décrivent la première livraison historique.

Commandes, sorties, versions, SHA d'entrée et SHA256 des fichiers effectivement
testés : [m31_experiments.json](m31_experiments.json).
Copies de source temporaires et résultats discriminants :
[m31_adversarial.json](m31_adversarial.json),
[checker rejouable](check_m31_visualizer.py).
Rejeu actuel par défaut vers `test-results/m31-review-replay.json` ; le checker
refuse d'écraser les deux JSON historiques. Résultats actuels dans M31_REVIEW.md.

Les tests rapides sont offline (`--disable-socket`), CPU/Agg, FITS/CSV/modèles
synthétiques dans `tmp_path` uniquement ; un `rc_context` et la restauration
RNG/builtins isolent leurs effets. Chaque test ferme ses figures. Les compteurs
FAIL sont des manifestations, pas des comptes de bugs. Pas de `xfail`, pas de
modification de production pour verdir la CI.

Les limites de la matrice restent valables après une CI verte.

## Propositions correctives, non créées

Voir [M31_FINDINGS.md](M31_FINDINGS.md). Recherche de doublons dans l'inventaire
ouvert/fermé et lecture de #21/#22/#25/#27/#30/#34 ; aucun ticket nouveau créé.
Les propositions sont regroupées par mécanisme ; le responsable scientifique
décide du regroupement, des priorités et des politiques avant création.

## Livraison initiale historique — HEAD2b4545b

**Section conservée pour provenance, remplacée pour la seconde revue par [M31_REVIEW.md](M31_REVIEW.md).**
Les anciennes étiquettes ADMITTED ne valent pas admission avec absence de régression :
A08 est non généralisable, A15 sur-rejette, A04 était codée en dur, A13 contourne le cache.

Entrées exécutables figées à **`8d320090ce6cab9f965bc53b5a0902c468dbf9f1`** ;
les commits suivants ne publient que documentation et manifestes. SHA256 avant/après
identiques dans les deux rapports. Python3.11.9 sous Windows, environnement existant
`../AstroSpectro/.venv/Scripts/python.exe`, dépendances verrouillées déjà installées,
aucune installation. Versions principales : NumPy2.4.6, pandas3.1.1, Astropy8.0.2,
specutils2.4.0, SciPy1.17.1, Matplotlib3.11.2, Plotly7.1.0, SHAP0.51.0,
sklearn1.9.1, pytest8.4.2, Black24.4.2, Ruff0.5.0.

Depuis le worktree, `PYTHONPATH=<worktree>/src`, `MPLBACKEND=Agg`, sockets bloqués :

| Commande réellement exécutée (préfixe Python ci-dessus) | Résultat final |
|---|---|
| `-m pytest -q tests/test_tools_visualizer.py` |33 PASS /0 FAIL/ERROR/SKIP, 69 warnings |
| même sélection, `--import-mode=importlib` |33 PASS, 69 warnings |
| même sélection, ordre des33 nodes inversé |33 PASS, 69 warnings |
| `-m pytest -q tests -m 'integration and not slow and not network'` |139 PASS /709 exclus, 69 warnings |
| `-m pytest -q tests` |772 PASS /76 exclus /0 FAIL/ERROR/SKIP, 76 warnings |
| `-m pytest -q audit/proofs_m31_visualizer.py` |26 FAIL /0 PASS/ERROR/SKIP/XFAIL, 85 warnings ; tous AssertionError de contrat |
| `-m black --check src tests scripts app.py audit/check_m31_visualizer.py audit/proofs_m31_visualizer.py` |49 fichiers inchangés |
| `-m ruff check` sur les mêmes chemins |PASS |
| `-m pip check` ; `git diff --check` |PASS |

Commandes orchestrées et sorties complètes dans `checks` du manifeste ; les noms
de fichiers JUnit sous `test-results/` sont locaux, leurs résumés/failures sont
conservés dans le JSON versionné. Les69 warnings ciblés viennent de la dépréciation
SHAP/Matplotlib et des bandes hors du domaine étroit des FITS synthétiques dans
FeatureEngineer (moyennes vides, degrés de liberté/division). Les7 avertissements
supplémentaires de la suite complète concernent Gaia/pandas déjà suivis. Les85
warnings rouges comprennent aussi les entrées volontairement non finies et les
frontières sklearn ; ils ne sont pas des ERROR. Aucun test slow/network n'est
présenté comme exécuté par cette campagne ;76 tests sont exclus par défaut.

`python audit/check_m31_visualizer.py` : **8 mutants KILLED**,0 survivant/0 non
concluant dans la campagne finale, avec contrôles originaux PASS et mêmes nombres
de tests collectés. Leurs échecs sont des assertions numériques/sémantiques,
pas des crashs de dépendances. **16 alternatives ADMITTED /18 exécutions PASS**,
dont mapping des pixels dans les3 renderers, deux normalisations statistiques,
RA circulaire, ticks cohérents, sélection par support, vraie préparation, axes SHAP,
profil d'absorption, sigma normalisé, cache régénéré, identités explicites et style local.
Il s'agit de remèdes bornés aux fixtures, pas de correctifs prêts pour production.
La variante de disclosure utilise volontairement les effectifs de cette fixture ;
une généralisation doit calculer ces effectifs. Le remède FWHM suppose ici continuum1.

Essais de développement non comptés dans les totaux finaux : première tolérance
de reconstruction trop serrée pour la progression géométrique401pixels ; premier
oracle Pearson oubliant l'imputation médiane ; profondeur de raie inférieure au
seuil du détecteur brut. Tous corrigés par référence analytique/contrat. Le premier
parseur JUnit omettait les assertions sans attribut XML `type` ; corrigé et campagne
rejouée. Un candidat de permutation DataFrame provoquait un ValueError sklearn :
non compté comme kill, remplacé par une permutation après imputation qui atteint
l'assertion numérique. Aucun de ces essais ne constitue un défaut de production.

Les quatre fichiers exécutables sont ajoutés sous `tests/` et `audit/` ; documents
et JSON complètent la livraison. Diff des chemins protégés vide ; blob visualizer
identique à la base. Aucun FITS, catalogue, modèle, notebook, configuration ou
workflow réel modifié. Aucun pickle non fiable chargé, aucun service/GPU contacté.

La CI officielle, le HEAD distant final et le merge de contrôle sont consignés
dans la description finale de la Draft PR et la passation #43, après lecture des
logs. Les preuves rouges et mutations sont exécutées localement hors CI officielle.
**Arrêt après publication de cette passation : première revue Claude encore à faire.**
