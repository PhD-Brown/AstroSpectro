# M3.1 — constats et propositions correctives

Refs #43. Base `b4018074269011e5fbec232df068b8d3326a4e8d` ; aucun correctif.
Les liens fichier:ligne ci-dessous visent cette base, pas les lignes des tests.
Les contre-exemples sont dans [proofs_m31_visualizer.py](proofs_m31_visualizer.py),
les sorties exactes dans [m31_experiments.json](m31_experiments.json).
**Ces textes sont des propositions ; aucune nouvelle issue n'est créée.**
Priorités proposées, à valider humainement ; aucune incidence chiffrée sur des
catalogues, modèles ou publications historiques n'est établie.

## Nouvelles manifestations dans le visualiseur : préparation/importance/SHAP — P1 proposé

- **Problème :** [visualizer.py:2394](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L2394)
  recherche `imputer/scaler/feature_selector`, alors que le classifieur produit
  `colfilter/prep/fs/pca/sampler/clf`. La vraie préparation SVM laisse ici
  `[5,30;1,NaN]` au lieu de `[sqrt(1.5),sqrt(1.5);-sqrt(1.5),0]`.
  Les noms des features sélectionnées sont remplacés par un préfixe arbitraire
  sans métadonnées (`2416`, `1508`). L'axe feature est réduit dans les valeurs
  SHAP3D (`2655`) ; pour4×3×2, seules alpha/beta sont retournées et gamma disparaît.
  Le calcul est exécuté deux fois (`2642`, `2647`). #27 décrit une classe de
  défauts comparable dans `master.py` et le notebook04, pas ces chemins du
  visualiseur. L'audit historique du4novembre déclare ce module non audité.
  Aucune cause racine commune entre ces implémentations distinctes n'est démontrée.
- **Pourquoi :** importances/métadonnées trompeuses dans ce chemin de visualisation.
- **Attendu :** utiliser les transformations réellement entraînées, noms issus de
  leur provenance, axes échantillons/features/classes identifiés ; une réduction
  déclarée valide ou un refus explicite ; calcul unique sans effet de bord inutile.
- **Vérification :** `actual_classifier_prep`, `selected_feature_names_derive`,
  `model_inspector_missing_selection_metadata`, `multiclass_shap`, `computed_once`.
  Moyennes absolues attendues `[5,37.5,375]` pour les trois features du cube contrôlé.
- **Hors scope :** attribuer ce défaut au notebook04, aux figures publiées ou aux
  modèles réels sans provenance ; corriger master/classifier/notebooks dans M3.1.

Extensions observées sur le même chemin : `explainer_mode` ignoré (`2633`) et
feature CSV absente remplacée par zéro (`2456`) avant l'imputation entraînée.
Le choix entre extension de #27 et ticket dédié pour ces nouvelles manifestations,
y compris `explainer_mode_is_respected` et `missing_measurement`, appartient à l'humain.
La provenance holdout du CSV le plus récent reste **NA** : ne pas interpréter
ces explications comme une évaluation indépendante du modèle.

## Nouveau chemin défectueux : conserver les indices de pixels après filtrage — P1 proposé

- **Problème :** [visualizer.py:332](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L332),
  `585`, `1916`, et consommateurs `1037`/`2490` : les indices émis par PeakDetector
  après suppression des non-finis sont utilisés sur les tableaux initiaux.
  Avec NaN au pixel0 et Inf au40, minimum au200 : marqueur rouge à4860.7761478 Å,
  flux0.5191959, au lieu de4861 Å/0.5. La preuve verte contrôle aussi le marqueur
  vert et les noms/positions de référence ; les trois routes sont reproduites.
- **Pourquoi :** associations de raies et mesures silencieusement déplacées.
- **Responsabilité démontrée :** `PeakDetector.detect_peaks` promet des indices
  dans les tableaux d'entrée (`peak_detector.py:112`), mais les renvoie après
  compactage (`126–131`). Les consommateurs du visualiseur réutilisent les indices
  sur les tableaux initiaux. Cette cause d'interface commune aux cinq consommateurs
  est reproduite ; `PeakDetector.analyze_spectrum` nettoie avant indexation et n'est
  pas affectée (test vert). Le rattachement à M4.2 reste distinct d'un correctif M3.1.
- **Attendu :** conserver le mapping vers les pixels d'origine, filtrer toutes
  les paires de façon cohérente, ou refuser explicitement les données invalides.
- **Vérification :** `nonfinite_prefix_keeps_original_peak_pixels` (3 routes) ;
  remède de mapping testé dans une copie, sans changer PeakDetector. Les deux
  consommateurs de features sont aussi exercés par `feature_consumers_keep_original_detected_wavelength`
  avec extraction réelle ; leur coordonnée de raie est vérifiée indépendamment.
- **Hors scope :** #30 (air/vide/repos et centres subpixel), audit intégral M4.2.

Pas de ticket spécifique trouvé dans l'inventaire ouvert/fermé ; proposition
distincte à valider. L'antériorité de ce défaut précis n'est pas déduite d'une similarité.

## Nouveau : centroïde de plan près de la coupure RA — gravité à arbitrer

- **Problème :** [visualizer.py:1259](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L1259),
  moyenne arithmétique RA359°/1°, Dec0° →180° au lieu d'une direction proche de0°.
- **Pourquoi :** plan placé dans la direction opposée sur la carte.
- **Attendu :** centroïde angulaire physiquement défini (moyenne circulaire ou
  vecteur sphérique, selon convention ratifiée), avec cas dégénérés explicites.
  La docstring annonce littéralement une moyenne en degrés, que respecte le code.
  Le sens physique de direction représentative est le contrat examiné, pas une
  convention déjà ratifiée. Pour RA0/180, une moyenne circulaire est indéfinie :
  A01 ne règle pas ce cas. P2 est une proposition défendable pour cette figure sans
  consommateur aval identifié ; la priorité finale reste humaine.
- **Vérification :** `ra_wrap_centroid_stays_near_neighboring_directions` ; même
  résultat attendu pour les deux conventions dans ce cas symétrique.
- **Hors scope :** imposer une convention universelle ou modifier les catalogues réels.

## Nouveau : coordonnées RA et graduations contradictoires — P1 proposé

- **Problème :** [visualizer.py:1339](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L1339)
  et `1389` : RA0° est dessinée au tick12h et RA180° au tick0h.
- **Pourquoi :** une carte numériquement plausible porte de mauvaises identités angulaires.
  Les points sont centrés12h, RA vers la gauche ; les graduations sont centrées0h,
  RA vers la droite : miroir autour de6h–18h. Une correction doit aussi rendre la
  docstring cohérente. Les deux conventions cohérentes sont admises par l'oracle.
- **Attendu :** sens et origine cohérents entre points et toutes les graduations,
  indépendamment du choix de méridien central admis et documenté.
- **Vérification :** `sky_ra_tick_labels_match_actual_transformed_points` vérifie
  les points réellement affichés, pas seulement une formule de projection.
- **Hors scope :** convention du fond image, qui n'a pas été validée ici.

## Nouveau : annoncer la population des pourcentages top N — P2 proposé

- **Problème :** [visualizer.py:3012](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L3012) :
  classes5/4/3/2/1, top3 ; barres41.667/33.333/25%, sans signaler les3/15 exemples exclus.
  Le second cas11/7/4/3/2, top2, distingue total27 et affichés18 (9exclus).
- **Pourquoi :** les pourcentages conditionnels peuvent être pris pour la population entière.
- **Attendu :** soit33.333/26.667/20% du total, soit pourcentages conditionnels
  accompagnés de la population affichée et des exclusions ; politique à ratifier.
- **Vérification :** `truncated_percentages_disclose_conditional_population` ;
  le dénominateur doit être associé à son rôle sémantique et calculé sur chaque
  population. Le texte « Top3, LAMOST DR5 (2015) » et le texte figé12/15 sont rejetés.
- **Hors scope :** défauts DimRed M2.8 (#111/R10), dont ce chemin est distinct.

## Nouveau : unités de normalisation et métriques notebook — P1/P2 proposés

- **Problème :** [visualizer.py:1709](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L1709)
  conserve le flux brut après refus/échec de normalisation, mais l'axe affirme
  « normalised units » (`1736`). Sigma brut `.5` associé au flux/6 (`726`), alors
  que sigma normalisé doit valoir1/12. À `742`, gaussian_fwhm reçoit le continuum :
  gaussienne σ=.8 Å →FWHM11.23 Å exportée au lieu de1.883856 Å.
- **Pourquoi :** unités et mesures physiques incohérentes. Le warning du
  préprocesseur signale le refus, mais ne rend pas le label de figure correct.
- **Attendu :** label/omission/refus explicite pour flux non normalisé ; propagation
  sigma avec le même facteur ; FWHM sur profil relatif au continuum adapté.
- **Vérification :** `failed_normalization_has_truthful_curve_units`,
  `normalized_flux_uncertainties_scale_with_flux`, `absorption_fwhm_export`.
  Preuve FWHM sur σ=.5/.8/1.6 Å et fenêtres±8/±12 Å. La borne utilise l'erreur
  de discrétisation d'ordre2, l'arrondiCSV et la troncature analytique gaussienne ;
  une constante1.883856 Å et un biais+10% sont rejetés. Les moments du profil
  d'absorption et les croisements interpolés à mi-hauteur sont admis sur ces fixtures.
  EW et positions sont désormais testées aux continuums1 et6.
- **Portée de sigma :** défaut latent P2 proposé. Les métriques CSV actuelles
  ne publient pas d'incertitude ; aucune erreur observable de ces valeurs n'est
  attribuée à sigma. L'oracle ne compte plus les constructeurs ; une incertitude
  attachée doit avoir les bonnes unités, ou peut être omise si non annoncée.
- **Autres routes :** un NaN annule la normalisation par médiane et laisse le
  continuum6 sous un label normalisé sur les trois analyseurs ; preuve séparée
  `nonfinite_normalization_refusal_has_truthful_spectrum_label`.
- **Hors scope :** réaudit FeatureEngineer, correction générale de continuum/EW #25 ;
  choisir le découpage des deux propositions physiques sans décision humaine.

## Nouveau : invalider un cache de positions après changement d'identités — P2

- **Problème :** [visualizer.py:1205](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L1205) :
  la date maximale ne détecte pas un nouveau FITS dont le timestamp est conservé.
  Deux spectres disponibles après refresh, compte en cache1.
  Suppression synthétique d'un FITS d'un second plan : le cache conserve aussi
  ce plan fantôme. `deleted_fits_invalidates_position_cache` reproduit ce cas.
- **Pourquoi :** le cache est présenté comme courant avec une population périmée.
- **Attendu :** provenance d'entrées suffisante pour invalidation, ou fraîcheur
  clairement non garantie ; `force_regenerate=True` fonctionne dans le test vert.
- **Vérification :** `added_backdated_fits_invalidates_position_cache` ; renoncer
  au cache est un remède admissible, sans imposer un format de manifeste.
- **Hors scope :** toucher aux caches/catalogues scientifiques du PC.

## Nouveau : ambiguïtés d'identité dans légendes/catalogues — P2

- **Problème :** [visualizer.py:1728](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L1728)
  donne la même légende à A/same et B/same ; `168` transforme des lignes F5/G2 pour
  la même identité en G2 sans avertissement.
- **Pourquoi :** courbes indiscernables et réattribution silencieuse de classe.
- **Attendu :** identité visible non ambiguë ; conflit rejeté ou signalé, sans
  imposer un ordre de préférence arbitraire entre classes.
- **Vérification :** `duplicate_basename_legend`, `duplicate_conflicting_catalog`.
- **Hors scope :** défaut de jointure DimRed #50/#53 ; aucune donnée réelle dédupliquée.

## Nouveau : style et état aléatoire globaux — P2

- **Problème :** [visualizer.py:338](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L338)
  applique `dark_background` à l'état global ; `1013` et `3103` reseedent `random`.
  Les preuves mesurent les clés réellement modifiées et l'état RNG.
- **Pourquoi :** cellules/figures/tirages ultérieurs affectés sans lien avec leurs entrées.
- **Attendu :** effet local/restauré ou API d'application globale volontaire.
- **Vérification :** `peak_plot_restores_global_matplotlib_style`, `sampling_preserves_global_random_state`.
- **Hors scope :** proposition P2 de façade M2.9 (ne pas la créer ici) ; effet propre
  du visualiseur démontré séparément, pas hérité d'une ancienne mesure.

## Nouveau : export PNG par basename — P2

- **Problème :** [visualizer.py:2865](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L2865)
  appelle makedirs sur le parent vide de `summary.png`, puis absorbe l'erreur.
- **Pourquoi :** un chemin de sortie valide n'est pas écrit ; l'avertissement est visible.
- **Attendu :** export dans le CWD pour basename, parents créés seulement si nécessaires.
- **Vérification :** `basename_export_is_written_in_current_directory` ; autres
  exports avec parent non vide sont vérifiés verts. Six routes sont maintenant
  rejouées (comparaison, barreSHAP, beeswarm, sous-classes, normalisation et barrefeatures) ;
  le défaut n'est plus seulement statique aux autres sites.
- **Hors scope :** réécriture de tous les exports, modification des workflows.

## EDA : non-finis et séparation d'objets — P2/P1 proposés

- **Problème :** [visualizer.py:3356](https://github.com/PhD-Brown/AstroSpectro/blob/b4018074269011e5fbec232df068b8d3326a4e8d/src/tools/visualizer.py#L3356)
  envoie Inf à l'histogramme (dropna ne l'exclut pas) ; erreur de plage sans nom
  de feature. À `3468`, split par ligne : des observations du même `source_id`
  se retrouvent des deux côtés, contaminant le holdout de permutation.
- **Pourquoi :** EDA interrompue et séparation scientifique insuffisante pour les
  observations répétées ; ne pas interpréter les importances comme validation indépendante.
- **Attendu :** exclusion documentée ou refus explicite des non-finis avec provenance ;
  séparation par objet lorsque les identités sont présentes, ou pas de prétention holdout indépendant.
- **Vérification :** `feature_explorer_nonfinite_histogram`, `does_not_split_repeat_observations`.
  L'oracle de partition observe les lignes réellement utilisées par le backend à
  fit/évaluation, sans espionner le splitter. Les groupes connus et masqués sont
  testés ; split groupé, refus explicite et annonce descriptive sans prétention
  d'indépendance sont admissibles. `source_id` est textuel et les classes alternent
  entre identités non ordonnées : ce fixture ne donne plus un ID numérique prédictif.
  Le cas6lignes/3classes a une erreur sklearn explicite admise par le test vert,
  **pas un nouveau défaut confirmé**. La sélection de colonnes par variance avant split
  reste un risque statique lié à #24, non une mesure de biais.
- **Hors scope :** #22/#24 du pipeline principal ; proposer une extension ciblée
  du même invariant au visualiseur, aucun run réel ni nouvel audit M4.
  Ce sont de nouvelles manifestations sur un autre chemin ; une cause racine
  commune avec les implémentations de #22/#24 n'est pas démontrée.

## Nouveau en revue R9 : sous-classe manquante — P2 proposé

- **Problème :** `visualizer.py:3022`, pandas3.1.1 : `[A,A,B,None,None,None]`
  provoque un `TypeError` backend dans `ax.bar`, sans diagnostic de label manquant.
- **Pourquoi :** figure de population interrompue sur des lignes sans sous-classe.
- **Attendu :** catégorie inconnue annoncée, exclusions comptées, ou refus explicite.
- **Vérification :** `missing_subclass_has_explicit_population_policy`, avec une
  variante qui exclut les lignes manquantes et publie leur nombre calculé.
- **Hors scope :** imputations scientifiques des classes ou modification des catalogues réels.

## Non-bugs et limites conservées

- Ledger mono-colonne `file_path` : correctement lu via fallback, confirmé par test.
- Offset dans sélection dédoublonnée, pas rang tracé : convention admise ici.
- Pourcentages top N : calcul conditionnel mathématiquement valide, disclosure insuffisante.
- Pas de diagnostic d'absence totale de bugs, de performance astrophysique,
  d'exactitude d'un catalogue ou d'une figure historique.
- Interfaces complètes, rapport HTML par clic, masques par bit, référentiel de
  repos, émission/continuum incliné et profils asymétriques : PA/NA dans la matrice.
