# M3.1 — corrections après première revue Claude

**Compte rendu historique de la livraison `39423aa`. Les limites R1/R14/R16/R18
identifiées à la deuxième revue sont corrigées et revérifiées séparément dans
[M31_ROUND3.md](M31_ROUND3.md). Les admissions A19/A27 ci-dessous décrivent l'ancien
oracle ; elles ne constituent plus le verdict actuel.**

Suivi réconcilié depuis cette livraison : [19 mécanismes → #121–#139](M31_FINDINGS.md#r1--suivi-correctif-réconcilié).
« Aucune issue créée » ci-dessous décrit l'action de Codex à cette livraison ;
la coordination ultérieure a ouvert les tickets. #119/#120 relèvent de M2.8/M2.9.

Refs #43. Même branche `audit/43-tools-visualizer`, même Draft PR#118.
Revue d'entrée : [5912550396](https://github.com/PhD-Brown/AstroSpectro/pull/118#pullrequestreview-5912550396),
[annexe adversariale](https://github.com/PhD-Brown/AstroSpectro/pull/118#issuecomment-6900238945),
[passation Claude](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6900280554).
Base `b4018074269011e5fbec232df068b8d3326a4e8d`, HEAD revu `2b4545bfb9ef2b15221dc3bd37cf4fbd92546c78`.
Entrées exécutables révisées figées à **`2ca692960da69c7118ebffb5e9ea9dcea7ba1824`**.

Les anciens JSON `m31_experiments.json` et `m31_adversarial.json` restent identiques
octet pour octet. Leurs résultats et labels sont historiques, pas les conclusions
de cette seconde livraison. Source/proofs/checker initiaux restent reconstructibles
au commit distant `2b4545b`, ancêtre conservé ; aucune branche/tag/archive supplémentaire
n'est nécessaire. Les nouvelles sorties sont distinctes :
[validation](m31_review_experiments.json), [matrice des candidats](m31_review_adversarial.json).

## Corrections demandées

| Revue | Correction et contrat actuel | Preuve / limite |
|---|---|---|
| R1 (deux fils) | Nouvelles manifestations dans visualizer ; #27 décrit d'autres chemins ; cause commune entre ces implémentations non démontrée ; rattachement humain | [FINDINGS](M31_FINDINGS.md), matrice corrigée ; rapport historique `AUDIT_2026-11-04.md:1061` et corps actuel de #27 relus |
| R7 | σ=.5/.8/1.6Å, fenêtres±8/±12Å ; référence analytique, erreur de discrétisation d'ordre2, troncature et arrondi | Six preuves `absorption_fwhm_export...` ; constante et biais+10% rejetés contre baseline physique PASS ; moments et croisements interpolés admis sur ces profils |
| R13 | Vrais CollinearityFilter et `_build_preprocessor`, sélection optionnelle, entrée à l'inférence réelle, noms/ordre/valeurs simultanés | `preparation_case`, deux preuves `actual_classifier_prep...` ; alpha_copy réellement supprimée, puis fs conserve gamma/delta ; ancienne A08 non généralisable |
| R14 | Observer les groupes dans les lignes fit/évaluation, sans imposer de splitter ; identité textuelle connue ou masquée | Deux preuves `repeat_observations...` ; GroupShuffleSplit correct, refus explicite sans IDs et absence annoncée d'indépendance admis ; adaptateur borné au backend RF existant, pas à l'API de split |
| R18 | Identité unique de chaque courbe : basename unique ou chemin relatif | `assert_curve_identities` dans le test vert ; collision dans la preuve rouge ; A14 doit passer toute la suite verte |
| R19 | Candidats × suite verte entière × toutes les preuves rouges ; cible non vide vérifiée séparément ; compte exact des sites ; mutations tracées | Checker révisé et `m31_review_candidates.py` ; baseline PASS et assertion pertinente requises pour un KILLED ; A15 doit être classée régression, pas admission |
| R20 | Deux populations distinctes, dénominateurs associés à leur rôle total/affiché/exclu | Cas15/12/3 et27/18/9 ; CM4 «2015», texte figé12/15 et rôles inversés rejetés ; deux conventions calculées et annoncées admises |

## Oracles et admissibilité

**FWHM.** Pas logarithmique `1e-5`, pixel maximal proche de `.1125Å` ; sigma le
plus étroit vaut environ4.4pixels. Pour chaque fixture, la tolérance vaut
`0.005 + FWHM*(h/sigma)**2/6 + 2*tail`, où `h` est le pas maximal et `tail`
la perte analytique du second moment d'une normale tronquée à `±window/sigma`.
Elle est approximativement `.015/.011/.008Å` selon sigma, bien inférieure à10%.
Le terme `.005` est la demi-unité d'arrondiCSV à deux décimales ; le terme en h²
est une enveloppe conservative pour moments/croisements interpolés sur ces profils
lisses, et non une incertitude instrumentale. Aucun continuum général ni bruit
astrophysique n'est validé. La variante de moments suppose ici un continuum1 connu ;
l'autre utilise les croisements à mi-hauteur, pas la même fonction de production.

**Préparation.** Le train contient `alpha`, sa copie×2, `gamma`, `delta`. Le vrai
filtre de colinéarité garde alpha/gamma/delta ; le vrai prep ajuste imputation
médiane et scaling sur le train. La sélection connue garde gamma/delta. Les
valeurs attendues sont des moyennes/variances et médianes calculables directement,
corroborées par les valeurs observées à l'appel réel de `LinearRegression.predict`
dans le pipeline synthétique, avant tout appel au visualiseur. La requête est
permutée, contient une colonne superflue et une valeur manquante. Les noms ne sont
jamais dérivés d'un préfixe tronqué. Ce test n'est pas un oracle de performance ML.
Samplers/PCA/calibration et modèles sauvegardés réels restent PA/NA.

**R17, composition.** `missing_csv_features_follow_trained_preparation_at_explainer`
part d'un vrai CSV auquel manquent gamma/delta et observe les données à l'explainer.
Préserver NaN (A10) seul ne suffit pas : le prep doit appliquer les médianes entraînées
et la sélection. Gamma s'impute à sa moyenne35 ; delta s'impute à sa médiane.35,
différente de sa moyenne.433333 : la valeur standardisée attendue est−.2966954145.
La variante jointe démontre cette composition ; aucune prétention d'indépendance
de deux réparations séparées n'est faite.

**R14/R15.** Les identités sont des chaînes non ordonnées, les classes alternent
entre objets ; aucun source_id numérique parfaitement prédictif n'est injecté.
Le tag alpha est un marqueur synthétique de provenance des rangées, sans prétention
physique. Le test observe les données réellement utilisées par le RF existant à
fit/predict, y compris pendant l'évaluation par permutation. L'adaptateur ne dépend
ni de `train_test_split`, ni de la bibliothèque choisie pour fabriquer les partitions.
Il n'est pas annoncé indépendant du backend de modèle : un remplacement de ce
backend requerrait un adaptateur de provenance approprié. Une annonce explicite
« descriptif, pas de holdout indépendant, observations répétées » est admise ; elle
ne prouve pas que les partitions ont été réparées. Les CSV/modèles réels restent NA.
Les identités partiellement manquantes ne sont pas couvertes par ces deux fixtures.

**R16.** Les axes et identités sont contractuels ; la normalisation sur les classes
reste une décision scientifique. Références par classe F/G : `[4,25,250]` et
`[6,50,500]`. Moyenne, somme des classes moyennée sur les échantillons, ou sorties
par classe déclarées sont acceptées. Les tests unitaires couvrent les trois ; la
variante de somme déclare son mode dans les métadonnées de la table. Propagation
de ces métadonnées aux figures/CSV UI non validée, donc remède de contrat borné,
pas correction générale prête à déployer. Pondérations de classes arbitraires NA.

**R12/R8.** Le cas symétrique359°/1° distingue la direction représentative de sa
moyenne arithmétique littérale, sans choisir cercle ou sphère. Le cas antipodal
0°/180° n'a pas de moyenne circulaire définie ; la variante A01 ne le résout pas.
Gravité finale et convention à ratifier (P2 proposé pour cette figure). Pour la
carte, centre0h/12h et sensRA sont libres s'ils restent cohérents ; les deux variantes
correctes sont testées. La variante centrée12h corrige aussi la docstring de sa copie.
L'adaptateur actuel vérifie des points coïncidant avec les ticks de ces deux grilles ;
les autres calages de méridien/graduations restent PA. Un rejet hors de ce domaine
ne constituerait pas une décision scientifique sur une convention alternative.

**R20.** Le parseur lie des nombres entiers bornés à leurs rôles et des énoncés de
dénominateur ; «15» ne peut pas venir de l'année2015. Les fixtures sont5/4/3/2/1,
top3, et11/7/4/3/2, top2. Un total ou un affiché/exclu mal associé est rejeté,
même avec des hauteurs plausibles. La convention totale et la convention conditionnelle
calculent leur texte sur les effectifs courants ; aucune politique finale imposée.

## Observations complémentaires examinées

| Revue | Résultat actuel / réserve |
|---|---|
| R2/R3 | Incohérence producteur/consommateurs démontrée. `analyze_spectrum` nettoie avant indexation : test vert. Deux preuves des consommateurs de features utilisent l'extracteur réel et vérifient la coordonnée Hβ originale. Le mapping réparé couvre5consommateurs dans une copie ; M4.2 reste séparé |
| R4/R6 | Pics et EW aux continuums1/6 ; CM1 omettant la normalisation est rejeté. Trois nouvelles preuves détectent le label normalisé après refus dû à un NaN. Ne pas confondre la normalisation avec la perte d'identité de pixel |
| R5 | Sigma mal associé est latent P2. Le test vert compare les octetsCSV actuels pour sigma×1/6 et×100. La preuve n'impose plus un nombre de constructeurs ; absence d'incertitude attachée admise si rien n'est annoncé |
| R8 | Points et graduations se réfléchissent autour de6h–18h ; docstring/convention à rendre cohérentes dans un futur correctif |
| R9 | Sous-classe manquante : TypeError backend reproduit, preuve rouge conservée ; catégorie inconnue, exclusions comptées ou refus qualifié admis |
| R10 | Suppression d'un FITS temporaire reproduite : plan fantôme ; A13 contourne le cache, ne valide pas une stratégie d'invalidation |
| R11 | Six exports par basename exercés ; avertissement visible et fichier absent, preuve sur chaque route ; garde de parent vide contrôlée aux6sites |
| R15 | Identifiant numérique dans un CSV réel non vérifié ; le mécanisme `select_dtypes` demeure un risque statique documenté, sans quantification rétrospective |
| R17 | NaN et vraie préparation testés conjointement ; voir l'oracle à l'explainer ci-dessus |
| R21 | A08, A15, ancien texte A04 et cache A13 distingués des remèdes généraux. Un doublon identique ne doit pas effacer des identités sans lien (nouveau test vert). Aucun candidat n'est présenté comme correctif de production complet |
| Exports | CSV métriques comparé numériquement, PNG vérifié ; LaTeX vérifié seulement par présence de `tabular`, pas compilation/lecture scientifique exhaustive ; rapportHTML/UI complète NA |

## Harnais et classification

Chaque mutation remplace un site nommé dans une méthode identifiée par AST ; les
comptes sont vérifiés. M01 touche explicitement2sites ; le mapping des consommateurs
en touche5, la garde d'export6. Les sites et méthodes modifiés, hashes et lignes
réellement atteintes sont consignés. Aucune mutation globale supplémentaire silencieuse.
Les candidats sont compilés et chargés depuis des copies temporaires, jamais écrits
dans src. Deux processus pytest indépendants peuvent tourner simultanément ; un seul
agent édite la branche.

Un mutant exige une baseline PASS non vide, la même collecte, des lignes modifiées
atteintes et une assertion pertinente (pas un ValueError de dépendance). Pour les
mutants FWHM/textuels, la baseline est un remède physique/statistique correct et non
la production rouge : les hashes des deux sources sont enregistrés. Chaque variante
corrective passe la suite verte entière, toutes les preuves et la cible résolue
par pytest `-k` (pas une recherche approximative de nom).

- `ADMITTED_BOUNDED` : ciblePASS et vertsPASS, sans nouveau crash/régression observable ;
  uniquement le contrat et les fixtures annoncés, jamais validation de production.
- `REGRESSION` : ciblePASS, mais un contrat vert valide casse.
- `TARGET_REJECTED` : le contre-exemple ciblé reste en échec ; préciser le défaut de portée.
- `INCONCLUSIVE` : crash incident, collecte/atteinte non établie ou autre manque de preuve.
- `KILLED`/`SURVIVED` : qualification des mutants uniquement, avec contrôle d'atteinte.

Les autres preuves rouges restent ouvertes ; leur exécution ne les transforme pas
en tests verts. Les limites de conventions et les cas non validés restent PA/NA.
Le checker refuse d'écraser les deux manifestes historiques. Un rejeu partiel est
explicitement indiqué par `--only` et ne remplace jamais la campagne complète publiée.

## Validation locale de cette révision

Windows/Python3.11.9, environnement existant `../AstroSpectro/.venv/Scripts/python.exe`,
Agg/CPU, `PYTHONPATH=<worktree>/src`, sockets bloqués. Versions exactes et commandes
dans [m31_review_experiments.json](m31_review_experiments.json). Aucun outil installé.

| Commande | Résultat |
|---|---|
| `-m pytest -q tests/test_tools_visualizer.py` |50PASS,0FAIL/ERROR/SKIP,135warnings |
| même sélection `--import-mode=importlib` |50PASS,135warnings |
| ordre inverse des50nodes |50PASS,91warnings (compte réellement observé) |
| intégration `-m 'integration and not slow and not network'` |156PASS,709exclus,135warnings |
| suite complète `-m pytest -q tests` |789PASS,76exclus,142warnings,0FAIL/ERROR/SKIP |
| preuves explicites `-m pytest -q audit/proofs_m31_visualizer.py` |47FAIL,0PASS/ERROR/SKIP/XFAIL,268warnings ; assertions de contrat |
| Black `--check src tests scripts app.py` + les3scriptsM31 audit |50fichiers inchangés |
| Ruff sur les mêmes chemins, pip check, git diff --check |PASS |

Warnings conservés : dépréciations SHAP/Matplotlib, bandes non couvertes par les
FITS étroits, moyennes vides/divisions et entrées invalides intentionnelles, plus
les7warnings Gaia/pandas de la suite générale. Le nombre en ordre inverse est
rapporté séparément ; il ne constitue pas un contrat scientifique. Les76tests
slow/network exclus ne sont pas présentés comme exécutés.

Les essais de développement ont corrigé une première attente d'imputation médiane
prise à tort pour une moyenne pour delta ; l'oracle actuel distingue les deux.
Un rapport exploratoire partiel a aussi refusé sa garde d'immutabilité après un
ajout de test en cours de rejeu : non retenu comme validation. Les rapports publiés
requièrent `inputs_unchanged=true` sur les octets figés.
Le premier croisement complet a révélé un appel de schéma interne à FeatureEngineer
qui précédait l'extraction du FITS audité. La preuve des consommateurs a été corrigée
pour identifier ses entrées par la grille de longueurs d'onde, sans imposer un nombre
d'appels ni utiliser leur ordre. Les nouveaux rapports rejouent la campagne après
cette correction d'oracle ; les données de production restent inchangées.

## Livraison et seconde revue

Campagne complète des47candidats : **15KILLED,30ADMITTED_BOUNDED,1TARGET_REJECTED,
1REGRESSION,0SURVIVED/INCONCLUSIVE**. Les15mutants ont une baseline PASS, une
collecte identique, leurs lignes modifiées atteintes et un rejet par assertion.
Les30variantes admises passent chacune la suite verte de50tests et leurs cibles ;
toutes les47preuves sont aussi rejouées par variante, avec les autres défauts ouverts.
A08 échoue sur le vrai pipeline ; A15 passe le refus de conflit mais casse le test
de doublons identiques (49/50verts) : aucune admission annoncée pour ces deux candidats.

| Cas sensible | Résultat de contre-essai vérifié |
|---|---|
| CM1 | Omission de normalisation rejetée sur continuum6, baseline PASS |
| CM2/CM3 | Constante et biais+10% rejetés ; baseline physique6/6PASS |
| A12/A26 | Moments et croisements interpolés :6ciblesPASS et50vertsPASS chacun |
| A08/A17 | Ancien prep seul rejeté ; préparation correcte :3ciblesPASS (dont2avec vrai pipeline),50vertsPASS |
| A17 joint | NaN préservés puis médianes/scaling/sélection :4ciblesPASS,50vertsPASS |
| A18/A19 | Split groupé/refus sans IDs et annonce descriptive :2ciblesPASS chacun,50vertsPASS |
| A14/A15 | Chemin relatif accepté avec50vertsPASS ; refus global de doublons = régression |
| CM4/CM4b | Année2015 et texte figé rejetés contre baseline dynamique2/2PASS |
| A03/A04 | Total déclaré ou conditionnel calculé :2ciblesPASS et50vertsPASS chacun |
| A16 | Trois renderers et deux consommateurs de features :5ciblesPASS,50vertsPASS |
| A21 | Six exports par basename :6ciblesPASS,50vertsPASS |

La CI sur le nouveau HEAD, le merge de contrôle et ses logs sont consignés dans
la description finale de la PR et la passation#43 après lecture, sans ajouter une
autre modification exécutable à ce checkpoint. Les47preuves et mutations sont horsCI.
Production, app, notebooks, workflows, dépendances et données protégés : diff vide ;
arbre src `c9669447df76c2886a29f636fafc828beb6c53ff`, blob visualizer
`f70aee91b15a4ad242667b2717c0100be494e807`, identiques à la base.
Les15fils restent ouverts pour seconde revue indépendante, aucune issue créée,
aucune fusion ni mission M3.2. Prochaine étape : seconde revue Claude, puis décisions humaines.
