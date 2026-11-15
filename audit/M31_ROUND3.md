# M3.1 — corrections ciblées après deuxième revue Claude

Refs #43. Même branche `audit/43-tools-visualizer`, même Draft PR #118.
Base `b4018074269011e5fbec232df068b8d3326a4e8d`, HEAD de deuxième revue
`39423aa243dfc2b728fb0ff1821ba8590db39f39` ; entrées exécutables de cette passe :
**`de6dd9e1ae5b9f45efe09d125f0f600bbffa14f0`**. Seuls R1/R14/R16/R18 sont repris.

Sources : [revue 5913367599](https://github.com/PhD-Brown/AstroSpectro/pull/118#pullrequestreview-5913367599),
[contre-exemples](https://github.com/PhD-Brown/AstroSpectro/pull/118#issuecomment-6902762386),
[passation Claude](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6902792580),
[réconciliation corrective](https://github.com/PhD-Brown/AstroSpectro/issues/43#issuecomment-6901747695).
Worktree propre et synchronisé à la reprise ; aucun transfert depuis un autre
worktree. Les quatre manifestes historiques restent inchangés et protégés contre
l'écrasement par le checker. Les nouveaux rapports sont distincts :
[validation](m31_round3_experiments.json) et [campagne complète](m31_round3_adversarial.json).

## R14 — indépendance et cohérence des sorties

L'observation du RF à fit/predict reste indépendante de l'API du splitter. Les
identités connues de la fixture sont conservées séparément des features numériques.
Sans déclaration descriptive/refus, chaque partition réellement évaluée doit être
disjointe des groupes d'entraînement. Une déclaration ne court-circuite plus les
preuves de présentation : la docstring effectivement chargée, tous les objets
affichés, titres/axes/textes/légendes des artistes Matplotlib, tableaux retournés
et CSV réellement écrits sont inspectés.

La qualification « descriptive only » est admise sans exiger la phrase exacte
« not an independent holdout ». Un contre-essai au checkpoint intermédiaire
`b3f2460` montrait le rejet indu de cette formulation sans contradiction ; le
contrôle positif existant a été enrichi et A19 l'utilise désormais partout.
La campagne complète et la validation ont donc été réexécutées sur `de6dd9e` ;
les sorties intermédiaires restent dans `test-results/`, hors manifestes publiés.

La branche descriptive rejette les revendications concurrentes de validation ou
holdout, et exige une qualification sur chaque surface portant la métrique de
permutation. Un avertissement isolé ne qualifie pas un export, une figure ou une
documentation contradictoire. A19 conserve le calcul exploratoire par ligne mais
qualifie aussi sa docstring, le tableau retourné/affiché, le nom et le contenu du
CSV. Cela ne répare pas la partition et ne valide pas une estimation indépendante.
A18 partitionne réellement les groupes présents et refuse avant entraînement si
les IDs manquent. Aucun nom de bibliothèque ni signature de split n'est exigé.

`V_contradict` reproduit exactement le print isolé de Claude sur la production.
`V_contradict_mutant` remplace la séparation d'une copie A18 admissible par le
split par ligne et ce print : baseline non vide PASS, mutation exécutée, assertion
de présentation attendue. La mutation du split est écrite sur une seule ligne
exécutable : une parenthèse de fermeture sans événement de trace rendait le
prérejeu inconclusif ; ce prérejeu n'est pas utilisé comme manifeste définitif.
Les contrôles unitaires discriminent séparément titre, documentation, CSV et
revendication d'indépendance résiduelle.

## R18 — fichier, valeurs de courbe et légende

`assert_curve_identities` résout la légende vers un seul fichier et compare les
vrais `xdata`/`ydata` à la référence indépendante de ce fichier. L'ordre des artistes
peut changer. Un basename unique, un chemin relatif ou absolu non ambigu restent
admis ; le séparateur Windows est normalisé. Les références du chemin nominal
tiennent compte de la normalisation par médiane et des offsets documentés.

La preuve de collision utilise A=[1,2,3], B=[4,5,6] sous le même basename. A14
doit préserver cette association. `V_swapped_identity` garde des labels uniques
mais attribue chaque flux au fichier opposé ; il est comparé à A14, avec baseline
PASS et ligne mutée atteinte. Les contrôles verts admettent les chemins relatifs,
les basenames différents et un ordre de courbes inversé ; intervertir seulement
les légendes doit échouer sur les valeurs effectivement dessinées.

## R16 — même statistique dans les nombres, le nom et le graphique

Référence indépendante du cube SHAP 4 échantillons × 3 features × 2 classes :
moyennes absolues F=[4,25,250], G=[6,50,500]. L'oracle exige toutes les identités
et compare les valeurs par nom de feature, sans imposer leur ordre de classement.
Le nom de colonne définit une statistique explicite et les métadonnées doivent
être compatibles ; `attrs` ne peut plus rendre acceptable une somme nommée moyenne.
La preuve vérifie aussi le cube et ses noms conservés, puis appelle réellement
`plot_shap_summary_bar` : labels de features, largeurs des barres et qualification
de l'axe doivent correspondre au tableau. Les textes/légendes concurrents ne
peuvent annoncer l'autre statistique.

Trois contrats bornés restent admis : moyenne absolue sur échantillons/classes,
somme des classes moyennée sur les échantillons, résultats par classe identifiée.
Pour les panneaux par classe, leur ordre est libre si les identités sont exactes.
L'ancien A27 est conservé comme contrôle négatif. Sa nouvelle variante renomme la
colonne en `mean_sample_sum_class_abs_shap` et adapte le vrai rendu de la copie
temporaire ; le chemin binaire continue d'utiliser `mean_abs_shap`. Des mutants
permutent les noms, permutent l'axe feature et étiquettent la somme comme moyenne.
Les tests de l'oracle contrôlent aussi le conflit inverse, moyenne présentée comme
somme, et l'accord des sorties par classe avec leurs artistes.

## R1 — suivi existant, sans création d'issue

[FINDINGS](M31_FINDINGS.md#r1--suivi-correctif-réconcilié) contient les 19 lignes
mécanisme → #121–#139, vérifiées contre les titres/corps GitHub. La matrice et
le compte rendu de première revue contextualisent l'absence de création au moment
de la livraison Codex initiale. La réconciliation ultérieure est une étape de
suivi, pas une correction de production. #119 appartient à M2.8, #120 à M2.9.
Plusieurs manifestations d'un même mécanisme ne deviennent pas des tickets en
double ; notamment pixels ×5, cache ajout/retrait et exports ×6 restent regroupés.

## Validation et limites

Campagne globale **54 candidats**, exécutée entièrement sur `de6dd9e` :
**20 KILLED, 30 ADMITTED_BOUNDED, 3 TARGET_REJECTED, 1 REGRESSION**,
aucun SURVIVED/INCONCLUSIVE. Les 20 mutants ont une baseline PASS non vide,
la même collecte, toutes leurs lignes modifiées atteintes et un rejet par
AssertionError, sans ERROR. Les 30 admissions passent leur cible, les 59 verts
et le contrôle des autres preuves sans crash incident ni nouvelle régression.
Les SHA256 avant/après sont identiques dans les deux rapports définitifs.

| Contrôle discriminant | Verdict réellement observé |
|---|---|
| A18, groupes présents / refus sans IDs | 2 cibles PASS + 59 verts PASS |
| A19, descriptif cohérent sans formule de disclaimer imposée | 2 cibles PASS + 59 verts PASS |
| `V_contradict`, print seul sur production | TARGET_REJECTED, 2 assertions de documentation contradictoire ; 59 verts PASS ne suffisent pas à l'admettre |
| `V_contradict_mutant`, référence A18 | baseline 2 PASS → 2 FAIL par assertion pertinente ; mutation atteinte, KILLED |
| A14, identités relatives correctes | collision PASS + 59 verts PASS |
| `V_swapped_identity`, référence A14 | baseline PASS → FAIL sur `[1,2,3]` affiché sous le fichier de `[4,5,6]` ; mutation atteinte, KILLED |
| A05, moyenne SHAP correctement représentée | cible PASS + 59 verts PASS |
| Ancien A27, somme sous nom de moyenne | TARGET_REJECTED, métadonnées incompatibles avec le nom ; 59 verts PASS ne suffisent pas |
| A27 renommé et rendu adapté sur copie | cible PASS + 59 verts PASS ; ADMITTED_BOUNDED, widget non validé |
| Noms/axe de features permutés, somme sous axe de moyenne | 3 KILLED, chacun baseline PASS, ligne atteinte, assertion numérique ou d'axe pertinente |
| A08, noms faux malgré valeurs transformées | TARGET_REJECTED sur les 2 cibles, même avec 59 verts PASS |
| A15, doublons identiques sur-rejetés | REGRESSION, cible PASS mais seulement 58/59 verts ; jamais ADMITTED |

Exécution locale Windows / Python 3.11.9, environnement existant
`../AstroSpectro/.venv/Scripts/python.exe`, `PYTHONPATH=<worktree>/src`, Agg/CPU,
sockets bloqués par la configuration pytest ; aucune dépendance installée :

| Commande exécutée par `check_m31_visualizer.py --validation` | Résultat |
|---|---|
| `python -m pytest -q tests/test_tools_visualizer.py` | 59 PASS, 135 avertissements |
| Même sélection `--import-mode=importlib` | 59 PASS, 135 avertissements |
| Même sélection, ordre inverse des node IDs | 59 PASS, 91 avertissements |
| `python -m pytest -q tests -m 'integration and not slow and not network'` | 165 PASS, 709 exclus, 135 avertissements |
| `python -m pytest -q tests` | 798 PASS, 76 exclus, 142 avertissements, aucun FAIL/ERROR/SKIP |
| `python -m pytest -q audit/proofs_m31_visualizer.py` | 47 FAIL par AssertionError, 268 avertissements, aucun ERROR/SKIP/XFAIL |
| Black `--check src tests scripts app.py` + trois scripts M31 sous audit | PASS, 50 fichiers inchangés |
| Ruff, mêmes chemins ; `python -m pip check` ; `git diff --check` | PASS |

La validation est reproductible avec
`python audit/check_m31_visualizer.py --validation --output test-results/m31-round3-replay.json`.
La campagne complète utilise la même commande sans `--validation` (autre chemin
de sortie). Les deux rapports figés déclarent leur HEAD et les SHA256 avant/après.
Les avertissements restent visibles dans les journaux : dépréciations de palettes
SHAP/Matplotlib, moyennes de fenêtres spectrales vides sur les fixtures limitées,
et avertissements numériques des cas limites de la suite. Aucun avertissement
n'est transformé en validation scientifique.

Les quatre rouges ciblés échouent pour les raisons pertinentes : feature gamma
perdue dans SHAP, identités de légende ambiguës, objets `object-13/23/39/75` communs
aux partitions EDA avec et sans colonne d'IDs. Ils ne tombent pas sur une exception
incidente. Les 47 FAIL caractérisent des manifestations, pas 47 bugs distincts.

Arbre `src/` **`c9669447df76c2886a29f636fafc828beb6c53ff`**, blob visualizer
**`f70aee91b15a4ad242667b2717c0100be494e807`**, identiques à la base historique.
Le diff hors `tests/` et `audit/` est vide ; app, notebooks, dépendances, workflows,
configurations et artefacts scientifiques restent inchangés. Les quatre anciens
manifestes sont identiques au HEAD de deuxième revue.

Les résultats exécutés de la campagne définitive sont consignés dans les deux
nouveaux JSON, avec SHA des entrées, environnement, commandes, assertions et
statuts par candidat. La CI du HEAD publié sera vérifiée séparément et liée dans
la description de PR et la passation #43 ; les rapports locaux ne prétendent pas
attester une exécution GitHub ultérieure à leur écriture.

Les expériences sont des copies temporaires, jamais des correctifs complets
prêts à déployer. AP porte sur ces contrats U/I/P/S synthétiques ; PA demeure sur
les UI complètes, les autres backends/adaptateurs, formats de déclaration et
formes SHAP non exercés. Le widget SHAP et ses exports ne sont pas validés par
la nouvelle admission bornée du rendu réutilisable. L'inspection EDA ne fait pas
d'OCR des PNG : elle contrôle les artistes qui sont enregistrés et le contenu
des CSV. Les identités partiellement manquantes ne sont pas validées ici.

R15, CSV réels, modèles privés et sorties scientifiques historiques restent NA.
Le contre-exemple non bloquant de titre top N statique listant deux fixtures et
la limite de trace des suppressions pures restent consignés dans la deuxième
revue, hors de ces corrections ciblées. Les contrats R7/R13/R19/R20 ne sont pas
réaudités ; leur campagne est rejouée pour la cohérence du manifeste global.
Les conventions SHAP, centroïde RA/gravité, méridien/sens RA et dénominateur top N
restent à ratifier humainement. Les recommandations Claude ne sont pas des décisions.
La prochaine action est la troisième vérification Claude de R1/R14/R16/R18,
avant toute décision humaine de fusion ; huit fils laissés ouverts.
