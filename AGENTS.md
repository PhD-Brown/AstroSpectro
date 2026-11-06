# AGENTS.md — Protocole de travail AstroSpectro (humain + IA)

> Guide opérationnel commun aux assistants (v1.2 — fusion GPT + Claude). Les instructions de projets ChatGPT/Claude et `CLAUDE.md` renvoient à ce fichier et ne dupliquent pas ses règles.
> Lecture directe, sans connecteur : `https://raw.githubusercontent.com/PhD-Brown/AstroSpectro/main/AGENTS.md`

## 0. Principe directeur

**Une tâche = une issue (sauf correction évidente), un espace de travail identifié, une validation vérifiable et un historique GitHub.** Ne pas créer un second outil de suivi. Toute information essentielle pour reprendre une tâche doit exister sur GitHub, pas uniquement dans un chat.

## 1. Responsabilités

- **Responsable scientifique (humain)** : priorités, validation scientifique, contrôle des modifications, décision de fusionner les PR.
- **Assistant IA** : recherche de contexte, proposition de méthode, rédaction de code/tests, aide à l'organisation et au compte rendu, selon les accès disponibles.
- **GitHub Actions (CI)** : vérifications automatisées configurées. **CI verte ≠ validation scientifique générale**.
- **Un seul agent éditeur sur une branche à la fois**. Un autre assistant peut faire une revue sur le même périmètre sans écrire simultanément.

## 2. Où chercher l'information

| Question | Référence | Précautions |
| --- | --- | --- |
| Quelle est la méthode de travail ? | Ce fichier, sur `main` | Vérifier la version actuelle avant de commencer. |
| Que faut-il faire et pourquoi ? | GitHub Issue, **tous** ses commentaires, PR associée | L'issue est le relais entre séances et IA. |
| Quel est le code accepté ? | `PhD-Brown/AstroSpectro`, branche `main` | `main` est l'état officiel intégré, pas forcément le dernier travail en cours. |
| Quel code est en cours de modification ? | Branche liée à l'issue ; répertoire local si accessible | Vérifier `git status`, branche et divergence avant d'écrire. Ne jamais écraser du travail non commité. |
| Quels résultats sont publiés ? | `PhD-Brown/research-docs` : `FACTS_INDEX.md`, `_facts/`, pages | Les chiffres et réserves doivent être vérifiés dans leurs artefacts sources. |
| Où sont les runs, catalogues et modèles non versionnés ? | PC de l'auteur / espace Drive connecté, si accessible ; W&B pour les runs consignés | Confirmer chemin, date, ID de run, hash et niveau de synchronisation. Ne pas supposer que Drive est le PC en temps réel. |

Les fichiers téléversés dans les projets ChatGPT et Claude sont des **instantanés historiques**, jamais une source prioritaire du code actuel. La mémoire conversationnelle sert au contexte humain, non à attester de l'état du dépôt.

En cas de contradiction : ne pas arbitrer de mémoire. Lire les artefacts pertinents, signaler la divergence et proposer un correctif tracé.

## 3. Démarrer une issue

L'utilisateur dit : « On travaille sur `PhD-Brown/AstroSpectro#N`. »

1. Lire `AGENTS.md` sur `main`, puis l'issue #N, ses commentaires, la PR et la branche liées (s'il y en a).
2. Lire le code de la **branche de l'issue** si elle existe ; sinon `main`. Identifier le commit (SHA) réellement consulté.
3. Avant toute écriture locale, demander ou vérifier `git status --short --branch` et les changements non commités. Un assistant uniquement connecté à GitHub doit déclarer qu'il ne voit pas le répertoire local. **S'il doit écrire du code**, l'humain pousse d'abord l'état actuel de la branche (un commit `wip:` suffit) : sinon l'assistant travaille sur du code périmé.
4. Lire les données sur Drive/W&B seulement si nécessaires à la tâche ; vérifier leur fraîcheur.
5. Résumer brièvement : **objectif, critères d'acceptation, état actuel, plan, fichiers concernés et incertitudes**. Développer si le risque scientifique le justifie.
6. Si aucune branche n'existe, proposer `gh issue develop N --name <type>/N-<slug> --checkout` ou l'équivalent Git. L'utilisateur décide du moment de la création.

## 4. Exécuter et vérifier

- Limiter le travail au périmètre de l'issue ; toute découverte indépendante devient une proposition d'issue séparée.
- Procéder par petits changements relisibles ; tester un comportement avant de le modifier quand c'est utile.
- Le code, les tests et les commits doivent identifier les fichiers et leur raison d'être. Pas de code écrasé ou de réécriture extensive sans accord.
- Les tests locaux se déroulent **sur le PC ou dans l'environnement réellement accessible** ; un assistant distant ne prétend pas avoir exécuté une commande sans résultat.
- La CI s'exécute sur la PR ; une CI verte atteste seulement des contrôles qui y sont configurés. Pour le ML, documenter séparément la validation scientifique et les preuves expérimentales.
- Éviter d'intégrer de gros jeux de données, des modèles, des secrets et des fichiers `.env` dans Git.

## 5. Terminer ou interrompre une séance — passation obligatoire

Quand une issue reste ouverte, ou lorsqu'on change d'assistant, publier (ou préparer pour publication) **un seul commentaire de passation concis** :

```markdown
### Passation — AAAA-MM-JJ — [IA ou humain]
- **État :** en cours / bloqué / PR prête / terminé
- **Branche + dernier commit :** `type/N-slug` · `SHA` (ou aucun)
- **Fait / décision :** résumé et justification
- **Vérification :** commande, résultat, lien CI ou « non exécuté »
- **Reste à faire :** 1 à 3 actions précises
- **Liens :** PR, run W&B, rapport ou documentation, si applicable
```

Une PR avec `Closes #N` ferme normalement l'issue lorsqu'elle est fusionnée sur la branche par défaut. La clôture ne signifie pas que l'historique est supprimé. En cas de tâches associées, référencer les autres issues plutôt que d'allonger indéfiniment une seule issue.

## 6. Règles de gouvernance

- **Interdit à l'IA sans décision humaine explicite** : fusionner une PR, pousser directement sur `main`, forcer l'historique, supprimer des données, révoquer un secret ou modifier des artefacts scientifiques de référence.
- Les créations d'issues, branches et PR ou les modifications sur une branche sont effectuées **sur demande**, avec un objectif connu. Aucun travail implicite sur plusieurs dépôts.
- Ne pas committer/pousser un état local non inspecté. Ne pas exiger de push pour pouvoir tester du code local ; on pousse quand le changement est prêt à être partagé.
- Ne jamais considérer une autre IA comme une preuve : vérifier le diff, les tests, les artefacts et les affirmations importantes.
- Dans les résumés scientifiques, distinguer générations de données, configurations, runs complets/partiels, limitations et éventuelles fuites. La liste chiffrée de ces faits n'est **pas** entretenue dans ce protocole.
- Toute modification du contenu publié dans `research-docs` est faite **dans une PR distincte de son dépôt**, avec provenance explicite.

## 7. Invariants scientifiques (sans chiffres)

Les valeurs, IDs de runs et listes de variables vivent dans `research-docs/FACTS_INDEX.md` et les rapports de runs ; ici seulement les règles qui ne changent pas :

- Toute transformation apprise (imputation, scaling, sélection de features — y compris `mi_top_k` —, SMOTE, PCA, calibration, seuils) est ajustée **sur le train seulement**.
- Les observations multiples d'un même objet ne doivent pas se retrouver à la fois dans le train et le test.
- Un modèle n'est dit « spectro-only » que si la liste réelle de ses features (métadonnées du modèle sauvegardé) le prouve, idéalement par test.
- Les générations de données (G1 / G2) et les runs complets / partiels ne se mélangent pas dans un même résultat. Un run référencé publiquement porte ses réserves connues.

## 8. Conventions

- **Branches** : `type/N-slug` (ex. `fix/14-mi-topk-leakage`).
- **Commits** : `type(scope): résumé`, avec `type` ∈ `feat` `fix` `test` `ci` `docs` `refactor` `chore`.
- **Issues** : titre `type: action concrète` ; corps en 5 rubriques — *Problème (avec preuve fichier:ligne) · Pourquoi · Attendu · Vérification · Hors scope* (modèles dans `.github/ISSUE_TEMPLATE/`).
- **Labels** : `bug` `enhancement` `science` `tests` `infra` `documentation` `idea` + priorité `P0` (validité/sécurité) · `P1` (essentiel cassé) · `P2` (amélioration).
- **Backlog** : un seul assistant crée les issues à la fois, après vérification qu'aucun doublon n'existe.

## 9. Commandes usuelles (selon les outils installés)

```bash
gh issue view N --comments
git status --short --branch
gh issue develop N --name <type>/N-<slug> --checkout
python -m pytest -m "not slow"    # après installation du socle pytest
pre-commit run --all-files         # après configuration
# gh pr create --fill  # compléter explicitement le corps avec Closes #N
gh pr checks
```

Les commandes et contrôles n'existent pas nécessairement tous encore : vérifier la configuration du dépôt avant de les présenter comme disponibles.