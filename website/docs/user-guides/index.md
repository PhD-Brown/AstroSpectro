---
id: index
title: Guides Pratiques
---

# Guides Pratiques

Bienvenue dans les guides pratiques du projet **Astro-Spectro**. Cette section vous accompagne dans chaque étape clé du pipeline de classification spectrale, avec des tutoriels concrets et orientés tâche.

L'objectif ici est de vous montrer **comment** accomplir des actions spécifiques : exploiter une étape du pipeline de manière indépendante, approfondir une fonction, ou adapter les paramètres à vos besoins.

---

## Guides disponibles

Chaque guide se concentre sur une partie du workflow. Vous n'êtes pas obligé de les suivre dans l'ordre.

<div className="container" style={{padding: '0'}}>
  <div className="row">
    <div className="col col--6 margin-bottom--lg">
      <div className="card">
        <div className="card__header"><h3>📥 Téléchargement des Données</h3></div>
        <div className="card__body"><p>Apprenez à récupérer des milliers de spectres depuis LAMOST DR5 et à gérer votre catalogue local.</p></div>
        <div className="card__footer">
          <a href="./user-guides/downloading-data" className="button button--primary button--block">Consulter le Guide</a>
        </div>
      </div>
    </div>
    <div className="col col--6 margin-bottom--lg">
      <div className="card">
        <div className="card__header"><h3>🧪 Prétraitement des Spectres</h3></div>
        <div className="card__body"><p>Normalisez, centrez et nettoyez les spectres pour les rendre exploitables par les modèles.</p></div>
        <div className="card__footer">
          <a href="./user-guides/preprocessing" className="button button--primary button--block">Consulter le Guide</a>
        </div>
      </div>
    </div>
    <div className="col col--6 margin-bottom--lg">
      <div className="card">
        <div className="card__header"><h3>✨ Extraction des Features</h3></div>
        <div className="card__body"><p>Détectez automatiquement les raies spectrales et transformez-les en features numériques.</p></div>
        <div className="card__footer">
          <a href="./user-guides/feature-extraction" className="button button--primary button--block">Consulter le Guide</a>
        </div>
      </div>
    </div>
    <div className="col col--6 margin-bottom--lg">
      <div className="card">
        <div className="card__header"><h3>🧠 Entraînement d'un Modèle</h3></div>
        <div className="card__body"><p>Utilisez les features extraites pour entraîner un classifieur supervisé comme Random Forest.</p></div>
        <div className="card__footer">
          <a href="./user-guides/model-training" className="button button--primary button--block">Consulter le Guide</a>
        </div>
      </div>
    </div>
    <div className="col col--6 margin-bottom--lg">
      <div className="card">
        <div className="card__header"><h3>📊 Visualisation des Résultats</h3></div>
        <div className="card__body"><p>Interprétez les spectres, les raies détectées, et les performances du modèle de façon intuitive.</p></div>
        <div className="card__footer">
          <a href="./user-guides/visualization" className="button button--primary button--block">Consulter le Guide</a>
        </div>
      </div>
    </div>
    
  </div>
</div>

:::info
N'hésitez pas à consulter la section [**Concepts & Architecture**](../docs/concepts) pour comprendre la théorie et le "pourquoi" derrière chaque étape.
:::