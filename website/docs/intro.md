---
slug: /
hide_title: true
className: 'doc-page-hub'
---

<div style={{textAlign: 'center'}}>
  <h1 style={{fontSize: '2.6rem', marginBottom: '0.2em'}}>
    Bienvenue sur la Documentation Astro-Spectro
  </h1>
  <p style={{maxWidth: 750, margin: 'auto', fontSize: '1.15rem'}}>
    Ce projet fournit un pipeline robuste et reproductible pour la classification automatique des spectres stellaires.
  </p>
</div>

<br/>

{/* Conteneur personnalisé pour les cartes */}
<div className="hub-container">
  <div className="container">
    <div className="row">
      {/* Carte 1 : Démarrage Rapide */}
      <div className="col col--6 margin-bottom--lg margin-horiz--auto">
        <div className="card">
          <div className="card__header">
            <h3>🚀 Getting Started</h3>
          </div>
          <div className="card__body">
            <p>
              Suivez un tutoriel complet pour installer le projet localement et lancer votre première classification en moins de 10 minutes.
            </p>
          </div>
          <div className="card__footer">
            <a href="./docs/getting-started" className="button button--secondary button--block">
              Commencer le Tutoriel
            </a>
          </div>
        </div>
      </div>
      {/* Carte 2 : Guides Utilisateur */}
      <div className="col col--6 margin-bottom--lg margin-horiz--auto">
        <div className="card">
          <div className="card__header">
            <h3>📚 User Guides</h3>
          </div>
          <div className="card__body">
            <p>
              Des tutoriels détaillés pour chaque étape clé : téléchargement des données, traitement des spectres, entraînement du modèle, visualisation des résultats.
            </p>
          </div>
          <div className="card__footer">
            <a href="./docs/user-guides" className="button button--secondary button--block">
              Consulter les Guides
            </a>
          </div>
        </div>
      </div>
      {/* Carte 3 : Contexte Scientifique */}
      <div className="col col--6 margin-bottom--lg margin-horiz--auto">
        <div className="card">
          <div className="card__header">
            <h3>🔬 Scientific Context</h3>
          </div>
          <div className="card__body">
            <p>
              Plongez dans le contexte astrophysique : catalogue LAMOST, types spectraux, classification stellaire et méthodologie derrière ce projet.
            </p>
          </div>
          <div className="card__footer">
            <a href="./docs/science" className="button button--secondary button--block">
              Explorer le Contexte
            </a>
          </div>
        </div>
      </div>
      {/* Carte 4 : Communauté & Contribution */}
      <div className="col col--6 margin-bottom--lg margin-horiz--auto">
        <div className="card">
          <div className="card__header">
            <h3>🌐 Community & Contributing</h3>
          </div>
          <div className="card__body">
            <p>
              Vous souhaitez contribuer, poser une question ou citer ce projet dans un article ? Toutes les ressources et canaux sont ici.
            </p>
          </div>
          <div className="card__footer">
            <a href="./docs/community" className="button button--secondary button--block">
              Rejoindre la Communauté
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<br/>

:::info 🚧 Note de développement
Cette documentation est en cours de rédaction active pour la version **`v0.1.0`** du pipeline.  
Certaines pages peuvent être incomplètes ou sujettes à changement.
:::