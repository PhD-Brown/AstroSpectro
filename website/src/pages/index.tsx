import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

// 1. Importer FontAwesome et les icônes nécessaires
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDownload, faWandSparkles, faRobot, faRocket, faHexagonNodes } from '@fortawesome/free-solid-svg-icons';

export default function Home() {
  return (
    <Layout
      title="Accueil"
      description="Documentation pour le pipeline de classification spectrale LAMOST DR5"
    >
      <main className={styles.heroSection}>
        <div className={styles.heroContent}>
          <h1>Astro-Spectro Docs</h1>
          <p className={styles.subtitle}>
            Un pipeline intelligent pour la classification automatique des spectres du catalogue <strong>LAMOST DR5</strong>.
          </p>
          <Link className={styles.ctaBtn} to="/docs/">
            Accéder à la documentation <FontAwesomeIcon icon={faRocket} />
          </Link>
        </div>
        <div className={styles.askBox}>
          <h2><FontAwesomeIcon icon={faRobot} /> Assistant IA</h2>
          <input
            className={styles.askInput}
            placeholder="Posez une question sur le projet, l’IA, LAMOST..."
            disabled
          />
          <p className={styles.soon}>(Fonctionnalité à venir)</p>
        </div>
        <div className={styles.featuresSection}>
          <div className="container">
            <div className="row">
              {/* Carte 1 avec icône */}
              <div className="col col--4">
                <div className={styles.featureCard}>
                  <h3><FontAwesomeIcon icon={faDownload} /> Téléchargement automatisé</h3>
                  <p>Récupération robuste de millions de spectres LAMOST directement depuis la source officielle.</p>
                </div>
              </div>
              {/* Carte 2 avec icône */}
              <div className="col col--4">
                <div className={styles.featureCard}>
                  <h3><FontAwesomeIcon icon={faWandSparkles} /> Extraction intelligente de features</h3>
                  <p>Détection automatique des raies Hα, Hβ, CaII K&H, et calcul de leurs propriétés physiques.</p>
                </div>
              </div>
              {/* Carte 3 avec icône */}
              <div className="col col--4">
                <div className={styles.featureCard}>
                  <h3><FontAwesomeIcon icon={faHexagonNodes} /> Modèles ML performants</h3>
                  <p>Random Forest, SVM, CNN 1D, et autres modèles personnalisés pour la classification spectrale.</p>
                </div>
              </div>
              
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}