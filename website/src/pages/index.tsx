import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link'; 
import styles from './index.module.css';

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
          {/* --- 2. MODIFIER LE BOUTON ICI --- */}
          <Link className={styles.ctaBtn} to="/docs/"> {/* Utilise <Link> et pointe vers /docs/ */}
            Accéder à la documentation 🚀
          </Link>
        </div>
        <div className={styles.askBox}>
          <h2>🤖 Assistant IA</h2>
          <input
            className={styles.askInput}
            placeholder="Posez une question sur le projet, l’IA, LAMOST..."
            disabled
          />
          <p className={styles.soon}>(Fonctionnalité à venir)</p>
        </div>
        <section className={styles.features}>
          <div className={styles.featureCard}>
            <h3>🔎 Téléchargement automatisé</h3>
            <p>Récupération robuste de millions de spectres LAMOST directement depuis la source officielle.</p>
          </div>
          <div className={styles.featureCard}>
            <h3>✨ Extraction intelligente de features</h3>
            <p>Détection automatique des raies Hα, Hβ, CaII K&H, FWHM, EW, etc.</p>
          </div>
          <div className={styles.featureCard}>
            <h3>🤖 Modèles ML performants</h3>
            <p>Random Forest, SVM, CNN 1D, personnalisés pour la classification spectrale.</p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
