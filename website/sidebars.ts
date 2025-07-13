// website/sidebars.ts

import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Cette configuration de sidebar est conçue pour être claire, évolutive et
 * inspirée des meilleures pratiques de documentation pour les projets
 * scientifiques et open-source.
 *
 * Structure par intention :
 * - Getting Started: Pour une prise en main rapide.
 * - Scientific Context: Pour comprendre le fond scientifique.
 * - User Guides: Pour réaliser des tâches spécifiques ("How-To").
 * - Concepts & Architecture: Pour comprendre le fonctionnement interne.
 * - API Reference: Pour les détails techniques du code.
 * - Community: Pour tout ce qui concerne la contribution et l'utilisation.
 */
const sidebars: SidebarsConfig = {
  // Le nom de notre sidebar principale.
  // Assurez-vous qu'il correspond au sidebarId dans docusaurus.config.ts
  docsSidebar: [
    
    // =====================================================================
    // CATÉGORIE 1 : DÉMARRAGE RAPIDE
    // =====================================================================
    {
      type: 'category',
      label: 'Démarrage Rapide',
      link: {
        type: 'doc',
        id: 'getting-started/index', // Pointe vers une page d'accueil de section dédiée
      },
      items: [
        'getting-started/installation',
        'getting-started/first-run',
      ],
    },

    // =====================================================================
    // CATÉGORIE 2 : CONTEXTE SCIENTIFIQUE (Le "Pourquoi")
    // =====================================================================
    {
      type: 'category',
      label: 'Contexte Scientifique',
      link: {
        type: 'doc',
        id: 'science/index',
      },
      items: [
        'science/lamost-dr5-data',
        'science/methodology',
        'science/results-analysis',
      ],
    },

    // =====================================================================
    // CATÉGORIE 3 : GUIDES D'UTILISATION (Le "Comment")
    // =====================================================================
    {
      type: 'category',
      label: 'Guides d\'Utilisation',
      link: {
        type: 'doc',
        id: 'user-guides/index', 
      },
      items: [
        'user-guides/downloading-data',
        'user-guides/preprocessing',
        'user-guides/feature-extraction',
        'user-guides/model-training',
        'user-guides/visualization',
      ],
    },
    
    // =====================================================================
    // CATÉGORIE 4 : CONCEPTS & ARCHITECTURE (Le "Comment ça marche")
    // =====================================================================
    {
      type: 'category',
      label: 'Concepts & Architecture',
      link: {
        type: 'doc',
        id: 'concepts/index', 
      },
      items: [
        'concepts/project-structure',
        'concepts/pipeline-overview',
        'concepts/feature-engineering-theory',
      ],
    },

    // =====================================================================
    // CATÉGORIE 5 : RÉFÉRENCE DE L'API
    // =====================================================================
    {
      type: 'category',
      label: 'API Reference',
      link: {
        type: 'doc',
        id: 'api/index', 
      },
      items: [
        'api/preprocessor',
        'api/feature-engineer',
        'api/classifier',
      ],
    },
    
    // =====================================================================
    // CATÉGORIE 6 : COMMUNAUTÉ & RESSOURCES
    // =====================================================================
    {
      type: 'category',
      label: 'Communauté',
      link: {
        type: 'doc',
        id: 'community/index',
      },
      items: [
        'community/acknowledgments',
        'community/contributing',
        'community/roadmap',
        'community/citing',
        'community/faq',
      ],
    },
  ],
};

export default sidebars;