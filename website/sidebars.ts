// website/sidebars.ts

import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro', // Fichier docs/intro.md
    {
      type: 'category',
      label: '🚀 Démarrage Rapide',
      link: {
        type: 'generated-index',
        title: 'Démarrage Rapide',
        description: 'Installez et lancez le pipeline en moins de 10 minutes.',
      },
      items: [
        'getting-started/installation',
        'getting-started/first-run',
      ],
    },
    {
      type: 'category',
      label: '📚 Guides Pratiques',
      link: {
        type: 'generated-index',
        title: 'Guides Pratiques',
        description: 'Tutoriels détaillés pour chaque étape du pipeline.',
      },
      items: [
        'guides/data-download',
        'guides/preprocessing',
        'guides/feature-extraction',
        'guides/model-training',
        'guides/visualization',
      ],
    },
    {
      type: 'category',
      label: '💡 Concepts Clés',
      link: {
        type: 'generated-index',
        title: 'Concepts Clés',
        description: 'Comprendre l\'architecture et les choix techniques du projet.',
      },
      items: [
        'concepts/project-structure',
        'concepts/preprocessing-module',
        'concepts/feature-module',
        'concepts/classifier-module',
      ],
    },
    'contributing',
    'roadmap',
  ],
};

export default sidebars;
