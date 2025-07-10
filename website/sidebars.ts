import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: '🚀 Démarrage Rapide',
      link: {
        type: 'generated-index',
        title: 'Démarrage Rapide',
        description: 'Installez et lancez le pipeline en moins de 10 minutes.',
        // On définit explicitement l'URL de cette page de catégorie
        slug: '/category/demarrage-rapide',
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
        // On définit explicitement l'URL ici aussi
        slug: '/category/guides-pratiques',
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
        // Et ici également
        slug: '/category/concepts-cles',
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