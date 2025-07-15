import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Ce fichier tourne côté Node.js (pas de code navigateur ici !)

const config: Config = {
  title: 'Astro-Spectro Docs',
  tagline: 'Documentation pour la pipeline de classification spectrale LAMOST DR5',
  favicon: 'img/favicon.ico',

  trailingSlash: false,

  // Flags futurs pour compatibilité Docusaurus v4+
  future: {
    v4: true,
  },

  url: 'https://phd-brown.github.io',
  baseUrl: '/AstroSpectro/',

  organizationName: 'PhD-Brown', // Nom GitHub
  projectName: 'AstroSpectro', // Repo GitHub

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // NOUVELLE SECTION POUR ACTIVER MERMAID
  markdown: {
    mermaid: true,
  },

  // NOUVEAU THÈME AJOUTÉ À LA LISTE
  themes: ['@docusaurus/theme-mermaid'],

  i18n: {
    defaultLocale: 'fr',
    locales: ['fr', 'en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.ts'),
          editUrl: undefined, // désactive le bouton “Edit this page”
        },
        // Blog supprimé
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: {
    // Vous pouvez ajouter une image pour les réseaux sociaux ici si vous en avez une
    // image: 'img/social-card.png', 

    // L'OBJET MANQUANT EST ICI
    navbar: {
      title: 'Astro Spectro Docs',
      logo: {
        alt: 'Logo PhD-Brown AB',
        src: 'img/logo.png', // Assurez-vous que ce chemin est correct
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/PhD-Brown/AstroSpectro/',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: '/docs/',
            },
            {
              label: 'Guides Pratiques',
              to: '/docs/user-guides',
            },
          ],
        },
        {
          title: 'Contact',
          items: [
            {
              label: 'alex.baker.1@ulaval.ca',
              href: 'mailto:alex.baker.1@ulaval.ca',
            },
          ],
        },
        {
          title: 'Liens',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/PhD-Brown/AstroSpectro',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Alex Baker. Built with Docusaurus.`,
    },

    // Vos autres configurations sont bonnes aussi
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },

  } satisfies Preset.ThemeConfig,
};

export default config;
