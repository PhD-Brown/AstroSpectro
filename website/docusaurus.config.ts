import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Ce fichier tourne côté Node.js (pas de code navigateur ici !)

const config: Config = {
  title: 'Astro-Spectro Docs',
  tagline: 'Documentation pour la pipeline de classification spectrale LAMOST DR5',
  favicon: 'img/favicon.ico',

  // Flags futurs pour compatibilité Docusaurus v4+
  future: {
    v4: true,
  },

  url: 'https://phd-brown.github.io',
  baseUrl: '/astro-spectro-classification/',

  organizationName: 'PhD-Brown', // Nom GitHub
  projectName: 'astro-spectro-classification', // Repo GitHub

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

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
    image: 'img/docusaurus-social-card.jpg', // optionnel, remplace si tu as une image
    navbar: {
      title: 'Astro Spectro Docs',
      logo: {
        alt: 'Astro Spectro Logo',
        src: 'img/logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar', // ou "mainSidebar" selon ta config
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/PhD-Brown/astro-spectro-classification',
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
              to: '/docs/intro',
            },
            {
              label: 'Guide complet',
              to: '/docs/guide',
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
              href: 'https://github.com/PhD-Brown/astro-spectro-classification',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Alex Baker. Built with Docusaurus.`,
    },
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
