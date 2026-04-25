import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

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
  
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzz0/zFqN82N020ZJ20BtcP81Ekt09yLff7fXz4eBtk0EihP+RkI5Gq0b',
      crossorigin: 'anonymous',
    },
  ],
  
  url: 'https://phd-brown.github.io',
  baseUrl: '/AstroSpectro/',

  organizationName: 'PhD-Brown', // Nom GitHub
  projectName: 'AstroSpectro', // Repo GitHub

  onBrokenLinks: 'throw',

  // NOUVELLE SECTION POUR ACTIVER MERMAID
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
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
          editUrl: undefined,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          path: './blog',
          routeBasePath: 'journal',
          showReadingTime: true,
          blogSidebarTitle: 'Dernières entrées',
          blogSidebarCount: 'ALL',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'AstroSpectro Docs',
      logo: {
        alt: 'Logo PhD-Brown AB',
        src: 'img/logo.png',
      },
      items: [
        {
          to: '/docs',
          label: 'Documentation',
          position: 'left',
          activeBasePath: 'docs',
        },
        {
          to: '/journal',
          activeBasePath: 'journal',
          position: 'left',
          label: 'Journal de Bord',
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
          title: 'Navigation Rapide',
          items: [
            {
              label: 'Démarrage Rapide',
              to: '/docs/getting-started',
            },
            {
              label: "Guides d'Utilisation",
              to: '/docs/user-guides',
            },
            {
              label: 'Référence API',
              to: '/docs/api',
            },
          ],
        },
        {
          title: 'Communauté',
          items: [
            {
              label: 'Comment Contribuer',
              to: '/docs/community/contributing',
            },
            {
              label: 'Feuille de Route (Roadmap)',
              to: '/docs/community/roadmap',
            },
            {
              label: 'Foire Aux Questions (FAQ)',
              to: '/docs/community/faq',
            },
          ],
        },
        {
          title: 'Plus',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/PhD-Brown/AstroSpectro',
            },
            {
              label: 'Signaler un problème',
              href: 'https://github.com/PhD-Brown/AstroSpectro/issues/new/choose',
            },
            {
              label: 'Me Contacter',
              href: 'mailto:alex.baker.1@ulaval.ca',
            },
          ],
        },
      ],
      // Logo et copyright en dessous des colonnes
      logo: {
        alt: 'Logo PhD-Brown AB',
        src: 'img/logo.png',
        href: 'https://github.com/PhD-Brown',
        width: 50,
        height: 50,
      },
      copyright: `Version 1.0.0 | Copyright © ${new Date().getFullYear()} Alex Baker. Built with Docusaurus. <br/> Licence MIT.`,
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
