import React from 'react';

// Page category colors — mapped to section names
const CATEGORY_COLORS = {
  science:       '#38BDF8',   // Contexte Scientifique
  guides:        '#34D399',   // Guides d'Utilisation
  concepts:      '#A78BFA',   // Concepts & Architecture
  api:           '#FB923C',   // API Reference
  community:     '#FBBF24',   // Communauté
  journal:       '#F87171',   // Journal de bord
  external:      '#94A3B8',   // liens externes
};

// Icon mapping
const CATEGORY_ICONS = {
  science:       '◎',
  guides:        '▶',
  concepts:      '⬡',
  api:           '</>', 
  community:     '◇',
  journal:       '📓',
  external:      '↗',
};

/**
 * RelatedPages — visual related pages grid
 *
 * Props:
 *   pages: array of {
 *     href     : string  — relative or absolute URL
 *     title    : string  — page title
 *     desc     : string  — short description
 *     category : string  — 'science' | 'guides' | 'concepts' | 'api' | 'community' | 'journal' | 'external'
 *     badge    : string  — optional small badge text (e.g. "87 % accuracy")
 *   }
 *   title: string  — optional section header (default "Pages connexes")
 */
export default function RelatedPages({ pages, title = 'Pages connexes' }) {
  return (
    <div style={{ margin: '28px 0 0' }}>
      {/* Section header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        marginBottom: '14px',
      }}>
        <div style={{
          flex: 1,
          height: '1px',
          background: 'rgba(255,255,255,0.07)',
        }} />
        <div style={{
          fontSize: '11px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.4,
          whiteSpace: 'nowrap',
          padding: '0 10px',
        }}>{title}</div>
        <div style={{
          flex: 1,
          height: '1px',
          background: 'rgba(255,255,255,0.07)',
        }} />
      </div>

      {/* Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
        gap: '10px',
      }}>
        {pages.map((page, i) => {
          const cat = page.category || 'science';
          const color = CATEGORY_COLORS[cat] || CATEGORY_COLORS.science;
          const icon  = CATEGORY_ICONS[cat]  || '◎';
          const isExternal = cat === 'external' || page.href?.startsWith('http');

          return (
            <a
              key={i}
              href={page.href}
              target={isExternal ? '_blank' : undefined}
              rel={isExternal ? 'noopener noreferrer' : undefined}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
                padding: '14px 16px',
                background: 'var(--ifm-card-background-color)',
                border: `1px solid rgba(255,255,255,0.07)`,
                borderLeft: `3px solid ${color}`,
                borderRadius: '8px',
                textDecoration: 'none',
                transition: 'all 0.15s',
                position: 'relative',
                overflow: 'hidden',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = `${color}0c`;
                e.currentTarget.style.borderColor = `${color}55`;
                e.currentTarget.style.borderLeftColor = color;
                e.currentTarget.style.transform = 'translateX(3px)';
                e.currentTarget.style.boxShadow = `0 3px 14px ${color}14`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'var(--ifm-card-background-color)';
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)';
                e.currentTarget.style.borderLeftColor = color;
                e.currentTarget.style.transform = 'none';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              {/* Top row: icon + title + badge */}
              <div style={{
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                gap: '8px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '7px', flex: 1, minWidth: 0 }}>
                  <span style={{
                    fontFamily: 'monospace',
                    fontSize: '13px',
                    color: color,
                    flexShrink: 0,
                    lineHeight: 1,
                  }}>{icon}</span>
                  <span style={{
                    fontWeight: '700',
                    fontSize: '13px',
                    color: color,
                    lineHeight: 1.3,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                  }}>{page.title}</span>
                </div>

                {/* Badge */}
                {page.badge && (
                  <div style={{
                    background: `${color}18`,
                    border: `1px solid ${color}35`,
                    borderRadius: '12px',
                    padding: '1px 7px',
                    fontSize: '10px',
                    fontWeight: '700',
                    color: color,
                    whiteSpace: 'nowrap',
                    flexShrink: 0,
                    fontFamily: 'monospace',
                  }}>{page.badge}</div>
                )}

                {/* External arrow */}
                {isExternal && (
                  <span style={{ fontSize: '11px', color: color, opacity: 0.5, flexShrink: 0 }}>↗</span>
                )}
              </div>

              {/* Description */}
              {page.desc && (
                <div style={{
                  fontSize: '11.5px',
                  color: 'var(--ifm-font-color-base)',
                  opacity: 0.55,
                  lineHeight: 1.5,
                  paddingLeft: '20px',
                }}>{page.desc}</div>
              )}
            </a>
          );
        })}
      </div>
    </div>
  );
}
