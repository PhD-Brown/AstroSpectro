import React from 'react';

// Colors chosen to approximate actual spectral line wavelength colors
const FAMILIES = [
  {
    name: 'Raies de Balmer',
    range: 'Hα – Hθ',
    count: '~35',
    indicator: 'Teff — température effective',
    color: '#60A5FA',     // Blue — Balmer series (visible blue/violet)
    description: 'Série de l\'hydrogène en absorption. Indicateur thermique primaire : force maximale vers 10 000 K (type A), s\'affaiblit vers M et vers O.',
    lines: 'Hα 6563 Å · Hβ 4861 Å · Hγ 4341 Å · Hδ 4102 Å · Hθ 3798 Å',
    pcaRole: 'PC1 (Balmer : 31.5%)',
  },
  {
    name: 'Ca II H&K + triplet IR',
    range: 'H 3968 · K 3933 · IR 8498–8662',
    count: '~20',
    indicator: 'Métallicité + activité chromosphérique',
    color: '#C084FC',     // Violet — CaII near UV
    description: 'Calcium ionisé en absorption. Sensible à la fois à la métallicité ([Ca/H]) et à l\'activité magnétique stellaire (remplissage en émission).',
    lines: 'Ca II H 3968 Å · Ca II K 3933 Å · IR triplet 8498, 8542, 8662 Å',
    pcaRole: 'PC1 (Ca : 22.4%) · SHAP top-3',
  },
  {
    name: 'Magnésium & indices α',
    range: 'Mg b · Mg I 5184 · Mg II',
    count: '~15',
    indicator: '[Mg/Fe] · population stellaire',
    color: '#34D399',     // Green — Mg b triplet ~5170 Å
    description: 'Triplet Mg b et raies du magnésium. Traceurs de l\'enrichissement en éléments α — discrimine les populations du halo et du disque galactique.',
    lines: 'Mg b 5167–5183 Å · Mg I 5184 Å · Mg II 4481 Å',
    pcaRole: 'PC1 (W(Mgb) = −0.171)',
  },
  {
    name: 'Fer & métaux du pic',
    range: 'Fe, Cr, V, Ni, Ba, Sr...',
    count: '~30',
    indicator: '[Fe/H] · métallicité',
    color: '#FB923C',     // Orange — Fe I lines ~6300 Å region
    description: 'Raies des métaux du pic du fer. Mesure directe de l\'abondance en fer et des métaux lourds. Dominantes dans PC2.',
    lines: 'Fe I 5270, 5335 Å · Cr I · V I · Ni I · Ba II · Sr II',
    pcaRole: 'PC2 ([Fe/H]proxy : 0.205)',
  },
  {
    name: 'Indices Lick / SDSS / CaH',
    range: 'Indices Lick IDS · CaH · TiO',
    count: '~18',
    indicator: 'Âge + métallicité composite',
    color: '#FBBF24',     // Amber — composite/broadband
    description: 'Indices spectroscopiques standardisés. Combinaisons de bandes larges calibrées pour mesurer âge et métallicité des populations stellaires.',
    lines: 'Indices Lick IDS · CaH1, CaH2 · TiO1, TiO2 · SDSS composites',
    pcaRole: 'PC2 (Idxmétal : 0.198)',
  },
  {
    name: 'Continuum · pentes · couleurs',
    range: 'Ratios de flux · indices photom.',
    count: '~65',
    indicator: 'Teff (pente) · luminosité (sauts)',
    color: '#F87171',     // Red — continuum shape = blackbody peak
    description: 'Forme globale du continu spectral. Plus grande famille en nombre. Encode la température via la pente de corps noir et la luminosité via les sauts spectraux.',
    lines: 'fbleu/frouge · B−V synthétique · pentes locales · indices photométriques',
    pcaRole: 'PC1 (continuum : fbleu/frouge = 0.161)',
  },
];

function FamilyCard({ name, range, count, indicator, color, description, lines, pcaRole }) {
  const [hovered, setHovered] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: 'var(--ifm-card-background-color)',
        border: `1px solid ${hovered ? color + '50' : 'rgba(255,255,255,0.07)'}`,
        borderLeft: `4px solid ${color}`,
        borderRadius: '10px',
        padding: '16px 18px',
        transition: 'transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease',
        transform: hovered ? 'translateX(3px)' : 'translateX(0)',
        boxShadow: hovered ? `0 4px 20px ${color}18` : '0 1px 4px rgba(0,0,0,0.06)',
        cursor: 'default',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontWeight: '700',
              fontSize: '14px',
              color: color,
              marginBottom: '2px',
              lineHeight: 1.3,
            }}
          >
            {name}
          </div>
          <div
            style={{
              fontSize: '10.5px',
              fontFamily: "'JetBrains Mono', monospace",
              color: 'var(--ifm-font-color-base)',
              opacity: 0.5,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {range}
          </div>
        </div>
        {/* Count badge */}
        <div
          style={{
            background: `${color}18`,
            border: `1px solid ${color}44`,
            borderRadius: '14px',
            padding: '3px 11px',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: '13px',
            fontWeight: '800',
            color: color,
            whiteSpace: 'nowrap',
            flexShrink: 0,
          }}
        >
          {count}
        </div>
      </div>

      {/* Description */}
      <div
        style={{
          fontSize: '12px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.72,
          lineHeight: 1.55,
        }}
      >
        {description}
      </div>

      {/* Lines snippet */}
      <div
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '10.5px',
          color: color,
          opacity: 0.75,
          background: `${color}0c`,
          border: `1px solid ${color}22`,
          borderRadius: '5px',
          padding: '5px 9px',
          lineHeight: 1.5,
        }}
      >
        {lines}
      </div>

      {/* PCA role */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '11px',
          fontWeight: '600',
          color: color,
        }}
      >
        <span style={{ opacity: 0.6 }}>→ PCA :</span>
        <span>{pcaRole}</span>
      </div>
    </div>
  );
}

export default function DescriptorFamilies() {
  const total = 183;

  return (
    <div style={{ margin: '24px 0' }}>
      {/* Summary bar */}
      <div
        style={{
          display: 'flex',
          gap: '0',
          height: '8px',
          borderRadius: '4px',
          overflow: 'hidden',
          marginBottom: '20px',
        }}
      >
        {FAMILIES.map((f) => {
          const n = parseInt(f.count.replace('~', ''), 10);
          return (
            <div
              key={f.name}
              title={`${f.name} : ${f.count} descripteurs`}
              style={{
                flex: `0 0 ${(n / total) * 100}%`,
                background: f.color,
                opacity: 0.85,
              }}
            />
          );
        })}
      </div>

      {/* Cards grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))',
          gap: '14px',
        }}
      >
        {FAMILIES.map((f) => (
          <FamilyCard key={f.name} {...f} />
        ))}
      </div>

      {/* Footer note */}
      <div
        style={{
          marginTop: '16px',
          fontSize: '12px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.5,
          textAlign: 'center',
        }}
      >
        183 descripteurs · 6 familles spectroscopiques ·{' '}
        <a href="/AstroSpectro/docs/science/features-list-183" style={{ color: 'inherit', textDecoration: 'underline dotted' }}>
          catalogue exhaustif →
        </a>
      </div>
    </div>
  );
}
