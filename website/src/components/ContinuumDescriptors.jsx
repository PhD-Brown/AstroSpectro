import React from 'react';

const COLOR = '#F87171'; // continuum family color

const WL_MIN = 3690;
const WL_MAX = 9100;
function wlPct(l) { return ((l - WL_MIN) / (WL_MAX - WL_MIN)) * 100; }

const SLOPES = [
  { name: 'feature_slope_blue',            range: [3800, 4200], interp: 'UV — étoiles chaudes', note: 'négatif = spectre bleu' },
  { name: 'feature_slope_green',           range: [4500, 5500], interp: 'Visible milieu',        note: '' },
  { name: 'feature_slope_red',             range: [5500, 6500], interp: 'Visible rouge',         note: '' },
  { name: 'feature_slope_deep_red',        range: [6500, 7500], interp: 'Proche IR — étoiles froides', note: '' },
  { name: 'feature_continuum_slope_global',range: [3800, 7500], interp: 'Pente globale bleue/rouge', note: '' },
];

const CURVATURES = [
  { name: 'feature_curvature_4000', centre: 4000, interp: 'Saut 4000 Å — frontière âge/métallicité' },
  { name: 'feature_curvature_5000', centre: 5000, interp: 'Visible milieu — courbure du corps noir' },
  { name: 'feature_curvature_6000', centre: 6000, interp: 'Rouge — transition vers IR' },
];

const JUMPS = [
  { name: 'feature_paschen_jump_strength', desc: 'Saut Paschen 8120/8220 Å',    indicator: 'Classe de luminosité', available: true },
  { name: 'feature_break_4000A_strength',  desc: 'Force du saut D4000',          indicator: 'Âge / métallicité', available: true },
  { name: 'feature_break_4000A_width',     desc: 'Largeur du saut D4000',        indicator: 'Âge / métallicité', available: true },
  { name: 'feature_balmer_jump_strength',  desc: 'Saut de Balmer 3646 Å',        indicator: 'NaN — hors couverture LAMOST', available: false },
  { name: 'feature_brackett_jump_strength',desc: 'Saut Brackett 14 500 Å',       indicator: 'NaN — hors couverture LAMOST', available: false },
];

const COLORS_SYN = [
  { name: 'feature_continuum_asymmetry',    formula: '(f_bleu − f_rouge) / (f_bleu + f_rouge)', interp: 'Asymétrie bleue/rouge' },
  { name: 'feature_flux_ratio_blue_red',    formula: 'f_bleu / f_rouge',                         interp: 'Ratio de flux B/R' },
  { name: 'feature_UV_excess_3900',         formula: 'f(3850–3950) / f(4400–4600)',              interp: 'Excès UV — étoiles A' },
  { name: 'feature_color_index_BlueRed',    formula: 'dérivé du flux',                           interp: 'Proxy couleur globale' },
  { name: 'feature_color_gr',              formula: 'magnitude g − r',                           interp: 'Couleur LAMOST g−r' },
  { name: 'feature_color_ri',              formula: 'magnitude r − i',                           interp: 'Couleur LAMOST r−i' },
  { name: 'feature_synthetic_BV',           formula: 'calculé depuis le spectre',                interp: 'Proxy B−V synthétique' },
];

function SectionHeader({ title, count, color, icon }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      padding: '8px 14px',
      background: `${color}0e`,
      borderBottom: `1px solid ${color}25`,
    }}>
      <span style={{ fontSize: '14px' }}>{icon}</span>
      <span style={{ fontWeight: '700', fontSize: '13px', color: color }}>{title}</span>
      <span style={{
        marginLeft: 'auto',
        background: `${color}18`,
        border: `1px solid ${color}35`,
        borderRadius: '12px',
        padding: '1px 8px',
        fontSize: '10px',
        fontWeight: '700',
        color: color,
        fontFamily: 'monospace',
      }}>{count}</span>
    </div>
  );
}

function SlopeRow({ name, range, interp, note }) {
  const [hov, setHov] = React.useState(false);
  const leftPct = wlPct(range[0]);
  const widthPct = wlPct(range[1]) - leftPct;

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '220px 1fr 160px',
        alignItems: 'center',
        gap: '12px',
        padding: '8px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        background: hov ? `${COLOR}09` : 'transparent',
        transition: 'background 0.1s',
        cursor: 'default',
      }}
    >
      <code style={{ fontSize: '11px', color: hov ? COLOR : 'var(--ifm-font-color-base)', opacity: hov ? 1 : 0.75, lineHeight: 1.3 }}>
        {name}
      </code>
      <div style={{ position: 'relative', height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute',
          left: `${leftPct}%`,
          width: `${widthPct}%`,
          top: 0, bottom: 0,
          background: `linear-gradient(90deg, ${COLOR}88, ${COLOR})`,
          borderRadius: '3px',
        }} />
      </div>
      <div>
        <div style={{ fontSize: '11.5px', fontWeight: '600', color: 'var(--ifm-font-color-base)', opacity: 0.8 }}>{interp}</div>
        {note && <div style={{ fontSize: '10px', opacity: 0.4, fontFamily: 'monospace', color: 'var(--ifm-font-color-base)' }}>{note}</div>}
      </div>
    </div>
  );
}

function CurvatureRow({ name, centre, interp }) {
  const [hov, setHov] = React.useState(false);
  const pct = wlPct(centre);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '220px 60px 1fr',
        alignItems: 'center',
        gap: '12px',
        padding: '8px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        background: hov ? `${COLOR}09` : 'transparent',
        transition: 'background 0.1s',
        cursor: 'default',
      }}
    >
      <code style={{ fontSize: '11px', color: hov ? COLOR : 'var(--ifm-font-color-base)', opacity: hov ? 1 : 0.75 }}>
        {name}
      </code>
      <div style={{ fontFamily: 'monospace', fontSize: '12px', fontWeight: '700', color: COLOR }}>{centre} Å</div>
      <div style={{ fontSize: '11.5px', opacity: 0.7, color: 'var(--ifm-font-color-base)' }}>{interp}</div>
    </div>
  );
}

function JumpRow({ name, desc, indicator, available }) {
  const [hov, setHov] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '220px 1fr auto',
        alignItems: 'center',
        gap: '12px',
        padding: '8px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        background: hov ? `${COLOR}09` : 'transparent',
        transition: 'background 0.1s',
        cursor: 'default',
        opacity: available ? 1 : 0.5,
      }}
    >
      <code style={{ fontSize: '11px', color: hov && available ? COLOR : 'var(--ifm-font-color-base)', opacity: hov ? 1 : 0.75 }}>
        {name}
      </code>
      <div>
        <div style={{ fontSize: '11.5px', fontWeight: '500', color: 'var(--ifm-font-color-base)', opacity: 0.8 }}>{desc}</div>
        <div style={{ fontSize: '10.5px', opacity: 0.5, marginTop: '1px', fontStyle: 'italic', color: 'var(--ifm-font-color-base)' }}>{indicator}</div>
      </div>
      <div style={{
        fontSize: '11px',
        fontWeight: '700',
        color: available ? '#34D399' : '#F87171',
        padding: '2px 8px',
        background: available ? 'rgba(52,211,153,0.1)' : 'rgba(248,113,113,0.1)',
        border: `1px solid ${available ? 'rgba(52,211,153,0.25)' : 'rgba(248,113,113,0.25)'}`,
        borderRadius: '4px',
        whiteSpace: 'nowrap',
      }}>
        {available ? '✓ actif' : '✗ NaN'}
      </div>
    </div>
  );
}

function ColorRow({ name, formula, interp }) {
  const [hov, setHov] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '200px 1fr 160px',
        alignItems: 'center',
        gap: '12px',
        padding: '8px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        background: hov ? `${COLOR}09` : 'transparent',
        transition: 'background 0.1s',
        cursor: 'default',
      }}
    >
      <code style={{ fontSize: '11px', color: hov ? COLOR : 'var(--ifm-font-color-base)', opacity: hov ? 1 : 0.75 }}>
        {name}
      </code>
      <div style={{ fontFamily: 'monospace', fontSize: '11px', color: COLOR, opacity: 0.8 }}>{formula}</div>
      <div style={{ fontSize: '11.5px', opacity: 0.7, color: 'var(--ifm-font-color-base)' }}>{interp}</div>
    </div>
  );
}

export default function ContinuumDescriptors() {
  return (
    <div style={{ margin: '20px 0', display: 'flex', flexDirection: 'column', gap: '14px' }}>

      {/* Pentes */}
      <div style={{ border: '1px solid rgba(255,255,255,0.07)', borderRadius: '9px', overflow: 'hidden' }}>
        <SectionHeader title="Pentes locales — sigma-clippées (σ = 2,5)" count={`${SLOPES.length} descripteurs`} color={COLOR} icon="〜" />
        <div style={{ padding: '6px 0 2px' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '220px 1fr 160px',
            gap: '12px',
            padding: '4px 14px 6px',
            borderBottom: '1px solid rgba(255,255,255,0.04)',
          }}>
            {['Descripteur', 'Position 3690–9100 Å', 'Interprétation'].map((h) => (
              <div key={h} style={{ fontSize: '9.5px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', opacity: 0.35, color: 'var(--ifm-font-color-base)' }}>{h}</div>
            ))}
          </div>
          {SLOPES.map((s) => <SlopeRow key={s.name} {...s} />)}
        </div>
      </div>

      {/* Courbures */}
      <div style={{ border: '1px solid rgba(255,255,255,0.07)', borderRadius: '9px', overflow: 'hidden' }}>
        <SectionHeader title="Courbures locales — Savitzky-Golay (dérivée seconde)" count={`${CURVATURES.length} descripteurs`} color={COLOR} icon="∪" />
        <div style={{ padding: '6px 0 2px' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '220px 60px 1fr',
            gap: '12px',
            padding: '4px 14px 6px',
            borderBottom: '1px solid rgba(255,255,255,0.04)',
          }}>
            {['Descripteur', 'Centre', 'Interprétation'].map((h) => (
              <div key={h} style={{ fontSize: '9.5px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', opacity: 0.35, color: 'var(--ifm-font-color-base)' }}>{h}</div>
            ))}
          </div>
          {CURVATURES.map((c) => <CurvatureRow key={c.name} {...c} />)}
        </div>
      </div>

      {/* Sauts */}
      <div style={{ border: '1px solid rgba(255,255,255,0.07)', borderRadius: '9px', overflow: 'hidden' }}>
        <SectionHeader title="Sauts spectraux — discontinuités du continuum" count={`${JUMPS.length} descripteurs`} color={COLOR} icon="↑" />
        <div style={{ padding: '6px 0 2px' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '220px 1fr auto',
            gap: '12px',
            padding: '4px 14px 6px',
            borderBottom: '1px solid rgba(255,255,255,0.04)',
          }}>
            {['Descripteur', 'Description', 'Statut'].map((h) => (
              <div key={h} style={{ fontSize: '9.5px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', opacity: 0.35, color: 'var(--ifm-font-color-base)' }}>{h}</div>
            ))}
          </div>
          {JUMPS.map((j) => <JumpRow key={j.name} {...j} />)}
        </div>
      </div>

      {/* Asymétrie + couleurs */}
      <div style={{ border: '1px solid rgba(255,255,255,0.07)', borderRadius: '9px', overflow: 'hidden' }}>
        <SectionHeader title="Asymétrie et couleurs synthétiques" count={`${COLORS_SYN.length} descripteurs`} color={COLOR} icon="≈" />
        <div style={{ padding: '6px 0 2px' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '200px 1fr 160px',
            gap: '12px',
            padding: '4px 14px 6px',
            borderBottom: '1px solid rgba(255,255,255,0.04)',
          }}>
            {['Descripteur', 'Formule', 'Interprétation'].map((h) => (
              <div key={h} style={{ fontSize: '9.5px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', opacity: 0.35, color: 'var(--ifm-font-color-base)' }}>{h}</div>
            ))}
          </div>
          {COLORS_SYN.map((c) => <ColorRow key={c.name} {...c} />)}
        </div>
      </div>

      <div style={{ fontSize: '11px', opacity: 0.45, color: 'var(--ifm-font-color-base)', fontStyle: 'italic', paddingLeft: '4px' }}>
        Tous les descripteurs de continuum sont calculés sur le spectre normalisé après sigma-clipping. Les NaN correspondent aux longueurs d'onde hors couverture LAMOST (3 690–9 100 Å).
      </div>
    </div>
  );
}
