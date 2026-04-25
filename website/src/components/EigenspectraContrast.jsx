import React from 'react';

const FLUX_PCS = [
  { pc: 'PC1', pct: 46.6, interp: 'Pente de couleur — T_eff', color: '#38BDF8', physical: true },
  { pc: 'PC2', pct: 31.8, interp: 'Montée infrarouge — étoiles K-M', color: '#60A5FA', physical: true },
  { pc: 'PC3', pct: 12.9, interp: 'Coupure instrumentale LAMOST', color: '#F87171', physical: false },
];

const DESC_PCS = [
  { pc: 'PC1', pct: 16.9, interp: 'Axe thermique ρ(Teff)=+0,831', color: '#38BDF8', physical: true },
  { pc: 'PC2', pct: 12.0, interp: 'Axe métallicité + SNR instrumental', color: '#F59E0B', physical: true  },
  { pc: 'PC3', pct:  9.5, interp: 'Structure spectrale mixte', color: '#94A3B8', physical: true },
];

function PCRow({ pc, pct, interp, color, physical }) {
  const [hov, setHov] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '8px 12px',
        borderRadius: '6px',
        background: hov ? `${color}0e` : 'transparent',
        transition: 'background 0.12s',
        cursor: 'default',
      }}
    >
      <div style={{
        fontFamily: 'monospace',
        fontWeight: '800',
        fontSize: '13px',
        color: color,
        width: '28px',
        flexShrink: 0,
      }}>{pc}</div>

      <div style={{ flex: 1 }}>
        <div style={{
          height: '8px',
          background: 'rgba(255,255,255,0.06)',
          borderRadius: '4px',
          overflow: 'hidden',
          marginBottom: '4px',
        }}>
          <div style={{
            width: `${pct}%`,
            height: '100%',
            background: color,
            borderRadius: '4px',
            opacity: 0.8,
          }} />
        </div>
        <div style={{ fontSize: '11px', color: 'var(--ifm-font-color-base)', opacity: 0.6, lineHeight: 1.3 }}>
          {interp}
        </div>
      </div>

      <div style={{
        fontFamily: 'monospace',
        fontWeight: '700',
        fontSize: '14px',
        color: color,
        flexShrink: 0,
        textAlign: 'right',
      }}>{pct}%</div>

      <div style={{
        width: '14px',
        height: '14px',
        borderRadius: '50%',
        background: physical ? 'rgba(52,211,153,0.2)' : 'rgba(248,113,113,0.2)',
        border: `1px solid ${physical ? '#34D399' : '#F87171'}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '8px',
        flexShrink: 0,
      }}>
        {physical ? '✓' : '⚠'}
      </div>
    </div>
  );
}

function Panel({ title, subtitle, color, nChannels, k95, totalPct3, pcs, badge, badgeColor }) {
  return (
    <div style={{
      flex: 1,
      background: 'var(--ifm-card-background-color)',
      border: `1px solid ${color}30`,
      borderTop: `3px solid ${color}`,
      borderRadius: '10px',
      padding: '18px',
      minWidth: 0,
    }}>
      <div style={{ marginBottom: '14px' }}>
        <div style={{
          fontWeight: '800',
          fontSize: '14px',
          color: 'var(--ifm-font-color-base)',
          marginBottom: '3px',
        }}>{title}</div>
        <div style={{ fontSize: '11px', opacity: 0.5, color: 'var(--ifm-font-color-base)' }}>
          {subtitle}
        </div>
      </div>

      {/* Key stat */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '10px 12px',
        background: `${color}0a`,
        border: `1px solid ${color}25`,
        borderRadius: '7px',
        marginBottom: '14px',
      }}>
        <div>
          <div style={{ fontSize: '10px', opacity: 0.45, marginBottom: '2px', color: 'var(--ifm-font-color-base)' }}>
            Composantes pour 95 % variance
          </div>
          <div style={{
            fontFamily: 'monospace',
            fontSize: '26px',
            fontWeight: '900',
            color: color,
            lineHeight: 1,
          }}>K = {k95}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '10px', opacity: 0.45, marginBottom: '2px', color: 'var(--ifm-font-color-base)' }}>
            Top 3 cumulent
          </div>
          <div style={{
            fontFamily: 'monospace',
            fontSize: '20px',
            fontWeight: '700',
            color: color,
          }}>{totalPct3}%</div>
        </div>
      </div>

      {/* PC rows */}
      <div style={{ marginBottom: '12px' }}>
        {pcs.map((p) => <PCRow key={p.pc} {...p} />)}
      </div>

      {/* Badge */}
      <div style={{
        padding: '8px 12px',
        background: `${badgeColor}0c`,
        border: `1px solid ${badgeColor}30`,
        borderRadius: '6px',
        fontSize: '11px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.8,
      }}>{badge}</div>
    </div>
  );
}

export default function EigenspectraContrast() {
  return (
    <div style={{ margin: '28px 0' }}>
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
        <Panel
          title="PCA sur flux bruts"
          subtitle="3 921 canaux spectraux · 10 000 spectres"
          color="#A78BFA"
          nChannels={3921}
          k95={3}
          totalPct3={91.3}
          pcs={FLUX_PCS}
          badge="✓ Physiquement : les pixels adjacents (Δλ~2 Å) sont fortement corrélés — longueur de cohérence ~10 Å. 3 composantes suffisent !"
          badgeColor="#34D399"
        />
        <Panel
          title="PCA sur 183 descripteurs"
          subtitle="Features ingéniérisées · 43 019 spectres"
          color="#38BDF8"
          nChannels={183}
          k95={91}
          totalPct3={38.4}
          pcs={DESC_PCS}
          badge="→ Les descripteurs éliminent la redondance spectrale mais introduisent une information orthogonale réelle : 91 composantes pour 95 %."
          badgeColor="#38BDF8"
        />
      </div>

      {/* Key insight box */}
      <div style={{
        marginTop: '14px',
        padding: '14px 18px',
        background: 'rgba(167,139,250,0.06)',
        border: '1px solid rgba(167,139,250,0.2)',
        borderRadius: '8px',
        fontSize: '13px',
        color: 'var(--ifm-font-color-base)',
        opacity: 0.85,
        lineHeight: 1.6,
      }}>
        <strong style={{ color: '#A78BFA' }}>Interprétation du contraste :</strong> Les 183 descripteurs éliminent la redondance des pixels adjacents mais introduisent une information physique réellement orthogonale (température, métallicité, gravité, activité chromosphérique…). Résultat : la dimensionnalité intrinsèque passe de ~3 à ~91 — ce n'est pas un signe d'inefficacité, mais la preuve que les features encodent des dimensions physiques <em>distinctes</em>.
      </div>
    </div>
  );
}
