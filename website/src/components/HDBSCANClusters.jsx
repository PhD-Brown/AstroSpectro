import React from 'react';

const CLUSTERS = [
  { id: 'C11', n: 16716, pop: 'Séquence principale G-K',  teff: '5000–6000', logg: '4,0–4,5', color: '#F59E0B' },
  { id: 'C13', n: 10373, pop: 'Naines K froides',          teff: '4500–5000', logg: '≈ 4,5',   color: '#FB923C' },
  { id: 'C5',  n:  5083, pop: null,                        teff: null,        logg: null,       color: '#FBBF24' },
  { id: 'C3',  n:   818, pop: null,                        teff: null,        logg: null,       color: '#A3E635' },
  { id: 'C2',  n:   915, pop: null,                        teff: null,        logg: null,       color: '#34D399' },
  { id: 'C19', n:   900, pop: 'Sous-géantes / base RGB',   teff: '5000–5500', logg: '3,0–3,5', color: '#38BDF8' },
  { id: 'C12', n:   851, pop: 'Sous-géantes / base RGB',   teff: '5000–5500', logg: '3,0–3,5', color: '#60A5FA' },
  { id: 'C1',  n:   654, pop: 'Sous-géantes / base RGB',   teff: '5000–5500', logg: '3,0–3,5', color: '#818CF8' },
  { id: 'C16', n:   620, pop: null,                        teff: null,        logg: null,       color: '#A78BFA' },
  { id: 'C17', n:   615, pop: null,                        teff: null,        logg: null,       color: '#C084FC' },
  { id: 'C9',  n:   576, pop: null,                        teff: null,        logg: null,       color: '#E879F9' },
  { id: 'C7',  n:   474, pop: null,                        teff: null,        logg: null,       color: '#FB7185' },
];

const MAX_N = 16716;

const HR_GROUPS = [
  {
    clusters: 'C19, C1, C12',
    pop: 'Sous-géantes / base RGB',
    teff: '5000–5500 K',
    logg: '≈ 3,0–3,5',
    color: '#38BDF8',
    note: 'Ca II renforcé · Balmer affaibli → pression de radiation plus basse',
  },
  {
    clusters: 'C13',
    pop: 'Naines K froides',
    teff: '4500–5000 K',
    logg: '≈ 4,5',
    color: '#FB923C',
    note: 'Cluster le plus peuplé avec identité physique claire',
  },
  {
    clusters: 'C11',
    pop: 'Séquence principale G-K',
    teff: '5000–6000 K',
    logg: '≈ 4,0–4,5',
    color: '#F59E0B',
    note: 'Cluster dominant — population de référence de la séquence principale',
  },
];

function ClusterBar({ cluster, maxN }) {
  const [hov, setHov] = React.useState(false);
  const pct = (cluster.n / maxN) * 100;

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '4px 0',
      }}
    >
      {/* Cluster ID */}
      <div style={{
        width: '32px',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: '12px',
        fontWeight: '700',
        color: cluster.color,
        flexShrink: 0,
        textAlign: 'right',
      }}>
        {cluster.id}
      </div>

      {/* Bar */}
      <div style={{
        flex: 1,
        height: '18px',
        background: 'rgba(255,255,255,0.05)',
        borderRadius: '4px',
        overflow: 'hidden',
        position: 'relative',
      }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: hov
            ? cluster.color
            : `${cluster.color}bb`,
          borderRadius: '4px',
          transition: 'background 0.15s, width 0.4s ease',
          display: 'flex',
          alignItems: 'center',
          paddingLeft: '8px',
          minWidth: '4px',
        }}>
          {pct > 12 && (
            <span style={{
              fontSize: '11px',
              fontWeight: '700',
              color: '#000',
              fontFamily: 'monospace',
              whiteSpace: 'nowrap',
            }}>
              {cluster.n.toLocaleString('fr-FR')}
            </span>
          )}
        </div>
        {pct <= 12 && (
          <span style={{
            position: 'absolute',
            left: `${pct}% + 6px`,
            left: `calc(${pct}% + 6px)`,
            top: '50%',
            transform: 'translateY(-50%)',
            fontSize: '11px',
            fontFamily: 'monospace',
            color: 'var(--ifm-font-color-base)',
            opacity: 0.6,
            whiteSpace: 'nowrap',
          }}>
            {cluster.n.toLocaleString('fr-FR')}
          </span>
        )}
      </div>

      {/* Population label */}
      {cluster.pop && (
        <div style={{
          fontSize: '11px',
          color: cluster.color,
          opacity: 0.75,
          flexShrink: 0,
          maxWidth: '160px',
          fontStyle: 'italic',
          display: hov ? 'block' : 'none',
        }}>
          {cluster.pop}
        </div>
      )}
    </div>
  );
}

export default function HDBSCANClusters() {
  return (
    <div style={{ margin: '24px 0' }}>
      {/* Summary stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
        gap: '10px',
        marginBottom: '20px',
      }}>
        {[
          { label: 'Clusters', value: '20', color: '#F59E0B' },
          { label: 'Bruit', value: '2 643', sub: '6,14 %', color: '#F87171' },
          { label: 'Cluster C11', value: '16 716', sub: 'étoiles', color: '#F59E0B' },
          { label: 'Cluster C13', value: '10 373', sub: 'étoiles', color: '#FB923C' },
        ].map((s) => (
          <div key={s.label} style={{
            background: 'var(--ifm-card-background-color)',
            border: `1px solid ${s.color}28`,
            borderLeft: `3px solid ${s.color}`,
            borderRadius: '7px',
            padding: '10px 14px',
          }}>
            <div style={{
              fontFamily: 'monospace',
              fontSize: '18px',
              fontWeight: '800',
              color: s.color,
              lineHeight: 1.1,
            }}>{s.value}</div>
            <div style={{ fontSize: '11px', opacity: 0.55, marginTop: '2px' }}>
              {s.label}{s.sub ? ` (${s.sub})` : ''}
            </div>
          </div>
        ))}
      </div>

      {/* Bar chart */}
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        padding: '18px 20px',
        marginBottom: '20px',
      }}>
        <div style={{
          fontSize: '12px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          opacity: 0.5,
          marginBottom: '14px',
          color: 'var(--ifm-font-color-base)',
        }}>
          Top 12 clusters — population (étoiles)
        </div>
        {CLUSTERS.map((c) => (
          <ClusterBar key={c.id} cluster={c} maxN={MAX_N} />
        ))}
        <div style={{
          marginTop: '12px',
          fontSize: '11px',
          opacity: 0.35,
          fontFamily: 'monospace',
          color: 'var(--ifm-font-color-base)',
        }}>
          min_cluster_size=75 · min_samples=20 · bruit exclus de ce graphe
        </div>
      </div>

      {/* HR location table */}
      <div style={{
        background: 'rgba(56,189,248,0.04)',
        border: '1px solid rgba(56,189,248,0.15)',
        borderRadius: '10px',
        padding: '18px 20px',
      }}>
        <div style={{
          fontSize: '12px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          color: '#38BDF8',
          opacity: 0.8,
          marginBottom: '14px',
        }}>
          Localisation sur le diagramme HR (Teff × log g × Gaia DR3)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {HR_GROUPS.map((g) => (
            <div key={g.clusters} style={{
              display: 'grid',
              gridTemplateColumns: '90px 1fr auto auto',
              gap: '12px',
              alignItems: 'center',
              padding: '10px 12px',
              background: `${g.color}0c`,
              border: `1px solid ${g.color}25`,
              borderRadius: '7px',
            }}>
              <div style={{
                fontFamily: 'monospace',
                fontSize: '12px',
                fontWeight: '800',
                color: g.color,
              }}>{g.clusters}</div>
              <div>
                <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--ifm-font-color-base)', marginBottom: '2px' }}>
                  {g.pop}
                </div>
                <div style={{ fontSize: '11px', opacity: 0.5, fontStyle: 'italic', color: 'var(--ifm-font-color-base)' }}>
                  {g.note}
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '10px', opacity: 0.45, marginBottom: '1px' }}>Teff</div>
                <div style={{ fontFamily: 'monospace', fontSize: '12px', fontWeight: '600', color: g.color }}>{g.teff}</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '10px', opacity: 0.45, marginBottom: '1px' }}>log g</div>
                <div style={{ fontFamily: 'monospace', fontSize: '12px', fontWeight: '600', color: g.color }}>{g.logg}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
