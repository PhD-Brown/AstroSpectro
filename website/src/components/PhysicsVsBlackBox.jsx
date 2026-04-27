import React from 'react';

const APPROACHES = [
  {
    name: 'Boîte Noire (Deep Learning brut)',
    subtitle: 'Non retenu pour AstroSpectro',
    color: '#F87171',
    icon: '◻',
    adopted: false,
    pros: [
      'Aucune connaissance préalable requise',
      'Peut capturer des patterns non connus',
    ],
    cons: [
      'Millions de paramètres à entraîner',
      'Nécessite ~10⁵–10⁶ spectres annotés',
      'Résultats difficiles à valider physiquement',
      'Sensible au bruit instrumental non modélisé',
      '"Pourquoi ce type A ?" — pas de réponse',
    ],
    example: 'CNN sur les 3921 canaux de flux bruts',
  },
  {
    name: 'Physics-Informed + ML classique',
    subtitle: 'Approche retenue — AstroSpectro V2',
    color: '#34D399',
    icon: '◆',
    adopted: true,
    pros: [
      '183 descripteurs physiques interprétables',
      'SHAP valide les décisions physiquement',
      '43 019 spectres suffisent — pas besoin de millions',
      'Robuste au bruit (features locales, pas pixels)',
      'Chaque décision expliquée en termes astrophysiques',
      '87 % balanced accuracy en ~5 min d\'entraînement',
    ],
    cons: [
      'Expertise astrophysique requise pour définir les raies',
      'Peut manquer des patterns non connus',
    ],
    example: 'XGBoost sur 183 descripteurs (W(Hα), Ca II prominence…)',
  },
];

const EVOLUTION = [
  { step: 1, label: 'Baseline',         acc: 32,  features: '4 features binaires',        model: 'Random Forest', color: '#94A3B8', note: 'Preuve de concept' },
  { step: 2, label: 'V1 — Raies',       acc: 50,  features: '~20 features physiques',      model: 'Random Forest', color: '#F59E0B', note: 'Prominences + FWHM' },
  { step: 3, label: 'V1 + Métadonnées', acc: 75,  features: '~25 features + metadata',     model: 'Random Forest', color: '#F59E0B', note: 'ra, dec, redshift ajoutés' },
  { step: 4, label: 'V1 + XGBoost',    acc: 84,  features: '~25 features + metadata',     model: 'XGBoost',       color: '#F59E0B', note: 'GridSearchCV tuning', warning: '⚠ fuite de données' },
  { step: 5, label: 'V2 spectro_only', acc: 87,  features: '183 features spectrales pures', model: 'XGBoost',       color: '#34D399', note: 'Physique intrinsèque uniquement', current: true },
];

function ApproachCard({ name, subtitle, color, icon, adopted, pros, cons, example }) {
  return (
    <div style={{
      flex: 1,
      background: adopted ? `${color}06` : 'rgba(255,255,255,0.02)',
      border: `2px solid ${adopted ? color + '40' : 'rgba(255,255,255,0.07)'}`,
      borderTop: `4px solid ${adopted ? color : 'rgba(248,113,113,0.4)'}`,
      borderRadius: '10px',
      padding: '20px',
      position: 'relative',
      minWidth: 0,
    }}>
      {adopted && (
        <div style={{
          position: 'absolute',
          top: '-1px',
          right: '16px',
          background: color,
          color: '#000',
          fontSize: '9px',
          fontWeight: '800',
          padding: '2px 8px',
          borderRadius: '0 0 5px 5px',
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}>RETENU</div>
      )}

      <div style={{ marginBottom: '14px' }}>
        <div style={{
          fontWeight: '800',
          fontSize: '14px',
          color: color,
          marginBottom: '3px',
          lineHeight: 1.3,
        }}>{name}</div>
        <div style={{ fontSize: '11px', opacity: 0.55, color: 'var(--ifm-font-color-base)', fontStyle: 'italic' }}>
          {subtitle}
        </div>
      </div>

      {/* Example */}
      <div style={{
        padding: '8px 12px',
        background: `${color}0c`,
        border: `1px solid ${color}25`,
        borderRadius: '6px',
        marginBottom: '14px',
      }}>
        <div style={{ fontSize: '10px', opacity: 0.45, marginBottom: '3px', color: 'var(--ifm-font-color-base)' }}>Exemple concret</div>
        <div style={{ fontFamily: 'monospace', fontSize: '11.5px', color: color, opacity: 0.85 }}>{example}</div>
      </div>

      {/* Pros */}
      <div style={{ marginBottom: '10px' }}>
        <div style={{ fontSize: '10px', fontWeight: '700', color: '#34D399', opacity: 0.8, marginBottom: '5px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          ✓ Avantages
        </div>
        {pros.map((p, i) => (
          <div key={i} style={{
            display: 'flex', gap: '6px', fontSize: '11.5px',
            color: 'var(--ifm-font-color-base)', opacity: 0.75,
            lineHeight: 1.5, marginBottom: '3px',
          }}>
            <span style={{ color: '#34D399', flexShrink: 0 }}>›</span>
            {p}
          </div>
        ))}
      </div>

      {/* Cons */}
      <div>
        <div style={{ fontSize: '10px', fontWeight: '700', color: '#F87171', opacity: 0.7, marginBottom: '5px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          ✗ Limites
        </div>
        {cons.map((c, i) => (
          <div key={i} style={{
            display: 'flex', gap: '6px', fontSize: '11.5px',
            color: 'var(--ifm-font-color-base)', opacity: 0.6,
            lineHeight: 1.5, marginBottom: '3px',
          }}>
            <span style={{ color: '#F87171', flexShrink: 0 }}>›</span>
            {c}
          </div>
        ))}
      </div>
    </div>
  );
}

function EvolutionStep({ step, label, acc, features, model, color, note, warning, current }) {
  const [hov, setHov] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '32px 140px 60px 1fr 120px',
        alignItems: 'center',
        gap: '12px',
        padding: '10px 14px',
        background: hov
          ? `${color}0e`
          : current ? `${color}06` : 'transparent',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        borderLeft: current ? `3px solid ${color}` : '3px solid transparent',
        transition: 'background 0.12s',
        cursor: 'default',
      }}
    >
      {/* Step */}
      <div style={{
        width: '24px',
        height: '24px',
        borderRadius: '50%',
        background: current ? color : `${color}30`,
        border: `2px solid ${color}60`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '11px',
        fontWeight: '800',
        color: current ? '#000' : color,
        flexShrink: 0,
      }}>{step}</div>

      {/* Label */}
      <div style={{
        fontSize: '12.5px',
        fontWeight: current ? '700' : '500',
        color: current ? color : 'var(--ifm-font-color-base)',
        lineHeight: 1.3,
      }}>
        {label}
        {warning && (
          <div style={{ fontSize: '10px', color: '#F59E0B', fontWeight: '400', marginTop: '1px' }}>
            {warning}
          </div>
        )}
      </div>

      {/* Accuracy big */}
      <div style={{
        fontFamily: 'monospace',
        fontSize: '20px',
        fontWeight: '900',
        color: color,
        textAlign: 'center',
      }}>{acc}%</div>

      {/* Features */}
      <div style={{ fontSize: '11.5px', color: 'var(--ifm-font-color-base)', opacity: 0.65 }}>
        {features}
      </div>

      {/* Model + note */}
      <div>
        <div style={{ fontFamily: 'monospace', fontSize: '11px', color: color, opacity: 0.8 }}>{model}</div>
        <div style={{ fontSize: '10px', opacity: 0.45, marginTop: '1px', color: 'var(--ifm-font-color-base)' }}>{note}</div>
      </div>
    </div>
  );
}

export default function PhysicsVsBlackBox() {
  const [tab, setTab] = React.useState('approach');

  return (
    <div style={{ margin: '28px 0' }}>
      {/* Tabs */}
      <div style={{
        display: 'flex',
        gap: '4px',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        marginBottom: '0',
      }}>
        {[
          { id: 'approach', label: 'Philosophie Physics-Informed', color: '#34D399' },
          { id: 'evolution', label: 'Évolution 32 % → 87 %',         color: '#F59E0B' },
        ].map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: tab === t.id ? `${t.color}12` : 'transparent',
            border: 'none',
            borderBottom: `2px solid ${tab === t.id ? t.color : 'transparent'}`,
            padding: '9px 16px',
            cursor: 'pointer',
            fontSize: '12px',
            fontWeight: tab === t.id ? '700' : '400',
            color: tab === t.id ? t.color : 'var(--ifm-font-color-base)',
            opacity: tab === t.id ? 1 : 0.5,
            borderRadius: '5px 5px 0 0',
            marginBottom: '-1px',
            transition: 'all 0.13s',
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderTop: 'none',
        borderRadius: '0 0 10px 10px',
        padding: '20px',
      }}>
        {tab === 'approach' && (
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            {APPROACHES.map((a) => <ApproachCard key={a.name} {...a} />)}
          </div>
        )}

        {tab === 'evolution' && (
          <>
            <div style={{ fontSize: '11px', opacity: 0.4, marginBottom: '14px', color: 'var(--ifm-font-color-base)' }}>
              Survoler chaque étape · ★ = version actuelle de référence
            </div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '32px 140px 60px 1fr 120px',
              gap: '12px',
              padding: '4px 14px 8px',
              borderBottom: '1px solid rgba(255,255,255,0.06)',
            }}>
              {['#', 'Étape', 'BA', 'Features', 'Modèle'].map((h) => (
                <div key={h} style={{ fontSize: '9.5px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', opacity: 0.35, color: 'var(--ifm-font-color-base)' }}>{h}</div>
              ))}
            </div>
            {EVOLUTION.map((e) => <EvolutionStep key={e.step} {...e} />)}
            <div style={{
              marginTop: '12px',
              padding: '10px 14px',
              background: 'rgba(248,113,113,0.06)',
              border: '1px solid rgba(248,113,113,0.2)',
              borderRadius: '7px',
              fontSize: '12px',
              color: 'var(--ifm-font-color-base)',
              opacity: 0.8,
            }}>
              <strong style={{ color: '#F87171' }}>Contre-intuitif :</strong> la progression 84 % → 87 % s'obtient en <em>supprimant</em> des features. <code>ra</code>, <code>dec</code>, <code>redshift</code> encodaient les biais observationnels de LAMOST, pas la physique stellaire.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
