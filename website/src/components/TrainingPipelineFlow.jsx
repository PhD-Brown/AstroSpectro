import React from 'react';

const PIPELINE_STEPS = [
  {
    step: 1,
    name: 'SimpleImputer',
    detail: 'Stratégie zéro',
    note: 'Valeurs manquantes → 0 (features spectrales physiquement absentes)',
    color: '#38BDF8',
    icon: '∅',
  },
  {
    step: 2,
    name: 'StandardScaler',
    detail: 'z-score sur train uniquement',
    note: 'Ajusté sur X_train exclusivement — aucune fuite vers X_test',
    color: '#F59E0B',
    icon: 'σ',
  },
  {
    step: 3,
    name: 'SMOTE',
    detail: '(optionnel)',
    note: 'Suréchantillonnage synthétique pour classes minoritaires (A, galaxies)',
    color: '#A78BFA',
    icon: '⊕',
    optional: true,
  },
  {
    step: 4,
    name: 'Classificateur',
    detail: 'XGBoost / LightGBM / ...',
    note: 'Modèle final — pipeline complet encapsulé dans imblearn.Pipeline',
    color: '#34D399',
    icon: '▶',
  },
];

const WANDB_CATEGORIES = [
  {
    cat: 'Performance globale',
    color: '#34D399',
    metrics: ['accuracy', 'balanced_accuracy', 'best_roc_auc', 'best_f1_macro'],
  },
  {
    cat: 'Par classe (×5)',
    color: '#38BDF8',
    metrics: ['precision', 'recall', 'f1_score', 'roc_auc', 'avg_precision', 'brier'],
    note: '5 classes A/F/G/K/M → 30 métriques',
  },
  {
    cat: 'Calibration',
    color: '#A78BFA',
    metrics: ['brier/mean', 'ECE (Expected Calibration Error)'],
  },
  {
    cat: 'Dataset',
    color: '#F59E0B',
    metrics: ['n_spectra', 'n_train', 'n_test', 'n_features_used'],
  },
  {
    cat: 'Visuels loggés',
    color: '#FB923C',
    metrics: ['Matrice de confusion', 'Courbes ROC', 'Courbes PR', 'Importance des features'],
  },
];

function PipelineStep({ step, name, detail, note, color, icon, optional }) {
  const [hov, setHov] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        flex: 1,
        background: hov ? `${color}0e` : optional ? 'rgba(255,255,255,0.02)' : 'var(--ifm-card-background-color)',
        border: `1px solid ${hov ? color + '55' : optional ? 'rgba(255,255,255,0.06)' : color + '25'}`,
        borderTop: `3px solid ${optional ? color + '55' : color}`,
        borderRadius: '8px',
        padding: '14px',
        transition: 'all 0.15s',
        cursor: 'default',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        opacity: optional ? 0.75 : 1,
        minWidth: 0,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{
          width: '28px',
          height: '28px',
          borderRadius: '50%',
          background: optional ? `${color}22` : color,
          color: optional ? color : '#000',
          fontWeight: '900',
          fontSize: '13px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: 'monospace',
          flexShrink: 0,
        }}>{icon}</div>
        {optional && (
          <div style={{
            fontSize: '9px',
            fontWeight: '700',
            color: color,
            opacity: 0.7,
            border: `1px solid ${color}40`,
            borderRadius: '4px',
            padding: '1px 5px',
          }}>optionnel</div>
        )}
      </div>

      <div>
        <div style={{ fontFamily: 'monospace', fontWeight: '700', fontSize: '13px', color: color }}>{name}</div>
        <div style={{ fontSize: '10.5px', opacity: 0.5, color: 'var(--ifm-font-color-base)' }}>{detail}</div>
      </div>

      {hov && (
        <div style={{
          fontSize: '11px',
          color: 'var(--ifm-font-color-base)',
          opacity: 0.6,
          lineHeight: 1.5,
          borderTop: `1px solid ${color}18`,
          paddingTop: '6px',
        }}>{note}</div>
      )}
    </div>
  );
}

function Arrow() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'rgba(255,255,255,0.2)',
      fontSize: '18px',
      flexShrink: 0,
      userSelect: 'none',
      paddingTop: '14px',
    }}>→</div>
  );
}

function WandBCategory({ cat, color, metrics, note }) {
  const [hov, setHov] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? `${color}0a` : 'var(--ifm-card-background-color)',
        border: `1px solid ${hov ? color + '40' : 'rgba(255,255,255,0.06)'}`,
        borderLeft: `3px solid ${color}`,
        borderRadius: '7px',
        padding: '10px 14px',
        transition: 'all 0.12s',
        cursor: 'default',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '7px' }}>
        <div style={{ fontWeight: '700', fontSize: '12px', color: color }}>{cat}</div>
        {note && (
          <div style={{ fontSize: '9.5px', opacity: 0.5, color: 'var(--ifm-font-color-base)', fontStyle: 'italic' }}>
            {note}
          </div>
        )}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
        {metrics.map((m) => (
          <code key={m} style={{
            fontSize: '10.5px',
            background: `${color}12`,
            border: `1px solid ${color}25`,
            borderRadius: '4px',
            padding: '1px 6px',
            color: color,
            opacity: 0.85,
          }}>{m}</code>
        ))}
      </div>
    </div>
  );
}

export default function TrainingPipelineFlow() {
  const [tab, setTab] = React.useState('pipeline');

  return (
    <div style={{ margin: '24px 0' }}>
      {/* Tabs */}
      <div style={{
        display: 'flex',
        gap: '4px',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        marginBottom: '0',
      }}>
        {[
          { id: 'pipeline', label: 'Pipeline intégré', color: '#38BDF8' },
          { id: 'wandb',    label: 'Suivi W&B (121 métriques)', color: '#F59E0B' },
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
            transition: 'all 0.13s',
            borderRadius: '5px 5px 0 0',
            marginBottom: '-1px',
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderTop: 'none',
        borderRadius: '0 0 10px 10px',
        padding: '18px',
      }}>
        {tab === 'pipeline' && (
          <>
            <div style={{ fontSize: '11px', opacity: 0.4, marginBottom: '16px', color: 'var(--ifm-font-color-base)' }}>
              imblearn.Pipeline — ajusté sur X_train uniquement · aucune fuite vers X_test · hover pour détails
            </div>
            <div style={{ display: 'flex', gap: '6px', alignItems: 'flex-start', flexWrap: 'wrap' }}>
              {PIPELINE_STEPS.map((s, i) => (
                <React.Fragment key={s.step}>
                  <PipelineStep {...s} />
                  {i < PIPELINE_STEPS.length - 1 && <Arrow />}
                </React.Fragment>
              ))}
            </div>
            <div style={{
              marginTop: '14px',
              padding: '10px 14px',
              background: 'rgba(52,211,153,0.05)',
              border: '1px solid rgba(52,211,153,0.15)',
              borderRadius: '6px',
              fontSize: '11.5px',
              color: 'var(--ifm-font-color-base)',
              opacity: 0.8,
            }}>
              Le pipeline chargé depuis <code>.pkl</code> inclut le scaler et l'imputer — <strong style={{ color: '#34D399' }}>aucune préparation supplémentaire</strong> des données n'est nécessaire pour prédire.
            </div>
          </>
        )}

        {tab === 'wandb' && (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
              <div style={{ fontSize: '11px', opacity: 0.4, color: 'var(--ifm-font-color-base)' }}>
                Projet <code style={{ color: '#F59E0B' }}>astrospectro</code> · 85+ runs enregistrés
              </div>
              <div style={{
                fontFamily: 'monospace',
                fontWeight: '900',
                fontSize: '22px',
                color: '#F59E0B',
                lineHeight: 1,
              }}>121 métriques<span style={{ fontSize: '12px', opacity: 0.5, fontWeight: '400' }}> / run</span></div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {WANDB_CATEGORIES.map((c) => <WandBCategory key={c.cat} {...c} />)}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
