import React from 'react';

const TREE = {
  root: 'data/',
  children: [
    {
      name: 'models/',
      color: '#F59E0B',
      children: [
        {
          name: 'spectral_classifier_xgboost_<ISO>.pkl',
          color: '#34D399',
          desc: 'Pipeline complet (scaler + imputer + modèle)',
          size: '~12 MB',
          badge: 'Principal',
        },
      ],
    },
    {
      name: 'reports/<ISO>/',
      color: '#38BDF8',
      children: [
        {
          name: 'spectral_classifier_xgboost_<ISO>_meta.json',
          color: '#38BDF8',
          desc: 'Hyperparamètres, features, classes, reproductibilité',
          badge: 'Méta',
        },
        {
          name: 'confusion_matrix.png',
          color: '#A78BFA',
          desc: 'Matrice de confusion normalisée',
          badge: 'Figure',
        },
      ],
    },
  ],
};

const META_FIELDS = [
  { key: 'model_type',          desc: 'Type de classifieur',              color: '#38BDF8' },
  { key: 'prediction_target',   desc: 'Cible de prédiction',              color: '#38BDF8' },
  { key: 'best_params_',        desc: 'Hyperparamètres finaux',           color: '#34D399' },
  { key: 'class_labels',        desc: 'Labels A/F/G/K/M',                 color: '#FBBF24' },
  { key: 'feature_names_used',  desc: 'Liste complète des features entrée', color: '#F59E0B' },
  { key: 'selected_features_',  desc: 'Features retenues après sélection',  color: '#A78BFA' },
  { key: 'trained_on_file',     desc: 'Fichier source (reproductibilité)', color: '#FB923C' },
  { key: 'python_version',      desc: 'Version Python',                   color: '#94A3B8' },
  { key: 'sklearn_version',     desc: 'Version scikit-learn',             color: '#94A3B8' },
  { key: 'xgboost_version',     desc: 'Version XGBoost',                  color: '#94A3B8' },
];

function TreeFile({ name, color, desc, size, badge, depth = 1 }) {
  const [hov, setHov] = React.useState(false);

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '7px 12px',
        marginLeft: `${depth * 20}px`,
        borderLeft: `2px solid rgba(255,255,255,0.06)`,
        background: hov ? `${color}0a` : 'transparent',
        borderRadius: '0 6px 6px 0',
        transition: 'background 0.1s',
        cursor: 'default',
      }}
    >
      <span style={{ fontSize: '12px', opacity: 0.35, flexShrink: 0 }}>└─</span>
      <span style={{ fontSize: '12px' }}>📄</span>
      <code style={{
        fontSize: '11.5px',
        color: hov ? color : 'var(--ifm-font-color-base)',
        opacity: hov ? 1 : 0.8,
        flex: 1,
        minWidth: 0,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        transition: 'color 0.1s',
      }}>{name}</code>

      {badge && (
        <div style={{
          background: `${color}18`,
          border: `1px solid ${color}35`,
          borderRadius: '4px',
          padding: '1px 7px',
          fontSize: '10px',
          fontWeight: '700',
          color: color,
          whiteSpace: 'nowrap',
          flexShrink: 0,
        }}>{badge}</div>
      )}

      {size && (
        <div style={{ fontSize: '10px', opacity: 0.4, fontFamily: 'monospace', flexShrink: 0, color: 'var(--ifm-font-color-base)' }}>
          {size}
        </div>
      )}

      {hov && desc && (
        <div style={{
          fontSize: '10.5px',
          color: color,
          opacity: 0.7,
          whiteSpace: 'nowrap',
          flexShrink: 0,
        }}>— {desc}</div>
      )}
    </div>
  );
}

function TreeDir({ name, color, children, depth = 0 }) {
  return (
    <div>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        marginLeft: `${depth * 20}px`,
      }}>
        <span style={{ fontSize: '14px' }}>📁</span>
        <code style={{ fontWeight: '700', fontSize: '12.5px', color: color }}>{name}</code>
      </div>
      {children?.map((child, i) => (
        child.children
          ? <TreeDir key={i} name={child.name} color={child.color} children={child.children} depth={depth + 1} />
          : <TreeFile key={i} {...child} depth={depth + 1} />
      ))}
    </div>
  );
}

function MetaField({ field, desc, color }) {
  const [hov, setHov] = React.useState(false);
  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '200px 1fr',
        alignItems: 'center',
        gap: '12px',
        padding: '7px 12px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        background: hov ? `${color}0a` : 'transparent',
        transition: 'background 0.1s',
        cursor: 'default',
      }}
    >
      <code style={{
        fontSize: '11.5px',
        color: hov ? color : 'var(--ifm-font-color-base)',
        opacity: hov ? 1 : 0.8,
        transition: 'color 0.1s',
      }}>{field}</code>
      <div style={{ fontSize: '11.5px', opacity: 0.6, color: 'var(--ifm-font-color-base)' }}>{desc}</div>
    </div>
  );
}

export default function ArtifactsTree() {
  return (
    <div style={{ margin: '24px 0', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
      {/* Left: file tree */}
      <div style={{
        background: '#06080f',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        padding: '14px 4px',
        fontFamily: 'monospace',
      }}>
        <div style={{
          padding: '4px 12px 12px',
          fontSize: '10px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          opacity: 0.35,
          color: 'var(--ifm-font-color-base)',
        }}>Structure des artefacts</div>

        <div style={{
          padding: '4px 12px',
          fontSize: '12px',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}>
          <span style={{ fontSize: '14px' }}>📂</span>
          <code style={{ fontWeight: '800', color: '#94A3B8' }}>data/</code>
        </div>

        {TREE.children.map((child, i) => (
          <TreeDir key={i} name={child.name} color={child.color} children={child.children} depth={1} />
        ))}

        <div style={{
          marginTop: '14px',
          padding: '8px 12px',
          borderTop: '1px solid rgba(255,255,255,0.05)',
          fontSize: '10px',
          opacity: 0.35,
          fontFamily: 'monospace',
          color: 'var(--ifm-font-color-base)',
        }}>
          Survoler un fichier pour la description
        </div>
      </div>

      {/* Right: meta.json fields */}
      <div style={{
        background: 'var(--ifm-card-background-color)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '10px',
        overflow: 'hidden',
      }}>
        <div style={{
          padding: '10px 14px',
          background: 'rgba(56,189,248,0.06)',
          borderBottom: '1px solid rgba(56,189,248,0.15)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <span>📋</span>
          <div style={{ fontWeight: '700', fontSize: '12px', color: '#38BDF8' }}>
            Contenu du fichier <code style={{ background: 'transparent', fontSize: 'inherit' }}>_meta.json</code>
          </div>
        </div>
        {META_FIELDS.map((f) => (
          <MetaField key={f.key} field={f.key} desc={f.desc} color={f.color} />
        ))}
      </div>
    </div>
  );
}
