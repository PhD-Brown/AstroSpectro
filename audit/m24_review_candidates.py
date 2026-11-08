"""Explicit reconstructions of review R1/R2, policy guards and valid alternatives.

These are not Claude's unpublished 19-case script. The six semantic candidates
are named separately from the original 54 sites; extra guards have their own
group. Controls are never counted as incorrect mutations.
"""

SEMANTIC_MUTATIONS = [
    (
        "cluster_mean_to_median",
        "physical_profile_table",
        'agg(["mean", "std"])',
        'agg(["median", "std"]).rename(columns={"median":"mean"},level=1)',
    ),
    (
        "noise_mean_to_median",
        "physical_profile_table",
        "                    row_noise[label] = (",
        "                    vals = pd.Series([vals.median()])\n                    row_noise[label] = (",
    ),
    (
        "feature_mean_to_median",
        "compute_feature_profiles",
        'groupby("cluster")[feat_cols].mean()',
        'groupby("cluster")[feat_cols].median()',
    ),
    (
        "returned_rows_reversed",
        "compute_feature_profiles",
        "return df_feat, cluster_means, cluster_labels_aligned",
        "return df_feat.iloc[::-1].reset_index(drop=True), cluster_means, cluster_labels_aligned",
    ),
    (
        "returned_columns_sorted_independently",
        "compute_feature_profiles",
        "return df_feat, cluster_means, cluster_labels_aligned",
        "return df_feat.apply(np.sort), cluster_means, cluster_labels_aligned",
    ),
    (
        "drop_head_instead_of_nan_rows",
        "compute_feature_profiles",
        "df_filt[feat_cols].dropna().reset_index(drop=True)",
        "df_filt[feat_cols].iloc[df_filt[feat_cols].isna().any(axis=1).sum():].fillna(0).reset_index(drop=True)",
    ),
]

EXTRA_MUTATIONS = [
    (
        "undefined_dispersion_becomes_zero",
        "physical_profile_table",
        "sig = summary.loc[cid, s_col]",
        "sig = 0.0 if pd.isna(summary.loc[cid, s_col]) else summary.loc[cid, s_col]",
    ),
    (
        "returned_frame_mixes_raw_and_standardized_columns",
        "compute_feature_profiles",
        "return df_feat, cluster_means, cluster_labels_aligned",
        "return df_feat.assign(**{feat_cols[0]: X_std_df[feat_cols[0]]}), cluster_means, cluster_labels_aligned",
    ),
]

REVIEW_CONTROLS = [
    (
        "undefined_dispersion_dash",
        "physical_profile_table",
        'f"{mu:.2f} ± {sig:.2f}"',
        '(f"{mu:.2f} ± —" if pd.isna(sig) else f"{mu:.2f} ± {sig:.2f}")',
    ),
    (
        "undefined_dispersion_na",
        "physical_profile_table",
        'f"{mu:.2f} ± {sig:.2f}"',
        '(f"{mu:.2f} ± N/A" if pd.isna(sig) else f"{mu:.2f} ± {sig:.2f}")',
    ),
    (
        "missing_physics_prose",
        "physical_profile_table",
        '"paramètres Gaia insuffisants"',
        '"No supported physical classification from the available columns"',
    ),
    ("alternative_colormap", "_build_palette", '"turbo"', '"viridis"'),
    (
        "presentation_prediction_data",
        "fit_presentation",
        "cluster_selection_method=self.cluster_selection_method,",
        "cluster_selection_method=self.cluster_selection_method,\n            prediction_data=True,",
    ),
    (
        "missing_physics_domain_error",
        "physical_profile_table",
        "col_names = [c for c, _ in available]",
        'col_names = [c for c, _ in available]\n        if not col_names:\n            raise ValueError("No physical columns available")',
    ),
    (
        "missing_class_placeholder",
        "physical_profile_table",
        ".apply(lambda x: x.value_counts().index[0])",
        '.apply(lambda x: x.value_counts().index[0] if len(x.value_counts()) else "?")',
    ),
    (
        "noise_prose_qualified",
        "physical_profile_table",
        '"anomalies spectrales, objets atypiques"',
        '"Unassigned in the density representation; spectral status unvalidated"',
    ),
    (
        "equivalent_phot_variable_flag",
        "compute_feature_profiles",
        '        "phot_variable_flag",',
        "",
    ),
]


def full_invalidation_control(source):
    return (
        source
        + """
def _audit_guard(method):
    def wrapped(self, *args, **kwargs):
        config = (self.min_cluster_size, self.min_samples, self.metric, self.cluster_selection_method)
        try:
            return method(self, *args, **kwargs)
        except Exception:
            clean = type(self)(*config)
            self.__dict__.clear()
            self.__dict__.update(clean.__dict__)
            raise
    return wrapped
HDBSCANAnalyzer.fit = _audit_guard(HDBSCANAnalyzer.fit)
HDBSCANAnalyzer.fit_presentation = _audit_guard(HDBSCANAnalyzer.fit_presentation)
"""
    )


def uniform_finite_control(source):
    return (
        source
        + """
def _audit_finite(method):
    def wrapped(self, Z, *args, **kwargs):
        if not np.isfinite(Z).all():
            raise ValueError("Coordinates must be finite")
        return method(self, Z, *args, **kwargs)
    return wrapped
HDBSCANAnalyzer.fit = _audit_finite(HDBSCANAnalyzer.fit)
HDBSCANAnalyzer.fit_presentation = _audit_finite(HDBSCANAnalyzer.fit_presentation)
"""
    )
