"""Temporary-source candidates for PR118 review; none is a production patch."""

import ast
import difflib
import textwrap


def method_span(source, name):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    node = next(n for n in cls.body if getattr(n, "name", None) == name)
    lines = source.splitlines(keepends=True)
    return sum(map(len, lines[: node.lineno - 1])), sum(
        map(len, lines[: node.end_lineno])
    )


def edit(source, method, old, new, count=1):
    start, end = method_span(source, method)
    body = source[start:end]
    assert body.count(old) == count, (method, old, body.count(old), count)
    return source[:start] + body.replace(old, new) + source[end:]


def replace_method(source, name, code):
    start, end = method_span(source, name)
    return (
        source[:start]
        + textwrap.indent(textwrap.dedent(code).strip() + "\n", "    ")
        + source[end:]
    )


PREP = """
def _prepare_for_model(self, df, clf_wrapper):
    used = list(getattr(clf_wrapper, "feature_names_used", list(df.columns)))
    X = df.reindex(columns=used).copy()
    names = used
    for name, step in clf_wrapper.model_pipeline.steps:
        if name in {"clf", "rf"}:
            break
        if name == "sampler":
            continue
        X = step.transform(X)
        if name == "colfilter":
            names = list(step.keep_columns_)
        elif hasattr(step, "get_support"):
            names = list(np.asarray(names)[step.get_support()])
        elif hasattr(step, "get_feature_names_out"):
            names = [str(s).split("__", 1)[-1] for s in step.get_feature_names_out(names)]
    if len(names) != X.shape[1]:
        raise ValueError("Transformed feature identities unavailable")
    return np.asarray(X), names
"""

SPLIT = """                X_train, X_test, y_train, y_test = train_test_split(
                    X_num,
                    y,
                    test_size=0.25,
                    random_state=random_state,
                    stratify=stratify_vec,
                )"""

GROUP_SPLIT = """                if "source_id" not in df:
                    raise ValueError("No independent holdout: object identities missing")
                from sklearn.model_selection import GroupShuffleSplit
                splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=random_state)
                train_rows, test_rows = next(splitter.split(X_num, y, groups=df["source_id"]))
                X_train, X_test = X_num.iloc[train_rows], X_num.iloc[test_rows]
                y_train, y_test = y.iloc[train_rows], y.iloc[test_rows]"""


def changed_lines(before, after):
    return sorted(
        {
            line + 1
            for op, a, b, c, d in difflib.SequenceMatcher(
                a=before.splitlines(), b=after.splitlines()
            ).get_opcodes()
            if op != "equal"
            for line in range(c, d)
        }
    )


def candidates(source, historical, local_style):
    """Return scoped mutations and bounded alternatives with declared scope."""
    out = []

    def add(
        name,
        kind,
        code,
        target,
        *,
        scope="bounded contract only",
        reference=None,
        sites=1,
    ):
        compile(code, name, "exec")
        before = source if reference is None else reference
        out.append(
            dict(
                id=name,
                kind=kind,
                source=code,
                target=target,
                scope=scope,
                reference=reference,
                changed_lines=changed_lines(before, code),
                sites=sites,
                source_sha_scope="visualizer.py copy only",
            )
        )

    initial = {item["id"]: item for item in historical["adversarial"]}
    method_for_mutant = {
        "M01": "_plot_spectrum_analysis",
        "M02": "_plot_spectra_comparison",
        "M03": "plot_subclass_distribution",
        "M04": "_run_shap_analysis",
        "M05": "_prepare_for_model",
        "M06": "_analyze_saved_model",
        "M07": "analyze_feature_zeros",
        "M08": "plot_sky_coverage",
    }
    expressions = {
        "M01": "finite_gaussian",
        "M02": "comparison_scale_invariance",
        "M03": "subclass_raw_counts",
        "M04": "binary_shap_mean",
        "M05": "prepare_flat",
        "M06": "model_inspector_selected",
        "M07": "zero_statistics",
        "M08": "sky_artists",
    }
    for name, item in initial.items():
        if not name.startswith("M"):
            continue
        key = name[:3]
        s = edit(source, method_for_mutant[key], item["site"], item["replacement"])
        sites = 1
        if key == "M01":
            s = edit(
                s, "_plot_peak_detection_notebook", item["site"], item["replacement"]
            )
            sites = 2
        add(name, "mutant_green", s, expressions[key], sites=sites)
    mapping = {
        "A01": "_generate_position_catalog",
        "A02": "plot_sky_coverage",
        "A03": "plot_subclass_distribution",
        "A05": "_run_shap_analysis",
        "A06": "_prepare_for_model",
        "A07": "_plot_peak_detection_notebook",
        "A08": "_prepare_for_model",
        "A09": "_run_shap_analysis",
        "A10": "_get_features_sample",
        "A12": "_plot_peak_detection_notebook",
        "A13": "_generate_position_catalog",
        "A14": "_plot_spectra_comparison",
        "A15": "_load_labels_catalog",
    }
    targets = {
        "A01": "ra_wrap_centroid",
        "A02": "sky_ra_tick",
        "A03": "truncated_percentages",
        "A05": "multiclass_shap",
        "A06": "selected_feature_names_derive",
        "A07": "normalized_flux_uncertainties",
        "A08": "actual_classifier_prep",
        "A09": "computed_once",
        "A10": "missing_measurement",
        "A12": "absorption_fwhm_export",
        "A13": "position_cache",
        "A14": "duplicate_basename_legend",
        "A15": "duplicate_conflicting_catalog",
    }
    for name, item in initial.items():
        key = name[:3]
        if key not in mapping:
            continue
        s = edit(source, mapping[key], item["site"], item["replacement"])
        scope = "bounded contract; not a production-ready remedy"
        if key == "A03":
            s = edit(
                s,
                "plot_subclass_distribution",
                'ax.set_title("Spectral Sub-class Distribution")',
                'ax.set_title(f"Frequency of total catalogue; total={len(df)}; displayed={int(counts.sum())}; excluded={len(df)-int(counts.sum())}; denominator=total" if normalize else "Spectral Sub-class Distribution")',
            )
        if key == "A02":
            s = edit(
                s,
                "plot_sky_coverage",
                "RA vers la gauche, centre 0h/24h",
                "RA vers la gauche, centre 12h",
            )
        if key == "A08":
            scope = "NON_GENERALIZABLE historical prep-only variant; expected rejection by real pipeline names/selection"
        if key == "A15":
            scope = "NON_GENERALIZABLE historical all-duplicates refusal; expected regression on identical duplicates"
        if key == "A13":
            scope = "CACHE_BYPASS, not evidence of correct invalidation"
        if key == "A01":
            scope = "symmetric short arc only; antipodal policy not resolved"
        if key == "A10":
            scope = "missing-value preservation only; does not repair the downstream preparation (R17)"
        add(name, "remedy", s, targets[key], scope=scope)
    dynamic = edit(
        source,
        "plot_subclass_distribution",
        'ax.set_title("Spectral Sub-class Distribution")',
        'ax.set_title(f"Conditional frequency; displayed={int(counts.sum())}; total={len(df)}; excluded={len(df)-int(counts.sum())}; denominator=displayed" if normalize else "Spectral Sub-class Distribution")',
    )
    add(
        "A04_dynamic_conditional",
        "remedy",
        dynamic,
        "truncated_percentages",
        scope="computed counts for both populations",
    )
    add("A11_local_style", "remedy", local_style(source), "restores_global_matplotlib")
    s = source
    for method in ["_plot_spectrum_analysis", "_plot_peak_detection_notebook"]:
        s = edit(
            s,
            method,
            "peak_wl = wavelength[peak_idx]",
            "peak_idx = np.flatnonzero(np.isfinite(wavelength) & np.isfinite(flux_norm))[peak_idx]\n            peak_wl = wavelength[peak_idx]",
        )
    s = edit(
        s,
        "_plot_spectrum_analysis_plotly",
        "peak_wavelengths = wavelength[peak_indices]",
        "peak_indices = np.flatnonzero(np.isfinite(wavelength) & np.isfinite(flux_norm))[peak_indices]\n            peak_wavelengths = wavelength[peak_indices]",
    )
    for method in ["analyze_feature_zeros", "_get_features_sample"]:
        old = (
            "matched = detector.match_known_lines(peak_idx, wl[peak_idx], props)"
            if method == "analyze_feature_zeros"
            else "matched = pk.match_known_lines(peaks, wl[peaks], props)"
        )
        new = (
            "peak_idx = np.flatnonzero(np.isfinite(wl) & np.isfinite(fl))[peak_idx]\n                "
            + old
            if method == "analyze_feature_zeros"
            else "peaks = np.flatnonzero(np.isfinite(wl) & np.isfinite(fl))[peaks]\n                "
            + old
        )
        s = edit(s, method, old, new)
    add(
        "A16_all_pixel_consumers",
        "remedy",
        s,
        "nonfinite_prefix or feature_consumers",
        sites=5,
    )
    prep = replace_method(source, "_prepare_for_model", PREP)
    add(
        "A17_real_preparation",
        "remedy",
        prep,
        "actual_classifier_prep or selected_feature_names_derive",
        scope="real colfilter/prep/fs and flat transforms; PCA/samplers not validated",
    )
    joint = edit(prep, "_get_features_sample", "df[c] = 0.0", "df[c] = np.nan")
    add(
        "A17_joint_missing_and_preparation",
        "remedy",
        joint,
        "missing_csv_features or missing_measurement or actual_classifier_prep",
        sites=2,
    )
    group = edit(source, "feature_explorer", SPLIT, GROUP_SPLIT)
    group = edit(
        group,
        "feature_explorer",
        "Calcule aussi la permutation importance (validation split).",
        "Compute permutation importance using group-disjoint object identities; refuse when identities are missing.",
    )
    add(
        "A18_group_shuffle_split",
        "remedy",
        group,
        "repeat_observations",
        scope="group-disjoint evaluation with known IDs; explicit refusal without IDs",
    )
    descriptive = edit(
        source,
        "feature_explorer",
        "Permutation importance (split de validation)",
        "Permutation importance (descriptive only; object identities may repeat)",
    )
    descriptive = edit(
        descriptive,
        "feature_explorer",
        "Calcule aussi la permutation importance (validation split).",
        "Compute descriptive permutation importance only; object identities may repeat.",
    )
    descriptive = edit(
        descriptive,
        "feature_explorer",
        "display(perm_df.head(30))",
        'perm_df["descriptive_context"] = "Descriptive only; object identities may repeat"\n                        display(perm_df.head(30))',
    )
    descriptive = edit(
        descriptive,
        "feature_explorer",
        '"permutation_importances.csv"',
        '"descriptive_permutation_importances.csv"',
    )
    add(
        "A19_descriptive_limit",
        "remedy",
        descriptive,
        "repeat_observations",
        scope="descriptive computation with explicit non-independence, not a repaired split",
        sites=4,
    )
    contradiction = edit(
        source,
        "feature_explorer",
        "pipe.fit(X_train, y_train)",
        'pipe.fit(X_train, y_train)\n                print("Note: descriptive EDA, not an independent holdout.")',
    )
    add(
        "V_contradict",
        "remedy",
        contradiction,
        "repeat_observations",
        scope="Claude cosmetic warning counterexample; unchanged row split and contradictory exposed surfaces",
    )
    row_split = edit(
        group,
        "feature_explorer",
        GROUP_SPLIT,
        "                X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.25, random_state=random_state, stratify=stratify_vec)",
    )
    row_split = edit(
        row_split,
        "feature_explorer",
        "pipe.fit(X_train, y_train)",
        'pipe.fit(X_train, y_train)\n                print("Note: descriptive EDA, not an independent holdout.")',
    )
    add(
        "V_contradict_mutant",
        "mutant_red",
        row_split,
        "repeat_observations",
        reference=group,
        sites=2,
        scope="admissible grouped baseline changed to row split plus cosmetic warning",
    )
    identity = next(item["source"] for item in out if item["id"].startswith("A14"))
    swapped = edit(
        identity,
        "_plot_spectra_comparison",
        "label=rel_path",
        "label=(next(r for r in file_paths if r != rel_path and os.path.basename(r) == os.path.basename(rel_path)) if sum(os.path.basename(r) == os.path.basename(rel_path) for r in file_paths) > 1 else rel_path)",
    )
    add(
        "V_swapped_identity",
        "mutant_red",
        swapped,
        "duplicate_basename_legend",
        reference=identity,
        scope="Claude swapped curve labels on basename collision",
    )
    add(
        "A20_missing_subclass",
        "remedy",
        edit(
            source,
            "plot_subclass_distribution",
            "# 3) Count & select top N",
            'missing = int(df[label_col].isna().sum())\n        if missing:\n            print(f"Excluded {missing} missing subclass labels")\n            df = df.dropna(subset=[label_col])\n        # 3) Count & select top N',
        ),
        "missing_subclass",
    )
    old = "os.makedirs(os.path.dirname(save_path), exist_ok=True)"
    assert source.count(old) == 6
    add(
        "A21_six_basename_exports",
        "remedy",
        source.replace(
            old, 'os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)'
        ),
        "basename_export",
        sites=6,
    )
    add(
        "A22_no_latent_uncertainty",
        "remedy",
        edit(
            source,
            "_plot_peak_detection_notebook",
            "uncertainty=StdDevUncertainty(sigma),",
            "",
        ),
        "normalized_flux_uncertainties",
        scope="no uncertainty advertised or attached; exported point metrics only",
    )
    add(
        "A23_inspector_refusal",
        "remedy",
        edit(
            source,
            "_analyze_saved_model",
            "feature_names_for_plot = used_cols[: importances.shape[0]]",
            'display(Markdown("Feature identities unavailable after selection"))\n                    return',
        ),
        "model_inspector_missing_selection_metadata",
    )
    add(
        "A24_mode_refusal",
        "remedy",
        edit(
            source,
            "_run_shap_analysis",
            "# 0) Load the model",
            'if explainer_mode not in {"auto", "tree"}:\n            raise ValueError("SHAP explainer mode unavailable")\n        # 0) Load the model',
        ),
        "explainer_mode",
    )
    add(
        "A25_nonfinite_refusal",
        "remedy",
        edit(
            source,
            "feature_explorer",
            "num_df = df.select_dtypes(include=[np.number])",
            'num_df = df.select_dtypes(include=[np.number])\n        invalid = [name for name in num_df if np.isinf(num_df[name]).any()]\n        if invalid:\n            raise ValueError("Nonfinite features: " + ", ".join(invalid))',
        ),
        "nonfinite_histogram",
    )
    moment = next(item["source"] for item in out if item["id"].startswith("A12"))
    interpolation = edit(
        source,
        "_plot_peak_detection_notebook",
        "gaussian_fwhm(spec_full, regions=region)",
        '__import__("specutils.analysis", fromlist=["fwhm"]).fwhm(Spectrum(spectral_axis=wavelength*u.AA, flux=(1-flux_norm)*u.adu), regions=region)',
    )
    add(
        "A26_interpolated_fwhm",
        "remedy",
        interpolation,
        "absorption_fwhm_export",
        scope="interpolated half-height crossings on known flat continuum",
    )
    shapsum = edit(
        source,
        "_run_shap_analysis",
        "vals = np.mean(np.abs(vals), axis=1)",
        "vals = np.abs(vals).sum(axis=2)",
    )
    shapsum = edit(
        shapsum,
        "_run_shap_analysis",
        "# Cache for plotting helpers",
        'df_out.attrs["class_reduction"] = "sum" if np.asarray(sv.values).ndim == 3 else "mean"\n        # Cache for plotting helpers',
    )
    add(
        "A27_declared_class_sum",
        "remedy",
        shapsum,
        "multiclass_shap",
        scope="historical contradictory sum under mean name; negative control, must be rejected",
    )
    correct_sum = edit(
        shapsum,
        "_run_shap_analysis",
        "# Cache for plotting helpers",
        'if np.asarray(sv.values).ndim == 3:\n            df_out = df_out.rename(columns={"mean_abs_shap": "mean_sample_sum_class_abs_shap"})\n        # Cache for plotting helpers',
    )
    correct_sum = edit(
        correct_sum,
        "plot_shap_summary_bar",
        'x_vals = top["mean_abs_shap"].astype(float).iloc[::-1]',
        'column = "mean_sample_sum_class_abs_shap" if "mean_sample_sum_class_abs_shap" in top else "mean_abs_shap"\n        x_vals = top[column].astype(float).iloc[::-1]',
    )
    correct_sum = edit(
        correct_sum,
        "plot_shap_summary_bar",
        'ax.set_xlabel("mean(|SHAP value|)")',
        'ax.set_xlabel("mean_sample_sum_class_abs_SHAP" if column == "mean_sample_sum_class_abs_shap" else "mean(|SHAP value|)")',
    )
    correct_sum = edit(
        correct_sum,
        "plot_shap_summary_bar",
        "with a `mean(|SHAP value|)` scale.",
        "with the sample-mean class-sum or mean absolute SHAP scale declared by the statistic column.",
    )
    add(
        "A27_correctly_named_and_plotted",
        "remedy",
        correct_sum,
        "multiclass_shap",
        sites=6,
        scope="class sum: coherent numeric table, metadata and summary bars on bounded fixture; widget not validated",
    )
    mean = next(item["source"] for item in out if item["id"].startswith("A05"))
    add(
        "V_shap_feature_names",
        "mutant_red",
        edit(
            mean,
            "_run_shap_analysis",
            "# Cache for plotting helpers",
            'df_out["feature"] = df_out["feature"].replace({"alpha": "beta", "beta": "alpha"})\n        # Cache for plotting helpers',
        ),
        "multiclass_shap",
        reference=mean,
    )
    add(
        "V_shap_feature_axis",
        "mutant_red",
        edit(
            mean,
            "_run_shap_analysis",
            "vals = np.mean(np.abs(vals), axis=2)",
            "vals = np.mean(np.abs(vals[:, [1, 0, 2], :]), axis=2)",
        ),
        "multiclass_shap",
        reference=mean,
    )
    add(
        "V_shap_sum_axis_mislabeled",
        "mutant_red",
        edit(
            correct_sum,
            "plot_shap_summary_bar",
            'ax.set_xlabel("mean_sample_sum_class_abs_SHAP" if column == "mean_sample_sum_class_abs_shap" else "mean(|SHAP value|)")',
            'ax.set_xlabel("mean(|SHAP value|)")',
        ),
        "multiclass_shap",
        reference=correct_sum,
    )
    sky = edit(source, "plot_sky_coverage", "x = np.pi - lon", "x = -lon")
    sky = edit(sky, "plot_sky_coverage", "((L % 360) / 15)", "(((-L) % 360) / 15)")
    add(
        "CM7_sky_alternate_convention",
        "remedy",
        sky,
        "sky_ra_tick",
        sites=2,
        scope="centre 0h, RA leftward, coherent with existing docstring",
    )
    conflict = edit(
        source,
        "_load_labels_catalog",
        'labels = pd.Series(df_cat["subclass"].values, index=key_series).to_dict()',
        'pairs = pd.DataFrame({"key": key_series.values, "label": df_cat["subclass"].values}).drop_duplicates()\n            conflicts = pairs.loc[pairs["key"].duplicated(), "key"].tolist()\n            if conflicts:\n                print("Conflicting catalogue identities")\n                pairs = pairs[~pairs["key"].isin(conflicts)]\n            labels = dict(zip(pairs["key"], pairs["label"]))',
    )
    add(
        "CM8_conflicts_only",
        "remedy",
        conflict,
        "duplicate_conflicting_catalog",
        scope="identical duplicates retained; only ambiguous identities excluded",
    )
    rng = edit(
        source,
        "analyze_feature_zeros",
        "random.seed(random_state)",
        "rng = random.Random(random_state)",
    )
    rng = edit(
        rng,
        "analyze_feature_zeros",
        "random.sample(files, k=n_files)",
        "rng.sample(files, k=n_files)",
    )
    rng = edit(
        rng,
        "plot_normalization_comparison",
        "if random_state is not None:\n                random.seed(random_state)",
        "rng = random.Random(random_state)",
    )
    rng = edit(
        rng,
        "plot_normalization_comparison",
        "random.sample(pool, k=max(2, min(n_samples, len(pool))))",
        "rng.sample(pool, k=max(2, min(n_samples, len(pool))))",
    )
    add("V_rng_local", "remedy", rng, "preserves_global_random_state", sites=4)
    normal = edit(
        source,
        "_plot_spectra_comparison",
        "plotted = 0",
        "plotted = 0\n        normalization_ok = True",
    )
    normal = edit(
        normal,
        "_plot_spectra_comparison",
        "flux = preprocessor.normalize_spectrum(flux)",
        "normalization_ok = normalization_ok and np.isfinite(np.median(flux)) and np.median(flux) > 0\n                    flux = preprocessor.normalize_spectrum(flux)",
    )
    normal = edit(
        normal,
        "_plot_spectra_comparison",
        '"Flux (normalised units)" if normalize else "Flux (adu)"',
        '"Flux (normalised units)" if normalize and normalization_ok else "Raw or mixed flux"',
    )
    add("V_normalization_label", "remedy", normal, "failed_normalization", sites=3)
    add(
        "CM1_notebook_no_normalization",
        "mutant_green",
        edit(
            source,
            "_plot_peak_detection_notebook",
            "flux_norm = preprocessor.normalize_spectrum(flux)",
            "flux_norm = flux",
        ),
        "finite_gaussian and notebook",
    )
    # FWHM mutants are compared to a physically correct baseline, not red production.
    correct_expr = "gaussian_fwhm(Spectrum(spectral_axis=wavelength*u.AA, flux=(1-flux_norm)*u.adu), regions=region)"
    add(
        "CM2_constant_fwhm",
        "mutant_red",
        edit(moment, "_plot_peak_detection_notebook", correct_expr, "(1.883856*u.AA)"),
        "absorption_fwhm_export",
        reference=moment,
    )
    add(
        "CM3_biased_fwhm",
        "mutant_red",
        edit(
            moment,
            "_plot_peak_detection_notebook",
            correct_expr,
            "(1.10*" + correct_expr + ")",
        ),
        "absorption_fwhm_export",
        reference=moment,
    )
    title = 'ax.set_title(f"Conditional frequency; displayed={int(counts.sum())}; total={len(df)}; excluded={len(df)-int(counts.sum())}; denominator=displayed" if normalize else "Spectral Sub-class Distribution")'
    add(
        "CM4_year_disguised_denominator",
        "mutant_red",
        edit(
            dynamic,
            "plot_subclass_distribution",
            title,
            'ax.set_title("Top 3 sub-classes, LAMOST DR5 (2015)")',
        ),
        "truncated_percentages",
        reference=dynamic,
    )
    add(
        "CM4b_frozen_population_text",
        "mutant_red",
        edit(
            dynamic,
            "plot_subclass_distribution",
            title,
            'ax.set_title("Conditional frequency: 12 of 15 examples; 3 excluded")',
        ),
        "truncated_percentages",
        reference=dynamic,
        scope="FIXTURE_ONLY counterexample from historical A04",
    )
    add(
        "CM5_reversed_class_labels",
        "mutant_green",
        edit(
            source,
            "plot_subclass_distribution",
            "ax.bar(counts.index.astype(str), values.values)",
            "ax.bar(counts.index.astype(str)[::-1], values.values)",
        ),
        "subclass_raw_counts",
    )
    add(
        "CM6_reversed_shap_names",
        "mutant_green",
        edit(
            source,
            "_run_shap_analysis",
            '{"feature": names_for_plot, "mean_abs_shap": mean_abs}',
            '{"feature": list(names_for_plot)[::-1], "mean_abs_shap": mean_abs}',
        ),
        "binary_shap_mean",
    )
    assert len({item["id"] for item in out}) == len(out)
    return out
