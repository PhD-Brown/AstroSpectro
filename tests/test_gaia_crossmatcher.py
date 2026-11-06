"""Issue #13 regressions: real pandas/astropy tables, mocked services, no network.

Run with ``python -m pytest -q tests/test_gaia_crossmatcher.py`` or, without
pytest, ``python -m unittest discover -s tests -p test_gaia_crossmatcher.py -v``.
"""

import importlib.util
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from astropy.table import Table


# Gaia's singleton retrieves status messages at import time. Suppress that GET
# before loading the real module; block sockets as a second line of defence.
with (
    patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")),
    patch(
        "astroquery.utils.tap.conn.tapconn.TapConn.execute_tapget",
        return_value=SimpleNamespace(status=204),
    ),
):
    from astroquery.utils.tap.model.job import Job
    from astroquery.xmatch import XMatch

    spec = importlib.util.spec_from_file_location(
        "gaia_crossmatcher_under_test",
        Path(__file__).resolve().parents[1] / "src/tools/gaia_crossmatcher.py",
    )
    gaia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gaia)


def completed_job(frame):
    job = Mock(spec=Job)
    job.jobid = "mock-job"
    job.get_phase.return_value = "COMPLETED"
    job.get_results.return_value = Table.from_pandas(frame)
    return job


class GaiaCrossmatchTests(unittest.TestCase):
    def setUp(self):
        for target in (
            "socket.socket.connect",
            "socket.create_connection",
            "requests.sessions.Session.request",
        ):
            blocker = patch(target, side_effect=AssertionError("Network forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)
        delay = patch.object(
            gaia.time, "sleep", side_effect=AssertionError("Unexpected retry")
        )
        delay.start()
        self.addCleanup(delay.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "enriched.csv"

    def catalog(self, n=1):
        return pd.DataFrame({"ra": np.full(n, 10.0), "dec": np.full(n, -5.0)})

    def enrich(self, mode="auto", n=51):
        return gaia.enrich_catalog_with_gaia(
            self.catalog(n),
            str(self.output),
            mode=mode,
            verbose=False,
        )

    def test_esa_complete_batches_allow_missing_parameters(self):
        for risky in (False, True):
            with self.subTest(include_risky=risky):
                frames = [
                    pd.DataFrame({"source_id": ids, "mh_gspphot": np.nan})
                    for ids in ([101, 102], [103, 104], [105])
                ]
                with (
                    patch.object(gaia, "_GSPPHOT_BATCH_SIZE", 2),
                    patch.object(
                        gaia.Gaia,
                        "launch_job_async",
                        autospec=True,
                        side_effect=[completed_job(f) for f in frames],
                    ) as launch,
                    self.assertLogs(gaia.logger, logging.INFO) as logs,
                ):
                    result = gaia._gaia_supplement_by_ids(
                        np.arange(101, 106),
                        risky,
                        debug=True,
                    )
                pd.testing.assert_frame_equal(
                    result, pd.concat(frames, ignore_index=True)
                )
                self.assertTrue(result.mh_gspphot.isna().all())
                self.assertEqual(launch.call_count, 3)
                for call, ids in zip(
                    launch.call_args_list, ("101, 102", "103, 104", "105")
                ):
                    query = call.args[0]
                    self.assertIn(f"WHERE gs.source_id IN ({ids})", query)
                    self.assertIn("LEFT JOIN gaiadr3.astrophysical_parameters", query)
                    self.assertNotIn("TAP_UPLOAD", query)
                    self.assertEqual("ap.radius_gspphot" in query, risky)
                    self.assertEqual(call.kwargs, {"dump_to_file": False})
                self.assertIn("Traitement complet", "\n".join(logs.output))
                self.assertNotIn("partiel", "\n".join(logs.output))

    def test_esa_failure_before_first_batch(self):
        with (
            patch.object(gaia, "_GSPPHOT_BATCH_SIZE", 2),
            patch.object(
                gaia.Gaia,
                "launch_job_async",
                autospec=True,
                side_effect=RuntimeError("ESA unavailable"),
            ) as launch,
            self.assertLogs(gaia.logger, logging.INFO) as logs,
        ):
            result = gaia._gaia_supplement_by_ids(np.arange(5), False)
        self.assertEqual(launch.call_count, 1)
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["source_id"])
        messages = "\n".join(logs.output)
        self.assertIn("abandon des lots restants", messages)
        self.assertIn("Aucun supplément", messages)
        self.assertNotIn("fallback", messages)

    def test_esa_partial_retains_completed_batch(self):
        frame = pd.DataFrame({"source_id": [101, 102], "mh_gspphot": [-0.2, np.nan]})
        for failure in (RuntimeError("ESA unavailable"), "poll", "results"):
            with self.subTest(failure=failure):
                failed_job = completed_job(frame)
                if failure == "poll":
                    failed_job.get_phase.return_value = "ERROR"
                elif failure == "results":
                    failed_job.get_results.side_effect = RuntimeError(
                        "Cannot read results"
                    )
                second = failure if isinstance(failure, Exception) else failed_job
                with (
                    patch.object(gaia, "_GSPPHOT_BATCH_SIZE", 2),
                    patch.object(
                        gaia.Gaia,
                        "launch_job_async",
                        autospec=True,
                        side_effect=[completed_job(frame), second],
                    ) as launch,
                    self.assertLogs(gaia.logger, logging.INFO) as logs,
                ):
                    result = gaia._gaia_supplement_by_ids(np.arange(101, 107), False)
                pd.testing.assert_frame_equal(result, frame)
                self.assertEqual(launch.call_count, 2)
                self.assertIn("Supplément partiel", "\n".join(logs.output))
                self.assertNotIn("Traitement complet", "\n".join(logs.output))

    def test_esa_success_with_zero_rows_has_empty_source_id_schema(self):
        frame = pd.DataFrame(
            {
                "source_id": pd.Series(dtype="int64"),
                "mh_gspphot": pd.Series(dtype="float64"),
            }
        )
        with (
            patch.object(
                gaia.Gaia,
                "launch_job_async",
                autospec=True,
                return_value=completed_job(frame),
            ),
            self.assertLogs(gaia.logger, logging.INFO) as logs,
        ):
            result = gaia._gaia_supplement_by_ids(np.array([101]), False)
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["source_id"])
        self.assertIn("Aucun supplément", "\n".join(logs.output))

    def test_esa_no_ids_does_not_query(self):
        with patch.object(gaia.Gaia, "launch_job_async", autospec=True) as launch:
            result = gaia._gaia_supplement_by_ids(np.array([np.nan]), False)
        launch.assert_not_called()
        self.assertEqual(list(result.columns), ["source_id"])
        self.assertTrue(result.empty)

    def test_auto_total_failure_raises_explicit_error_without_saving(self):
        with (
            patch.object(
                gaia, "_xmatch_crossmatch", side_effect=RuntimeError("CDS down")
            ),
            patch.object(
                gaia, "_bulk_crossmatch", side_effect=RuntimeError("ESA down")
            ),
            patch.object(gaia, "_cone_crossmatch") as cone,
        ):
            with self.assertRaisesRegex(RuntimeError, "Échec du cross-match Gaia"):
                self.enrich()
        cone.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_valid_xmatch_without_matches_is_not_a_service_failure(self):
        for mode in ("xmatch", "auto"):
            with (
                self.subTest(mode=mode),
                patch.object(
                    XMatch,
                    "query",
                    autospec=True,
                    return_value=Table(names=["objid"], dtype=["int64"]),
                ) as query,
                patch.object(gaia, "_gaia_supplement_by_ids") as supplement,
                patch.object(
                    gaia, "_bulk_crossmatch", side_effect=RuntimeError("ESA down")
                ) as bulk,
                patch.object(gaia, "_cone_crossmatch") as cone,
            ):
                if self.output.exists():
                    self.output.unlink()
                result, stats = self.enrich(mode=mode)
                self.assertEqual(len(result), 51)
                self.assertEqual(stats["matched"], 0)
                self.assertEqual(stats["match_rate_pct"], 0.0)
                self.assertEqual(stats["mode_used"], "xmatch")
                self.assertTrue(self.output.exists())
                self.assertEqual(len(pd.read_csv(self.output)), 51)
                query.assert_called_once()
                supplement.assert_not_called()
                bulk.assert_not_called()
                cone.assert_not_called()

    def test_auto_total_cone_failure_is_not_a_zero_match_catalog(self):
        with (
            patch.object(
                gaia, "_xmatch_crossmatch", side_effect=RuntimeError("CDS down")
            ),
            patch.object(
                gaia, "_bulk_crossmatch", side_effect=RuntimeError("ESA down")
            ),
            patch.object(
                gaia.Gaia,
                "cone_search_async",
                autospec=True,
                side_effect=RuntimeError("Cone unavailable"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Échec du cross-match Gaia"):
                self.enrich(n=2)
        self.assertFalse(self.output.exists())

    def test_auto_total_bulk_failure_is_not_a_zero_match_catalog(self):
        with (
            patch.object(
                gaia, "_xmatch_crossmatch", side_effect=RuntimeError("CDS down")
            ),
            patch.object(gaia, "_INITIAL_BATCH_SIZE", 50),
            patch.object(
                gaia.Gaia,
                "launch_job_async",
                autospec=True,
                side_effect=RuntimeError("ESA unavailable"),
            ),
            patch.object(gaia, "_cone_crossmatch") as cone,
        ):
            with self.assertRaisesRegex(RuntimeError, "Échec du cross-match Gaia"):
                self.enrich()
        cone.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_valid_cone_without_matches_is_not_a_service_failure(self):
        empty = pd.DataFrame({"source_id": pd.Series(dtype="int64")})
        with (
            patch.object(
                gaia, "_xmatch_crossmatch", side_effect=RuntimeError("CDS down")
            ),
            patch.object(
                gaia, "_bulk_crossmatch", side_effect=RuntimeError("ESA down")
            ),
            patch.object(
                gaia.Gaia,
                "cone_search_async",
                autospec=True,
                return_value=completed_job(empty),
            ) as cone,
        ):
            result, stats = self.enrich(n=2)
        self.assertEqual(cone.call_count, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(stats["matched"], 0)
        self.assertEqual(stats["mode_used"], "cone")
        self.assertTrue(self.output.exists())

    def test_valid_bulk_without_matches_stops_auto_fallback(self):
        empty = pd.DataFrame({"objid": pd.Series(dtype="int64")})
        with (
            patch.object(
                gaia, "_xmatch_crossmatch", side_effect=RuntimeError("CDS down")
            ),
            patch.object(
                gaia.Gaia,
                "launch_job_async",
                autospec=True,
                return_value=completed_job(empty),
            ) as launch,
            patch.object(gaia, "_cone_crossmatch") as cone,
        ):
            result, stats = self.enrich()
        self.assertEqual(len(result), 51)
        self.assertEqual(stats["matched"], 0)
        self.assertEqual(stats["mode_used"], "bulk")
        launch.assert_called_once()
        cone.assert_not_called()

    def test_partial_esa_supplement_merges_without_losing_xmatch_rows(self):
        raw = Table(
            {
                "objid": [0, 1, 2],
                "Source": [101, 102, 103],
                "angDist": [0.1, 0.2, 0.1],
                "Teff": [5100.0, 5200.0, 5300.0],
            }
        )
        supplement = pd.DataFrame({"source_id": [101], "mh_gspphot": [-0.4]})
        with (
            patch.object(XMatch, "query", autospec=True, return_value=raw),
            patch.object(gaia, "_GSPPHOT_BATCH_SIZE", 1),
            patch.object(
                gaia.Gaia,
                "launch_job_async",
                autospec=True,
                side_effect=[
                    completed_job(supplement),
                    RuntimeError("ESA unavailable"),
                ],
            ) as launch,
            self.assertLogs(gaia.logger, logging.INFO) as logs,
        ):
            result, stats = self.enrich(mode="xmatch", n=3)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(stats["matched"], 3)
        self.assertEqual(result.source_id.tolist(), [101, 102, 103])
        self.assertEqual(result.teff_gspphot.tolist(), [5100.0, 5200.0, 5300.0])
        self.assertEqual(result.loc[0, "mh_gspphot"], -0.4)
        self.assertTrue(result.loc[1:, "mh_gspphot"].isna().all())
        self.assertIn("Supplément partiel", "\n".join(logs.output))

    def test_alias_collision_keeps_first_received_column_and_warns(self):
        for first in (-0.4, np.nan):
            with self.subTest(first=first):
                raw = Table(
                    {
                        "objid": [0],
                        "Source": [101],
                        "angDist": [0.1],
                        "MH": [first],
                        "[M/H]": [-0.8],
                    }
                )
                with (
                    patch.object(XMatch, "query", autospec=True, return_value=raw),
                    patch.object(
                        gaia,
                        "_gaia_supplement_by_ids",
                        return_value=pd.DataFrame(columns=["source_id"]),
                    ),
                    self.assertLogs(gaia.logger, logging.WARNING) as logs,
                ):
                    result = gaia._xmatch_crossmatch(
                        gaia._validate_catalog(self.catalog()),
                        0.5,
                        False,
                    )
                value = result.loc[0, "mh_gspphot"]
                self.assertTrue(np.isscalar(value))
                if pd.isna(first):
                    self.assertTrue(pd.isna(value))
                else:
                    self.assertEqual(value, first)
                self.assertIn(
                    "première colonne reçue conservée", "\n".join(logs.output)
                )

    def test_metallicity_aliases_map_to_a_single_scalar_column(self):
        aliases = ("[M/H]", "_M_H_", "__M_H_", "MH")
        for columns in [(alias,) for alias in aliases] + [aliases]:
            with self.subTest(columns=columns):
                raw = {"objid": [0], "Source": [101], "angDist": [0.1]}
                raw.update({alias: [-0.25] for alias in columns})
                with (
                    patch.object(
                        XMatch, "query", autospec=True, return_value=Table(raw)
                    ),
                    patch.object(
                        gaia,
                        "_gaia_supplement_by_ids",
                        return_value=pd.DataFrame(columns=["source_id"]),
                    ),
                ):
                    result = gaia._xmatch_crossmatch(
                        gaia._validate_catalog(self.catalog()),
                        0.5,
                        False,
                    )
                self.assertTrue(result.columns.is_unique)
                self.assertEqual(result.columns.tolist().count("mh_gspphot"), 1)
                self.assertEqual(result.loc[0, "mh_gspphot"], -0.25)


if __name__ == "__main__":
    unittest.main()
