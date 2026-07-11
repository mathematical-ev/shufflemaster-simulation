# SPDX-License-Identifier: GPL-3.0-or-later

from scripts.run_one2six_source_diagnostics import run_diagnostics

from shufflemaster_sim.card_sources import One2SixConfig


def test_one2six_diagnostic_core_returns_summary() -> None:
    result = run_diagnostics(
        draws=100,
        seed=42,
        recycle_every=20,
        config=One2SixConfig(),
    )

    assert result["draws"] == 100
    assert result["unique_physical_ids_seen"] > 0
    assert result["ejection_count"] > 0
    assert result["invariant_check"] == "passed"
