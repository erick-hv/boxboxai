"""
pytest suite for boxboxai_bot core trigger and driver-resolution functions.
Run with: python3 -m pytest test_boxboxai_core.py -v
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boxboxai_bot as bot


# ── resolve_driver_code ───────────────────────────────────────────────────────

class TestResolveDriverCode:

    # Basic resolution
    def test_hamilton_resolves_to_ham(self):
        assert bot.resolve_driver_code("hamilton") == "HAM"

    def test_checo_resolves_to_per(self):
        assert bot.resolve_driver_code("checo") == "PER"

    def test_multi_word_max_verstappen(self):
        # "max" was removed from the map (common word) but "verstappen" is present
        assert bot.resolve_driver_code("max verstappen") == "VER"

    def test_uppercase_code_resolves(self):
        # "VER".lower() == "ver" which is in the map
        assert bot.resolve_driver_code("VER") == "VER"

    # False-positive protection
    def test_championship_does_not_resolve_to_ham(self):
        # "ham" is in the map but \bham\b won't match inside "championship"
        assert bot.resolve_driver_code("championship standings") is None

    def test_stopper_does_not_resolve_to_per(self):
        # "per" was removed from the map entirely (common English word)
        assert bot.resolve_driver_code("is barcelona a 1 or 2 stopper") is None

    def test_had_in_sentence_does_not_resolve(self):
        # "had" was removed from the map entirely (common English word)
        assert bot.resolve_driver_code("i had a great day at the circuit") is None

    def test_gas_in_sentence_does_not_resolve(self):
        # "gas" was removed from the map entirely (common English word)
        assert bot.resolve_driver_code("fill up the gas tank") is None

    def test_no_driver_returns_none(self):
        assert bot.resolve_driver_code("who won the race today") is None


# ── _needs_fia_docs ───────────────────────────────────────────────────────────

class TestNeedsFiaDocs:

    def test_penalties_plural_triggers(self):
        assert bot._needs_fia_docs("how many penalties were given out") is True

    def test_penalized_triggers(self):
        assert bot._needs_fia_docs("was leclerc penalized for that move") is True

    def test_why_did_retire_triggers(self):
        assert bot._needs_fia_docs("why did leclerc retire from the race") is True

    def test_stewards_decision_triggers(self):
        assert bot._needs_fia_docs("stewards decision on the incident") is True

    def test_weather_query_does_not_trigger(self):
        assert bot._needs_fia_docs("what's the weather like in monaco") is False

    def test_spanish_penalizacion_triggers(self):
        assert bot._needs_fia_docs("recibió una penalización de 5 segundos") is True


# ── _is_live_session_question ─────────────────────────────────────────────────

class TestIsLiveSessionQuestion:

    def test_top_10_fp2_triggers(self):
        assert bot._is_live_session_question("top 10 fp2") is True

    def test_who_got_pole_triggers(self):
        assert bot._is_live_session_question("who got pole") is True

    def test_qualifying_results_triggers(self):
        assert bot._is_live_session_question("qualifying results for spain") is True

    def test_who_won_today_does_not_trigger(self):
        # "who won" is deliberately not a session keyword
        assert bot._is_live_session_question("who won today") is False


# ── _is_weather_query ─────────────────────────────────────────────────────────

class TestIsWeatherQuery:

    def test_weather_keyword_triggers(self):
        assert bot._is_weather_query("what's the weather like in monaco") is True

    def test_spanish_que_tiempo_hace_triggers(self):
        assert bot._is_weather_query("qué tiempo hace en barcelona") is True

    def test_tiempo_de_vuelta_does_not_trigger(self):
        # "tiempo de vuelta" means lap time, not weather — no keyword matches
        assert bot._is_weather_query("tiempo de vuelta de norris") is False

    def test_hace_tiempo_does_not_trigger(self):
        # Reversed phrase — "hace tiempo" (a while ago) is not in the keyword list
        assert bot._is_weather_query("hace tiempo que no gana alonso") is False


# ── _resolve_multiple_driver_codes ────────────────────────────────────────────

class TestResolveMultipleDriverCodes:

    def test_compare_hamilton_and_russell(self):
        codes = bot._resolve_multiple_driver_codes("compare hamilton and russell")
        assert codes == ["HAM", "RUS"]

    def test_verstappen_vs_leclerc(self):
        codes = bot._resolve_multiple_driver_codes("verstappen vs leclerc")
        assert codes == ["VER", "LEC"]

    def test_stopper_query_returns_empty(self):
        # No driver names present despite "stopper" containing "per" as substring
        codes = bot._resolve_multiple_driver_codes("is barcelona a 1 or 2 stopper")
        assert codes == []


# ── get_predictor_context stale CSV detection ─────────────────────────────────

class TestGetPredictorContextStaleness:

    def _write_csv(self, tmp_path, round_num: int, race_name: str = "Test GP") -> None:
        csv = tmp_path / "f1_2026_predicciones.csv"
        csv.write_text(
            f"round_num,race_name,code,FullName,TeamName,win_mc_pct,podium_mc_pct,avg_mc_pos\n"
            f"{round_num},{race_name},NOR,Lando Norris,McLaren,35.2,68.4,2.1\n"
        )
        return csv

    def test_returns_data_when_round_matches(self, tmp_path, monkeypatch):
        csv_path = self._write_csv(tmp_path, round_num=8)
        monkeypatch.setattr(bot, "PREDICTOR_CSV", csv_path)
        bot._PREDICTOR_CACHE.clear()
        block, rows = bot.get_predictor_context(expected_round=8)
        assert rows, "Should return rows when CSV round matches expected"
        assert block != ""

    def test_returns_empty_when_round_mismatches(self, tmp_path, monkeypatch):
        # CSV was written for R7 (Spain) but we're asking for R8 (Austria)
        csv_path = self._write_csv(tmp_path, round_num=7, race_name="Spanish GP")
        monkeypatch.setattr(bot, "PREDICTOR_CSV", csv_path)
        bot._PREDICTOR_CACHE.clear()
        block, rows = bot.get_predictor_context(expected_round=8)
        assert rows == [], "Stale CSV (R7) must not be returned for R8"
        assert block == ""

    def test_returns_data_when_no_expected_round(self, tmp_path, monkeypatch):
        # Callers that don't pass expected_round (e.g. cache-clear call in auto-predictor)
        # should always get the data regardless of round stamp
        csv_path = self._write_csv(tmp_path, round_num=7)
        monkeypatch.setattr(bot, "PREDICTOR_CSV", csv_path)
        bot._PREDICTOR_CACHE.clear()
        block, rows = bot.get_predictor_context(expected_round=None)
        assert rows, "Should return rows when no expected_round is given"

    def test_returns_data_for_old_csv_without_round_num(self, tmp_path, monkeypatch):
        # Pre-fix CSVs have no round_num column — should degrade gracefully, not error
        csv = tmp_path / "f1_2026_predicciones.csv"
        csv.write_text(
            "code,FullName,TeamName,win_mc_pct,podium_mc_pct,avg_mc_pos\n"
            "NOR,Lando Norris,McLaren,35.2,68.4,2.1\n"
        )
        monkeypatch.setattr(bot, "PREDICTOR_CSV", csv)
        bot._PREDICTOR_CACHE.clear()
        block, rows = bot.get_predictor_context(expected_round=8)
        assert rows, "Old CSV without round_num should still be served (graceful degradation)"

    def test_returns_empty_when_csv_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bot, "PREDICTOR_CSV", tmp_path / "nonexistent.csv")
        bot._PREDICTOR_CACHE.clear()
        block, rows = bot.get_predictor_context(expected_round=8)
        assert rows == []
        assert block == ""
