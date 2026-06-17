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


# ── _is_pre_qualifying_csv + format_predictor_for_claude sentinel ─────────────

class TestPreQualifyingSentinel:

    def _make_rows(self, quali_pos_next_values: list) -> list:
        return [
            {
                "code": f"D{i}", "FullName": f"Driver {i}", "TeamName": "Team",
                "win_mc_pct": "10.0", "podium_mc_pct": "30.0",
                "avg_mc_pos": "5.0", "mechanical_risk": "0.05",
                "champ_pts": str(100 - i * 5),
                "quali_pos_next": str(v),
                "recent_form": "8.0", "circuit_score": "0.9",
            }
            for i, v in enumerate(quali_pos_next_values, 1)
        ]

    def test_all_sentinel_values_detected_as_pre_qualifying(self):
        rows = self._make_rows([22.0] * 10)
        assert bot._is_pre_qualifying_csv(rows) is True

    def test_differentiated_positions_not_pre_qualifying(self):
        rows = self._make_rows(list(range(1, 11)))
        assert bot._is_pre_qualifying_csv(rows) is False

    def test_empty_rows_returns_false(self):
        # Empty rows = no CSV at all; get_predictor_context returns ("", []) before
        # this helper is called, so the correct return is False (not pre-qualifying)
        assert bot._is_pre_qualifying_csv([]) is False

    def test_pre_qualifying_block_contains_warning(self):
        rows = self._make_rows([22.0] * 10)
        block = bot.format_predictor_for_claude(rows)
        assert "PRE-QUALIFYING PREVIEW" in block, \
            "Pre-qualifying block must contain warning header"

    def test_pre_qualifying_block_omits_sentinel_quali_field(self):
        rows = self._make_rows([22.0] * 10)
        block = bot.format_predictor_for_claude(rows)
        assert "Quali=" not in block, \
            "Sentinel quali_pos_next must not appear in KEY FEATURES"

    def test_post_qualifying_block_includes_quali_field(self):
        rows = self._make_rows(list(range(1, 11)))
        block = bot.format_predictor_for_claude(rows)
        assert "Quali=" in block, \
            "Real quali positions must appear in KEY FEATURES"

    def test_champ_pts_in_formatted_block(self):
        rows = self._make_rows(list(range(1, 11)))
        block = bot.format_predictor_for_claude(rows)
        assert "Pts" in block, "Championship points column must appear in table header"

    def test_champ_pts_in_pre_qualifying_block(self):
        rows = self._make_rows([22.0] * 10)
        block = bot.format_predictor_for_claude(rows)
        assert "Pts" in block, "Pts column must appear even in pre-qualifying block"


# ── tyre strategy surfaced in system prompt ───────────────────────────────────

class TestTyreStrategyInSystemPrompt:
    """
    Regression tests for the three key-name mismatches that previously caused
    tyre strategy data to be collected correctly but never reach Claude.
    """

    def _make_mem(self, tyre_strategies: str) -> dict:
        return {
            "episodic": [{
                "round": 9,
                "race_name": "Spanish Grand Prix",
                "track": "barcelona",
                "winner": "HAM",
                "p2": "RUS",
                "p3": "NOR",
                "fastest_lap": "VER",
                "fastest_lap_time": "1:14.234",
                "pitstops": {"tyre_strategies": tyre_strategies},
            }],
            "semantic": {},
        }

    def test_tyre_strategies_appear_in_compact_race_lines(self):
        strats = "HAM:M(15)→H(35) | RUS:M(13)→H(37)"
        mem = self._make_mem(strats)
        prompt = bot.build_system_prompt(mem)
        assert "HAM:M(15)→H(35)" in prompt, \
            "Tyre strategies must appear in RACE RESULTS compact lines"

    def test_empty_tyre_strategies_does_not_crash(self):
        mem = self._make_mem("")
        prompt = bot.build_system_prompt(mem)
        assert "Spanish Grand Prix" in prompt

    def test_missing_pitstops_key_does_not_crash(self):
        mem = {
            "episodic": [{
                "round": 9, "race_name": "Spanish Grand Prix",
                "track": "barcelona", "winner": "HAM",
                "p2": "RUS", "p3": "NOR",
            }],
            "semantic": {},
        }
        prompt = bot.build_system_prompt(mem)
        assert "Spanish Grand Prix" in prompt


# ── circuit map PDF parser ────────────────────────────────────────────────────

# Exact pypdf page-2 text from the 2026 Barcelona-Catalunya Circuit Map PDF
# (FIA Document 8, published 11 June 2026).  pypdf reads the CIRCUIT DATA
# table column-by-column, so zone names appear before the label rows and the
# overtake distances appear at the end.
BARCELONA_CIRCUIT_MAP_TEXT = """
1
M0.8
2
3
4
5
6
7
8
9
10
11
13
14
M1
M2
M3
M5
M3.5
M4
M4.2
M4.4
M6
M7
M9
M9.2 M9.5
M10
M10.3
M11.2
M13
M16
M0.1
M3.2
M8
M14M12
M16.6
12
1
2
3
4
9
5
8
10
6
11
7
14
12
13
16
15
S2
T
S1
O
T
 D
OT A
SM      A1
SM A2
SM      A2
SM
 A
3
SM      A1
SM
 A
3
SM
 A
4
SM
 A
4
VERSION 1 - ISSUED 21.05.26
© 2026 Formula One World Championship Limited
FORMULA 1 MSC CRUISES GRAN PREMIO DE BARCELONA-CATALUNYA 2026 - Barcelona
CIRCUIT DATA
SECTOR 3
SECTOR 2
SECTOR 1
-  1.273km
-  1.765km
-  1.619km
 SPEED TRAP  [T]
 INTERMEDIATE 2  [S2]
-  220m before T1
-  90m before T10
-  50m before T4 INTERMEDIATE 1  [S1]
CIRCUIT CENTRELINE LENGTH -  4.657km
Circuit Map
OVERTAKE STRAIGHT MODE
-   ZONE A1   -
-   ZONE A2   -
-   ZONE A3   -
-   ZONE A4   -
-   ZONE A5   -
ACTIVATION
DETECTION 45m after T14
40m before T3 exit
90m after T5
40m after T9
n/a
85m after T14
T3 exit
90m after T5
90m after T9
n/a
-  Entry T14
-  Apex T13
15
LEGEND
START LINE
CONTROL LINE
CORNER NUMBER
SM A [ZONE No.]
STRAIGHT MODE
NORMAL GRIP ACTIVATION
SM A [ZONE No.]
 STRAIGHT MODE
LOW GRIP ACTIVATION
OT A
OT D
OVERTAKE ACTIVATION
OVERTAKE DETECTION
FIA LIGHT PANEL22
M2 MARSHAL POST
"""


class TestCircuitMapParser:
    """
    Regression tests for _parse_circuit_map_pdf_text using the real
    pypdf-extracted text from the 2026 Barcelona FIA Circuit Map PDF.
    """

    def test_overtake_detection_parsed(self):
        result = bot._parse_circuit_map_pdf_text(BARCELONA_CIRCUIT_MAP_TEXT)
        assert result["overtake"]["detection"] == "Apex T13"

    def test_overtake_activation_parsed(self):
        result = bot._parse_circuit_map_pdf_text(BARCELONA_CIRCUIT_MAP_TEXT)
        assert result["overtake"]["activation"] == "Entry T14"

    def test_four_active_zones_returned(self):
        # Zone A5 is n/a for both grip levels and must be omitted
        result = bot._parse_circuit_map_pdf_text(BARCELONA_CIRCUIT_MAP_TEXT)
        assert len(result["straight_mode_zones"]) == 4

    def test_zone_names_in_order(self):
        result = bot._parse_circuit_map_pdf_text(BARCELONA_CIRCUIT_MAP_TEXT)
        names = [z["zone"] for z in result["straight_mode_zones"]]
        assert names == ["A1", "A2", "A3", "A4"]

    def test_zone_a1_normal_grip(self):
        result = bot._parse_circuit_map_pdf_text(BARCELONA_CIRCUIT_MAP_TEXT)
        z = next(z for z in result["straight_mode_zones"] if z["zone"] == "A1")
        assert z["activation_normal"] == "45m after T14"

    def test_zone_a1_low_grip(self):
        result = bot._parse_circuit_map_pdf_text(BARCELONA_CIRCUIT_MAP_TEXT)
        z = next(z for z in result["straight_mode_zones"] if z["zone"] == "A1")
        assert z["activation_low_grip"] == "85m after T14"

    def test_zone_a2_values(self):
        result = bot._parse_circuit_map_pdf_text(BARCELONA_CIRCUIT_MAP_TEXT)
        z = next(z for z in result["straight_mode_zones"] if z["zone"] == "A2")
        assert z["activation_normal"] == "40m before T3 exit"
        assert z["activation_low_grip"] == "T3 exit"

    def test_zone_a5_omitted(self):
        result = bot._parse_circuit_map_pdf_text(BARCELONA_CIRCUIT_MAP_TEXT)
        names = [z["zone"] for z in result["straight_mode_zones"]]
        assert "A5" not in names

    def test_missing_section_returns_empty_dict(self):
        assert bot._parse_circuit_map_pdf_text("No circuit data here.") == {}

    def test_empty_string_returns_empty_dict(self):
        assert bot._parse_circuit_map_pdf_text("") == {}
