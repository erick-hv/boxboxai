"""
pytest suite for boxboxai_bot core trigger and driver-resolution functions.
Run with: python3 -m pytest test_boxboxai_core.py -v
"""
import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boxboxai_bot as bot
import context_builder
import data_layer


# ── resolve_driver_code ───────────────────────────────────────────────────────

class TestResolveDriverCode:

    # Basic resolution
    def test_hamilton_resolves_to_ham(self):
        assert context_builder.resolve_driver_code("hamilton") == "HAM"

    def test_checo_resolves_to_per(self):
        assert context_builder.resolve_driver_code("checo") == "PER"

    def test_multi_word_max_verstappen(self):
        # "max" was removed from the map (common word) but "verstappen" is present
        assert context_builder.resolve_driver_code("max verstappen") == "VER"

    def test_uppercase_code_resolves(self):
        # "VER".lower() == "ver" which is in the map
        assert context_builder.resolve_driver_code("VER") == "VER"

    # False-positive protection
    def test_championship_does_not_resolve_to_ham(self):
        # "ham" is in the map but \bham\b won't match inside "championship"
        assert context_builder.resolve_driver_code("championship standings") is None

    def test_stopper_does_not_resolve_to_per(self):
        # "per" was removed from the map entirely (common English word)
        assert context_builder.resolve_driver_code("is barcelona a 1 or 2 stopper") is None

    def test_had_in_sentence_does_not_resolve(self):
        # "had" was removed from the map entirely (common English word)
        assert context_builder.resolve_driver_code("i had a great day at the circuit") is None

    def test_gas_in_sentence_does_not_resolve(self):
        # "gas" was removed from the map entirely (common English word)
        assert context_builder.resolve_driver_code("fill up the gas tank") is None

    def test_no_driver_returns_none(self):
        assert context_builder.resolve_driver_code("who won the race today") is None


# ── _needs_fia_docs ───────────────────────────────────────────────────────────

class TestNeedsFiaDocs:

    def test_penalties_plural_triggers(self):
        assert context_builder._needs_fia_docs("how many penalties were given out") is True

    def test_penalized_triggers(self):
        assert context_builder._needs_fia_docs("was leclerc penalized for that move") is True

    def test_why_did_retire_triggers(self):
        assert context_builder._needs_fia_docs("why did leclerc retire from the race") is True

    def test_stewards_decision_triggers(self):
        assert context_builder._needs_fia_docs("stewards decision on the incident") is True

    def test_weather_query_does_not_trigger(self):
        assert context_builder._needs_fia_docs("what's the weather like in monaco") is False

    def test_spanish_penalizacion_triggers(self):
        assert context_builder._needs_fia_docs("recibió una penalización de 5 segundos") is True


# ── _is_live_session_question ─────────────────────────────────────────────────

class TestIsLiveSessionQuestion:

    def test_top_10_fp2_triggers(self):
        assert context_builder._is_live_session_question("top 10 fp2") is True

    def test_who_got_pole_triggers(self):
        assert context_builder._is_live_session_question("who got pole") is True

    def test_qualifying_results_triggers(self):
        assert context_builder._is_live_session_question("qualifying results for spain") is True

    def test_who_won_today_does_not_trigger(self):
        # "who won" is deliberately not a session keyword
        assert context_builder._is_live_session_question("who won today") is False


# ── _is_weather_query ─────────────────────────────────────────────────────────

class TestIsWeatherQuery:

    def test_weather_keyword_triggers(self):
        assert context_builder._is_weather_query("what's the weather like in monaco") is True

    def test_spanish_que_tiempo_hace_triggers(self):
        assert context_builder._is_weather_query("qué tiempo hace en barcelona") is True

    def test_tiempo_de_vuelta_does_not_trigger(self):
        # "tiempo de vuelta" means lap time, not weather — no keyword matches
        assert context_builder._is_weather_query("tiempo de vuelta de norris") is False

    def test_hace_tiempo_does_not_trigger(self):
        # Reversed phrase — "hace tiempo" (a while ago) is not in the keyword list
        assert context_builder._is_weather_query("hace tiempo que no gana alonso") is False


# ── _detect_driver_stat_query ─────────────────────────────────────────────────

class TestDetectDriverStatQuery:

    def test_how_many_wins_hamilton_returns_HAM(self):
        assert context_builder._detect_driver_stat_query("how many wins does hamilton have") == "HAM"

    def test_career_stats_verstappen_returns_VER(self):
        assert context_builder._detect_driver_stat_query("what are verstappen career stats") == "VER"

    def test_no_stat_keyword_returns_none(self):
        assert context_builder._detect_driver_stat_query("who is winning the championship") is None

    def test_no_driver_name_returns_none(self):
        # stat keyword present but no recognized driver name
        assert context_builder._detect_driver_stat_query("how many wins did that guy get") is None

    def test_spanish_cuantas_victorias_norris_returns_NOR(self):
        assert context_builder._detect_driver_stat_query("cuántas victorias tiene lando") == "NOR"


# ── _resolve_multiple_driver_codes ────────────────────────────────────────────

class TestResolveMultipleDriverCodes:

    def test_compare_hamilton_and_russell(self):
        codes = context_builder._resolve_multiple_driver_codes("compare hamilton and russell")
        assert codes == ["HAM", "RUS"]

    def test_verstappen_vs_leclerc(self):
        codes = context_builder._resolve_multiple_driver_codes("verstappen vs leclerc")
        assert codes == ["VER", "LEC"]

    def test_stopper_query_returns_empty(self):
        # No driver names present despite "stopper" containing "per" as substring
        codes = context_builder._resolve_multiple_driver_codes("is barcelona a 1 or 2 stopper")
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
        monkeypatch.setattr(data_layer, "PREDICTOR_CSV", csv_path)
        bot._PREDICTOR_CACHE.clear()
        block, rows = bot.get_predictor_context(expected_round=8)
        assert rows, "Should return rows when CSV round matches expected"
        assert block != ""

    def test_returns_empty_when_round_mismatches(self, tmp_path, monkeypatch):
        # CSV was written for R7 (Spain) but we're asking for R8 (Austria)
        csv_path = self._write_csv(tmp_path, round_num=7, race_name="Spanish GP")
        monkeypatch.setattr(data_layer, "PREDICTOR_CSV", csv_path)
        bot._PREDICTOR_CACHE.clear()
        block, rows = bot.get_predictor_context(expected_round=8)
        assert rows == [], "Stale CSV (R7) must not be returned for R8"
        assert block == ""

    def test_returns_data_when_no_expected_round(self, tmp_path, monkeypatch):
        # Callers that don't pass expected_round (e.g. cache-clear call in auto-predictor)
        # should always get the data regardless of round stamp
        csv_path = self._write_csv(tmp_path, round_num=7)
        monkeypatch.setattr(data_layer, "PREDICTOR_CSV", csv_path)
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
        monkeypatch.setattr(data_layer, "PREDICTOR_CSV", csv)
        bot._PREDICTOR_CACHE.clear()
        block, rows = bot.get_predictor_context(expected_round=8)
        assert rows, "Old CSV without round_num should still be served (graceful degradation)"

    def test_returns_empty_when_csv_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_layer, "PREDICTOR_CSV", tmp_path / "nonexistent.csv")
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
        prompt = context_builder.build_system_prompt(mem)
        assert "HAM:M(15)→H(35)" in prompt, \
            "Tyre strategies must appear in RACE RESULTS compact lines"

    def test_empty_tyre_strategies_does_not_crash(self):
        mem = self._make_mem("")
        prompt = context_builder.build_system_prompt(mem)
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
        prompt = context_builder.build_system_prompt(mem)
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


# ── get_circuit_guide with zone data ─────────────────────────────────────────

_BARCELONA_ZONE_DATA = {
    "overtake": {"detection": "Apex T13", "activation": "Entry T14"},
    "straight_mode_zones": [
        {"zone": "A1", "activation_normal": "45m after T14",
                       "activation_low_grip": "85m after T14"},
        {"zone": "A2", "activation_normal": "40m before T3 exit",
                       "activation_low_grip": "T3 exit"},
    ],
}


class TestCircuitGuideWithZoneData:
    """
    Tests for get_circuit_guide() zone-data suffix injection.
    _get_circuit_zone_data is mocked — no Playwright / network calls.
    """

    def test_zone_data_appended_when_available(self):
        with patch.object(context_builder, "_get_circuit_zone_data_fn",
                          new=[lambda _: _BARCELONA_ZONE_DATA]):
            guide, _img = context_builder.get_circuit_guide("what about barcelona strategy")
        assert "PRECISE 2026 ZONE DATA" in guide
        assert "Apex T13" in guide
        assert "Entry T14" in guide
        assert "A1" in guide
        assert "45m after T14" in guide

    def test_qualitative_guide_still_present(self):
        with patch.object(context_builder, "_get_circuit_zone_data_fn",
                          new=[lambda _: _BARCELONA_ZONE_DATA]):
            guide, _img = context_builder.get_circuit_guide("barcelona")
        # Qualitative content from CIRCUIT_GUIDES must still be there
        assert "CIRCUIT GUIDE" in guide
        assert "BARCELONA" in guide

    def test_graceful_degradation_when_no_zone_data(self):
        with patch.object(context_builder, "_get_circuit_zone_data_fn",
                          new=[lambda _: {}]):
            guide, _img = context_builder.get_circuit_guide("barcelona")
        assert "CIRCUIT GUIDE" in guide
        assert "PRECISE 2026 ZONE DATA" not in guide

    def test_no_zone_data_for_unknown_circuit(self):
        # No circuit match → empty string and None image regardless of zone data
        with patch.object(context_builder, "_get_circuit_zone_data_fn",
                          new=[lambda _: {}]):
            guide, img = context_builder.get_circuit_guide("what is F1?")
        assert guide == ""
        assert img is None

    def test_zone_suffix_exception_does_not_propagate(self):
        # If _get_circuit_zone_data raises, guide still returns qualitative text
        def _raise(_): raise RuntimeError("network down")
        with patch.object(context_builder, "_get_circuit_zone_data_fn",
                          new=[_raise]):
            guide, _img = context_builder.get_circuit_guide("barcelona")
        assert "CIRCUIT GUIDE" in guide
        assert "PRECISE 2026 ZONE DATA" not in guide


# ── Spa / Spanish substring collision regression ──────────────────────────────

class TestSpanishNotSpa:
    """
    'spa' is a CIRCUIT_GUIDES key.  'spanish' / 'spain' contain 'spa' as a
    substring.  Word-boundary matching in _resolve_circuit_key must prevent
    Spa-Francorchamps from triggering on Spanish GP queries.
    """

    def test_spanish_gp_resolves_to_barcelona(self):
        assert context_builder._resolve_circuit_key("tell me about the Spanish GP") == "barcelona"

    def test_spain_resolves_to_barcelona(self):
        assert context_builder._resolve_circuit_key("what happened in spain last year") == "barcelona"

    def test_spanish_grand_prix_strategy(self):
        assert context_builder._resolve_circuit_key("Spanish Grand Prix strategy") == "barcelona"

    def test_literal_spa_still_resolves_to_spa(self):
        assert context_builder._resolve_circuit_key("tell me about spa") == "spa"

    def test_spa_francorchamps_resolves_to_spa(self):
        assert context_builder._resolve_circuit_key("spa francorchamps sector 1") == "spa"

    def test_belgium_gp_resolves_to_spa(self):
        assert context_builder._resolve_circuit_key("belgium gp at spa") == "spa"

    def test_spanner_does_not_match_spa(self):
        assert context_builder._resolve_circuit_key("spanner in the works") == ""

    def test_spacecraft_does_not_match_spa(self):
        assert context_builder._resolve_circuit_key("spacecraft trajectory") == ""


# ── Context truncation regression tests ──────────────────────────────────────

def _minimal_mem():
    return {"episodic": [], "semantic": {}}


class TestContextTruncationLimits:
    """
    Confirms that long content in HISTORY, SESSION DATA, and DRIVER PROFILE
    reaches Claude at its real size, not silently truncated to a smaller limit.
    Each test builds a synthetic string just above the old broken limit and
    verifies the system prompt contains the FULL string.
    """

    def test_history_not_truncated_at_200(self):
        # HISTORICAL_DATA is 1153 chars; old limit was 200. Use 1100-char string.
        long_history = "H" * 1100
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), historical_context=long_history)
        assert long_history in prompt, (
            "HISTORY block was truncated — check the [:N] limit in ctx_blocks")

    def test_session_data_not_truncated_at_300(self):
        # get_practice_context can produce up to 800 chars; old limit was 300.
        long_session = "S" * 750
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), practice_context=long_session)
        assert long_session in prompt, (
            "SESSION DATA block was truncated — check the [:N] limit in ctx_blocks")

    def test_driver_profile_not_truncated_via_user_block(self):
        # build_driver_profile produces a multi-section block; old path merged it
        # into user_profile ([:150]). New DRIVER PROFILE block allows up to 1500.
        long_profile = "D" * 1200
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), driver_profile=long_profile)
        assert long_profile in prompt, (
            "DRIVER PROFILE block was truncated — confirm driver_deep_ctx is "
            "routed to driver_profile= kwarg, not merged into user_profile")

    def test_driver_profile_gets_own_block_label(self):
        profile = "VER profile data " * 20  # ~340 chars
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), driver_profile=profile)
        assert "DRIVER PROFILE:" in prompt

    def test_race_replay_not_truncated_via_user_block(self):
        # get_race_replay_context produces ~597 chars; old path merged into USER[:150]
        long_replay = "R" * 580
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), race_replay=long_replay)
        assert long_replay in prompt, (
            "RACE REPLAY content was truncated — confirm race_replay_ctx is "
            "routed to race_replay= kwarg, not merged into user_profile")

    def test_race_replay_gets_own_block_label(self):
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), race_replay="some replay data")
        assert "RACE REPLAY:" in prompt

    def test_champ_scenarios_not_truncated_via_user_block(self):
        # build_championship_scenarios produces ~1000 chars; old path → USER[:150]
        long_champ = "C" * 950
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), champ_scenarios=long_champ)
        assert long_champ in prompt, (
            "CHAMPIONSHIP SCENARIOS content was truncated — confirm "
            "champ_scenario_ctx is routed to champ_scenarios= kwarg")

    def test_champ_scenarios_gets_own_block_label(self):
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), champ_scenarios="NOR needs 50 pts")
        assert "CHAMPIONSHIP SCENARIOS:" in prompt

    def test_fan_profile_not_truncated_via_user_block(self):
        # build_fan_context produces ~311 chars; old path → USER[:150]
        long_fan = "F" * 300
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), fan_profile=long_fan)
        assert long_fan in prompt, (
            "FAN PROFILE content was truncated — confirm fan_ctx is "
            "routed to fan_profile= kwarg, not merged into user_profile")

    def test_fan_profile_gets_own_block_label(self):
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), fan_profile="Supports McLaren")
        assert "FAN PROFILE:" in prompt

    def test_user_block_still_present_when_only_user_profile_set(self):
        # USER block should still work for its intended short content
        prompt = context_builder.build_system_prompt(
            _minimal_mem(), user_profile="Prefers Spanish language")
        assert "USER:" in prompt


# ── cmd_reingest race-condition regression ────────────────────────────────────

class TestReingestRaceCondition:
    """
    Regression test for the cmd_reingest vs auto_ingest_loop silent data-loss bug.

    Root cause: cmd_reingest previously called load_f1_memory() (fresh disk read),
    creating a separate dict from mem_ref[0]. If auto_ingest_loop had modified
    mem_ref[0] in memory (e.g. wrote qualifying data) but hadn't yet saved when
    /reingest fired, reingest would read the OLD on-disk state, save, and then
    auto_ingest_loop would resume and overwrite disk with its stale copy —
    silently dropping the reingest changes.

    Fix: cmd_reingest now uses mem_ref[0] directly (same dict object as the
    background loops), so any in-flight mutations are always visible and preserved.
    """

    def test_reingest_preserves_concurrent_ingest_loop_changes(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        async def _run():
            # Simulate auto_ingest_loop mid-flight: qualifying data already written
            # to the shared in-memory dict but NOT yet saved to disk.
            initial_episode = {
                "round": 7,
                "race_name": "Spanish GP",
                "pole": "NOR",           # written by auto_ingest_loop mid-flight
                "pole_time": "1:11.383",
            }
            mem_ref = [{"episodic": [initial_episode], "semantic": {}}]

            fake_result = {
                "round": 7, "race_name": "Spanish GP",
                "winner": "NOR", "p2": "PIA", "p3": "RUS",
                "dnfs": [], "full_classification": ["NOR", "PIA", "RUS", "HAM", "VER"],
            }

            update = MagicMock()
            update.effective_user.id = int(bot.BOT_OWNER_ID)
            update.message.reply_text = AsyncMock()
            ctx = MagicMock()
            ctx.args = ["7"]

            saved_objects = []

            with patch.object(bot, "fetch_race_result", return_value=fake_result), \
                 patch.object(bot, "save_f1_memory",
                              side_effect=lambda m: saved_objects.append(m)), \
                 patch.object(bot, "load_ingest_state", return_value={}), \
                 patch.object(bot, "save_ingest_state"), \
                 patch.object(bot, "load_enrichment_state", return_value={}), \
                 patch.object(bot, "save_enrichment_state"), \
                 patch.object(bot, "load_predictor_state", return_value={}), \
                 patch.object(bot, "save_predictor_state"):
                await bot.cmd_reingest(update, ctx, mem_ref=mem_ref)

            episodes = mem_ref[0]["episodic"]
            r7 = next(e for e in episodes if e.get("round") == 7)

            # Race result from /reingest must be present
            assert r7.get("winner") == "NOR"
            # Qualifying data from auto_ingest_loop must NOT be overwritten
            assert r7.get("pole") == "NOR", (
                "Qualifying data written by auto_ingest_loop mid-flight was lost — "
                "cmd_reingest created a stale local copy instead of using mem_ref[0]"
            )
            # save_f1_memory must have been called with the shared object, not a copy
            assert saved_objects, "save_f1_memory was never called"
            assert saved_objects[-1] is mem_ref[0], (
                "save_f1_memory was called with a separate dict, not mem_ref[0] — "
                "a stale write could still overwrite concurrent changes"
            )

        asyncio.run(_run())


# ── _format_debug_context_report ─────────────────────────────────────────────

def _empty_ctx():
    return {
        "next_race_ctx": "", "news_ctx": "", "live_search_ctx": "",
        "fia_docs_ctx": "", "weather_ctx": "", "live_ctx": "",
        "practice_ctx": "", "circuit_ctx": "", "driver_stats_ctx": "",
        "driver_deep_ctx": "", "race_replay_ctx": "", "champ_scenario_ctx": "",
        "fan_ctx": "", "historical_ctx": "", "pred_accuracy": "",
        "user_profile_ctx": "",
    }


class TestFormatDebugContextReport:

    def test_empty_context_shows_no_blocks_triggered(self):
        report = context_builder._format_debug_context_report("test query", _empty_ctx())
        assert "(no context blocks triggered)" in report

    def test_populated_block_appears_with_label(self):
        ctx = _empty_ctx()
        ctx["circuit_ctx"] = "Barcelona: 4.657km, 16 turns, medium-high speed"
        report = context_builder._format_debug_context_report("barcelona strategy", ctx)
        assert "CIRCUIT:" in report
        assert "Barcelona" in report

    def test_char_count_reflects_limit(self):
        # NEWS limit is 300 — a 500-char string should report 300 chars
        ctx = _empty_ctx()
        ctx["news_ctx"] = "N" * 500
        report = context_builder._format_debug_context_report("news query", ctx)
        assert "NEWS: 300 chars" in report

    def test_preview_capped_at_100_chars(self):
        # HISTORY limit is 1200; raw string is 200 chars → sliced stays 200,
        # but the inline preview must be capped at 100.
        ctx = _empty_ctx()
        ctx["historical_ctx"] = "H" * 200
        report = context_builder._format_debug_context_report("history query", ctx)
        assert "HISTORY: 200 chars" in report
        preview_part = report.split("|")[1]
        assert preview_part.count("H") == 100

    def test_query_appears_in_header(self):
        report = context_builder._format_debug_context_report("antonelli in spain", _empty_ctx())
        assert "antonelli in spain" in report

    def test_empty_blocks_are_omitted(self):
        ctx = _empty_ctx()
        ctx["circuit_ctx"] = "Some circuit data"
        report = context_builder._format_debug_context_report("q", ctx)
        # Only CIRCUIT should appear; all other empty labels must not
        assert "NEWS:" not in report
        assert "WEATHER:" not in report
        assert "CIRCUIT:" in report


class TestCmdDebugContext:

    def test_non_owner_gets_no_reply(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        async def _run():
            mem_ref = [{"episodic": [], "semantic": {}}]
            update = MagicMock()
            update.effective_user.id = 9999999  # not the owner
            update.message.reply_text = AsyncMock()
            ctx = MagicMock()
            ctx.args = ["what happened to antonelli"]
            await bot.cmd_debug_context(update, ctx, mem_ref=mem_ref)
            update.message.reply_text.assert_not_called()

        asyncio.run(_run())

    def test_no_args_sends_usage(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        async def _run():
            mem_ref = [{"episodic": [], "semantic": {}}]
            update = MagicMock()
            update.effective_user.id = int(bot.BOT_OWNER_ID)
            update.message.reply_text = AsyncMock()
            ctx = MagicMock()
            ctx.args = []
            await bot.cmd_debug_context(update, ctx, mem_ref=mem_ref)
            call_text = update.message.reply_text.call_args[0][0]
            assert "Usage:" in call_text

        asyncio.run(_run())

    def test_valid_query_returns_formatted_report(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        async def _run():
            mem_ref = [{"episodic": [], "semantic": {}}]
            update = MagicMock()
            update.effective_user.id = int(bot.BOT_OWNER_ID)
            update.message.reply_text = AsyncMock()
            ctx_mock = MagicMock()
            ctx_mock.args = ["what", "happened", "to", "antonelli", "in", "spain"]
            fake_gathered = {
                **_empty_ctx(),
                "circuit_ctx":  "Barcelona guide text",
                "fia_docs_ctx": "FIA doc content for Spain",
            }
            with patch.object(bot, "_gather_context", return_value=fake_gathered):
                await bot.cmd_debug_context(update, ctx_mock, mem_ref=mem_ref)
            call_text = update.message.reply_text.call_args[0][0]
            assert "CIRCUIT:" in call_text
            assert "FIA_DOCS:" in call_text

        asyncio.run(_run())


# ── ContextBlock and _unpack_ctx ─────────────────────────────────────────────

class TestContextBlockMetadata:

    def test_context_block_construction(self):
        cb = context_builder.ContextBlock(content="test", data_age_hours=2.5, completeness="partial")
        assert cb.content == "test"
        assert cb.data_age_hours == 2.5
        assert cb.completeness == "partial"

    def test_context_block_defaults(self):
        cb = context_builder.ContextBlock(content="x")
        assert cb.data_age_hours is None
        assert cb.completeness == "full"

    def test_unpack_ctx_plain_string_returns_empty_meta(self):
        content, meta = context_builder._unpack_ctx("plain string")
        assert content == "plain string"
        assert meta == ""

    def test_unpack_ctx_full_block_no_age_returns_empty_meta(self):
        cb = context_builder.ContextBlock(content="data", completeness="full")
        content, meta = context_builder._unpack_ctx(cb)
        assert content == "data"
        assert meta == ""

    def test_unpack_ctx_with_age_formats_hours(self):
        cb = context_builder.ContextBlock(content="data", data_age_hours=3.0)
        content, meta = context_builder._unpack_ctx(cb)
        assert content == "data"
        assert "3.0h old" in meta

    def test_unpack_ctx_partial_completeness_in_meta(self):
        cb = context_builder.ContextBlock(content="data", completeness="partial")
        _, meta = context_builder._unpack_ctx(cb)
        assert "partial" in meta

    def test_unpack_ctx_unknown_completeness_in_meta(self):
        cb = context_builder.ContextBlock(content="data", completeness="unknown")
        _, meta = context_builder._unpack_ctx(cb)
        assert "unknown" in meta

    def test_unpack_ctx_age_and_partial_both_present(self):
        cb = context_builder.ContextBlock(content="data", data_age_hours=72.5, completeness="partial")
        _, meta = context_builder._unpack_ctx(cb)
        assert "72.5h old" in meta
        assert "partial" in meta

    def test_news_context_block_age_in_system_prompt(self):
        cb = context_builder.ContextBlock(content="Norris wins sprint", data_age_hours=0.4)
        prompt = context_builder.build_system_prompt(_minimal_mem(), news_context=cb)
        assert "NEWS" in prompt
        assert "0.4h old" in prompt
        assert "Norris wins sprint" in prompt

    def test_race_replay_partial_label_in_system_prompt(self):
        cb = context_builder.ContextBlock(content="race replay data", completeness="partial")
        prompt = context_builder.build_system_prompt(_minimal_mem(), race_replay=cb)
        assert "RACE REPLAY" in prompt
        assert "partial" in prompt

    def test_practice_context_block_age_in_system_prompt(self):
        cb = context_builder.ContextBlock(content="FP2 classification data", data_age_hours=5.2)
        prompt = context_builder.build_system_prompt(_minimal_mem(), practice_context=cb)
        assert "SESSION DATA" in prompt
        assert "5.2h old" in prompt

    def test_static_blocks_as_plain_strings_unaffected(self):
        prompt = context_builder.build_system_prompt(
            _minimal_mem(),
            circuit_guide="Barcelona circuit guide",
            driver_profile="VER profile data",
        )
        assert "CIRCUIT:" in prompt
        assert "Barcelona circuit guide" in prompt
        assert "DRIVER PROFILE:" in prompt
        assert "VER profile data" in prompt

    def test_format_debug_report_surfaces_age(self):
        ctx = _empty_ctx()
        ctx["news_ctx"] = context_builder.ContextBlock(content="N" * 200, data_age_hours=1.5)
        report = context_builder._format_debug_context_report("query", ctx)
        assert "1.5h old" in report
        assert "NEWS" in report

    def test_format_debug_report_surfaces_partial(self):
        ctx = _empty_ctx()
        ctx["race_replay_ctx"] = context_builder.ContextBlock(content="race data", completeness="partial")
        report = context_builder._format_debug_context_report("query", ctx)
        assert "partial" in report
        assert "RACE_REPLAY" in report

    def test_format_debug_report_plain_string_unchanged(self):
        ctx = _empty_ctx()
        ctx["circuit_ctx"] = "Barcelona guide"
        report = context_builder._format_debug_context_report("query", ctx)
        assert "CIRCUIT:" in report
        assert "Barcelona" in report


# ── Area 1: Predictor CSV staleness detection ─────────────────────────────────
#
# Findings:
#   EXISTS: round_num stamp check when expected_round is provided
#   GAP 1: expected_round=None (most user queries) skips ALL staleness checks
#   GAP 2: absent round_num in CSV + provided expected_round → silently served
#   NO mtime age gate; the mtime is only a cache invalidation key, not a
#   freshness check. Owner alerts on >24h age but serving is never blocked.

class TestPredictorCSVStaleness:

    @staticmethod
    def _make_csv(path, rows):
        import csv as _csv
        if not rows:
            path.write_text("")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_missing_csv_returns_empty(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            non_existent = Path(td) / "no_such.csv"
            with patch.object(data_layer, "PREDICTOR_CSV", non_existent):
                bot._PREDICTOR_CACHE.clear()
                block, rows = bot.get_predictor_context(expected_round=7)
        assert block == "" and rows == []

    def test_correct_round_is_served(self):
        from pathlib import Path
        import tempfile
        csv_rows = [{"round_num": "7", "code": "NOR",
                     "win_mc_pct": "45.0", "podium_mc_pct": "80.0"}]
        with tempfile.TemporaryDirectory() as td:
            csv_file = Path(td) / "pred.csv"
            self._make_csv(csv_file, csv_rows)
            with patch.object(data_layer, "PREDICTOR_CSV", csv_file):
                bot._PREDICTOR_CACHE.clear()
                _block, rows = bot.get_predictor_context(expected_round=7)
        assert rows and rows[0]["code"] == "NOR"

    def test_wrong_round_returns_empty(self):
        from pathlib import Path
        import tempfile
        csv_rows = [{"round_num": "6", "code": "NOR",
                     "win_mc_pct": "45.0", "podium_mc_pct": "80.0"}]
        with tempfile.TemporaryDirectory() as td:
            csv_file = Path(td) / "pred.csv"
            self._make_csv(csv_file, csv_rows)
            with patch.object(data_layer, "PREDICTOR_CSV", csv_file):
                bot._PREDICTOR_CACHE.clear()
                block, rows = bot.get_predictor_context(expected_round=7)
        assert block == "" and rows == []

    def test_absent_round_num_silently_served_gap(self):
        """Gap: missing round_num column + expected_round provided → except clause
        silently serves the stale CSV. Documented as intentional for pre-fix CSVs
        but is a risk if a post-fix CSV loses its round_num column."""
        from pathlib import Path
        import tempfile
        csv_rows = [{"code": "NOR", "win_mc_pct": "45.0"}]  # no round_num field
        with tempfile.TemporaryDirectory() as td:
            csv_file = Path(td) / "pred.csv"
            self._make_csv(csv_file, csv_rows)
            with patch.object(data_layer, "PREDICTOR_CSV", csv_file):
                bot._PREDICTOR_CACHE.clear()
                _block, rows = bot.get_predictor_context(expected_round=7)
        assert rows, (
            "Gap confirmed: absent round_num → stale CSV served without rejection "
            "even when expected_round is provided"
        )

    def test_no_expected_round_bypasses_staleness_check_gap(self):
        """Gap: get_predictor_context(expected_round=None) skips all round checks.
        A CSV from any past race weekend is served without warning. This is the
        most common call path for user queries (/predict, /winner)."""
        from pathlib import Path
        import tempfile
        csv_rows = [{"round_num": "1", "code": "NOR", "win_mc_pct": "45.0"}]
        with tempfile.TemporaryDirectory() as td:
            csv_file = Path(td) / "pred.csv"
            self._make_csv(csv_file, csv_rows)
            with patch.object(data_layer, "PREDICTOR_CSV", csv_file):
                bot._PREDICTOR_CACHE.clear()
                _block, rows = bot.get_predictor_context(expected_round=None)
        assert rows, (
            "Gap confirmed: round 1 CSV served silently when expected_round=None — "
            "no staleness check fires on the most common call path"
        )


# ── Area 2: Concurrent memory write safety ────────────────────────────────────
#
# Findings:
#   No asyncio.Lock around mem_ref writes anywhere in the codebase.
#   Both background loops do `mem = mem_ref[0]` (reference, not copy) then
#   `mem_ref[0] = mem` (no-op — same object). In-place mutations are visible
#   to all callers holding the same reference, so no stale-copy data loss.
#   cmd_debug_context: read-only, no write-back — safe.
#   cmd_reingest: in-place mutation, no mem_ref[0]=mem assignment — correct.
#   Gap: enrich_episode_with_telemetry overwrites full_classification — see Area 4.

class TestConcurrentMemoryWriteSafety:

    def test_two_async_writers_do_not_lose_data(self):
        """Two async tasks that both read mem_ref[0] and append to episodic
        must both see their writes persist, because they share the same dict object."""
        import asyncio

        async def writer_a(mem_ref):
            mem = mem_ref[0]
            await asyncio.sleep(0)  # yield to allow interleaving
            mem["episodic"].append({"round": 1, "winner": "NOR"})

        async def writer_b(mem_ref):
            mem = mem_ref[0]
            await asyncio.sleep(0)
            mem["episodic"].append({"round": 2, "winner": "VER"})

        async def _run():
            mem_ref = [{"episodic": [], "semantic": {}}]
            await asyncio.gather(writer_a(mem_ref), writer_b(mem_ref))
            return mem_ref[0]

        result = asyncio.run(_run())
        rounds = {ep["round"] for ep in result["episodic"]}
        assert rounds == {1, 2}, "Both writers' data must survive async interleaving"

    def test_loop_pattern_uses_same_dict_object(self):
        """Verify the pattern used by both background loops: mem = mem_ref[0] gives
        a reference to the same dict, not a copy. The write-back is a no-op."""
        mem_ref = [{"episodic": [], "semantic": {}}]
        mem = mem_ref[0]
        mem["episodic"].append({"round": 1})
        mem["episodic"].sort(key=lambda x: x.get("round", 0))
        mem_ref[0] = mem  # background loop write-back pattern

        assert mem_ref[0] is mem, "write-back must not create a new dict object"
        assert len(mem_ref[0]["episodic"]) == 1

    def test_cmd_debug_context_does_not_write_mem(self):
        """cmd_debug_context reads mem_ref[0] but must never write back to it."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        async def _run():
            mem_ref = [{"episodic": [{"round": 5, "winner": "NOR"}], "semantic": {}}]
            update = MagicMock()
            update.effective_user.id = int(bot.BOT_OWNER_ID)
            update.message.reply_text = AsyncMock()
            ctx = MagicMock()
            ctx.args = ["what happened"]
            with patch.object(bot, "_gather_context", return_value=_empty_ctx()):
                await bot.cmd_debug_context(update, ctx, mem_ref=mem_ref)
            return mem_ref[0]["episodic"]

        result = asyncio.run(_run())
        assert result == [{"round": 5, "winner": "NOR"}], (
            "cmd_debug_context must not modify mem_ref[0]['episodic']"
        )


# ── Area 3: Context block size enforcement ────────────────────────────────────
#
# Findings: All [:N] slices exist and are enforced. Undertested — no tests
# verified that the limits actually truncate (only that content is present).
# ContextBlock meta suffix appears in the label, not the content, so it does
# not reduce the content budget. Python str slicing is character-based
# (Unicode-safe) — no byte-level truncation issue.

class TestContextBlockSizeEnforcement:

    @staticmethod
    def _prompt(**kwargs):
        return context_builder.build_system_prompt(_minimal_mem(), **kwargs)

    def test_news_truncated_at_300(self):
        marker = "OVERFLOW_BEYOND_LIMIT"
        prompt = self._prompt(news_context="N" * 300 + marker)
        assert "N" * 300 in prompt
        assert marker not in prompt

    def test_session_data_truncated_at_800(self):
        marker = "OVERFLOW_BEYOND_LIMIT"
        prompt = self._prompt(practice_context="S" * 800 + marker)
        assert "S" * 800 in prompt
        assert marker not in prompt

    def test_circuit_guide_truncated_at_1500(self):
        marker = "OVERFLOW_BEYOND_LIMIT"
        prompt = self._prompt(circuit_guide="C" * 1500 + marker)
        assert "C" * 1500 in prompt
        assert marker not in prompt

    def test_race_replay_truncated_at_800(self):
        marker = "OVERFLOW_BEYOND_LIMIT"
        prompt = self._prompt(race_replay="R" * 800 + marker)
        assert "R" * 800 in prompt
        assert marker not in prompt

    def test_driver_profile_truncated_at_1500(self):
        marker = "OVERFLOW_BEYOND_LIMIT"
        prompt = self._prompt(driver_profile="D" * 1500 + marker)
        assert "D" * 1500 in prompt
        assert marker not in prompt

    def test_fia_docs_truncated_at_600(self):
        marker = "OVERFLOW_BEYOND_LIMIT"
        prompt = self._prompt(fia_docs_context="F" * 600 + marker)
        assert "F" * 600 in prompt
        assert marker not in prompt

    def test_context_block_meta_does_not_consume_content_budget(self):
        """The age/completeness suffix is in the LABEL, not sliced from content.
        An 800-char ContextBlock must inject all 800 content chars."""
        cb = context_builder.ContextBlock(content="Y" * 800, data_age_hours=3.5, completeness="partial")
        prompt = self._prompt(race_replay=cb)
        assert "Y" * 800 in prompt
        assert "3.5h old" in prompt
        assert "partial" in prompt

    def test_context_block_over_limit_content_is_truncated(self):
        """Even a ContextBlock respects the [:800] content limit."""
        marker = "OVERFLOW_BEYOND_LIMIT"
        cb = context_builder.ContextBlock(content="Z" * 800 + marker, data_age_hours=1.0)
        prompt = self._prompt(race_replay=cb)
        assert "Z" * 800 in prompt
        assert marker not in prompt

    def test_unicode_content_not_corrupted(self):
        """Python str slicing is character-based so accented chars must survive
        truncation intact (no mojibake, no replacement chars)."""
        unicode_news = "Sérgio Pérez dominó en Baku. " * 20  # well over 300 chars
        prompt = self._prompt(news_context=unicode_news)
        assert "Sérgio" in prompt
        assert "�" not in prompt  # no Unicode replacement character


# ── Area 4: Jolpica / OpenF1 conflict detection ───────────────────────────────
#
# Findings:
#   NO conflict detection exists anywhere.
#   Data sources are mostly partitioned (Jolpica = official results; FastF1/OpenF1
#   = telemetry). They don't write the same fields EXCEPT:
#   enrich_episode_with_telemetry overwrites full_classification if telemetry
#   provides full_order — this is the one field both sources can determine.
#   Impact: post-race penalties that change official standings (Jolpica reflects
#   them; FastF1 on-track order does not) are silently lost after enrichment.
#   Additionally, episode['winner'] (Jolpica) and full_classification[0] (FastF1)
#   can disagree with no detection or reconciliation.

class TestJolpicaOpenF1ConflictDetection:

    def test_enrichment_preserves_official_classification_when_present(self):
        """Fix: full_classification set by Jolpica (official, penalty-aware) must
        not be overwritten by FastF1 on-track order during enrichment.
        E.g. NOR wins after a VER time penalty — FastF1 records VER first on track,
        but the official classification correctly shows NOR P1."""
        from unittest.mock import MagicMock

        episode = {
            "round": 7,
            "winner": "NOR",
            "full_classification": ["P1:NOR", "P2:VER", "P3:PIA"],
        }
        # FastF1 on-track order: VER led to the flag before the penalty decision
        fake_telemetry = {"full_order": ["VER", "NOR", "PIA"], "source": "fastf1"}
        with patch.object(data_layer, "_safe_ff1_load", return_value=MagicMock()), \
             patch.object(data_layer, "_ff1_session_summary", return_value=fake_telemetry):
            enriched = bot.enrich_episode_with_telemetry(episode.copy(), round_num=7)

        assert enriched["full_classification"] == ["P1:NOR", "P2:VER", "P3:PIA"], (
            "Jolpica official classification must survive enrichment unchanged"
        )
        assert enriched["full_classification"][0] == "P1:NOR", (
            "P1 must remain NOR (official) not VER (on-track only)"
        )

    def test_winner_and_classification_consistent_after_enrichment(self):
        """After enrichment, episode['winner'] and full_classification[0] must agree.
        Before the fix they could silently diverge: winner=NOR (Jolpica) but
        full_classification[0]=P1:VER (FastF1). Now both reflect the official result."""
        from unittest.mock import MagicMock

        episode = {
            "round": 7,
            "winner": "NOR",
            "full_classification": ["P1:NOR", "P2:VER", "P3:PIA"],
        }
        fake_telemetry = {"full_order": ["VER", "NOR", "PIA"], "source": "fastf1"}
        with patch.object(data_layer, "_safe_ff1_load", return_value=MagicMock()), \
             patch.object(data_layer, "_ff1_session_summary", return_value=fake_telemetry):
            enriched = bot.enrich_episode_with_telemetry(episode.copy(), round_num=7)

        assert enriched["winner"] == "NOR"
        assert "P1:NOR" in enriched["full_classification"], (
            "winner and full_classification must agree on who won"
        )

    def test_no_conflict_when_sources_agree(self):
        """Happy path: when FastF1 on-track order matches official result,
        enrichment produces consistent episode data — Jolpica classification unchanged."""
        from unittest.mock import MagicMock

        episode = {
            "round": 7,
            "winner": "VER",
            "full_classification": ["P1:VER", "P2:NOR", "P3:PIA"],
        }
        fake_telemetry = {"full_order": ["VER", "NOR", "PIA"], "source": "fastf1"}
        with patch.object(data_layer, "_safe_ff1_load", return_value=MagicMock()), \
             patch.object(data_layer, "_ff1_session_summary", return_value=fake_telemetry):
            enriched = bot.enrich_episode_with_telemetry(episode.copy(), round_num=7)

        assert enriched["winner"] == "VER"
        assert enriched["full_classification"][0] == "P1:VER"

    def test_enrichment_does_not_overwrite_classification_when_telemetry_has_no_order(self):
        """If telemetry lacks full_order, official Jolpica classification is
        preserved unchanged (was already correct before the fix)."""
        from unittest.mock import MagicMock

        episode = {
            "round": 7,
            "winner": "NOR",
            "full_classification": ["P1:NOR", "P2:VER", "P3:PIA"],
        }
        # Telemetry without full_order (e.g., only sector data available)
        fake_telemetry = {"sector_bests": {"S1": "VER"}, "source": "openf1"}
        with patch.object(data_layer, "_safe_ff1_load", return_value=MagicMock()), \
             patch.object(data_layer, "_ff1_session_summary", return_value=fake_telemetry):
            enriched = bot.enrich_episode_with_telemetry(episode.copy(), round_num=7)

        assert enriched["full_classification"] == ["P1:NOR", "P2:VER", "P3:PIA"]


# ── news.py extraction regressions (Step 1) ──────────────────────────────────

class TestNewsModuleExtraction:

    def test_import_does_not_start_background_thread(self):
        """Importing news.py must not start the scheduler thread as a side effect.
        Verifies start_news_scheduler() is callable-only, not triggered at module level."""
        import threading
        import sys
        import importlib
        # Save the module already held by boxboxai_bot so we can restore it
        original_module = sys.modules.get("news")
        try:
            del sys.modules["news"]
            count_before = threading.active_count()
            importlib.import_module("news")
            count_after = threading.active_count()
            assert count_after == count_before, (
                f"news import started {count_after - count_before} unexpected thread(s)"
            )
        finally:
            # Always restore the original module so bot's function references
            # remain valid for subsequent tests
            if original_module is not None:
                sys.modules["news"] = original_module
            elif "news" in sys.modules:
                del sys.modules["news"]

    def test_get_news_cache_time_reflects_live_updates(self):
        """get_news_cache_time() returns the live _news_cache_time, not an
        import-time snapshot — regression for the reassignment-vs-mutation bug."""
        import news
        from datetime import datetime as _dt
        original = news._news_cache_time
        try:
            test_time = _dt(2026, 6, 19, 12, 0, 0)
            news._news_cache_time = test_time
            assert news.get_news_cache_time() == test_time
        finally:
            news._news_cache_time = original

    def test_get_news_cache_reflects_live_updates(self):
        """get_news_cache() returns the live _news_cache list, not an
        import-time snapshot — same pattern as get_news_cache_time."""
        import news
        original = news._news_cache
        try:
            fake = [{"title": "Test article", "url": "http://x"}]
            news._news_cache = fake
            assert news.get_news_cache() is fake
        finally:
            news._news_cache = original

    def test_bot_get_news_cache_time_reads_live_news_state(self):
        """bot.get_news_cache_time() (re-exported from news) must read the
        live news._news_cache_time, not a stale import-time snapshot.
        This verifies _gather_context's ContextBlock age annotation works
        end-to-end after the news.py extraction."""
        import news
        from datetime import datetime as _dt
        original = news._news_cache_time
        try:
            sentinel = _dt(2026, 6, 19, 15, 30, 0)
            news._news_cache_time = sentinel
            assert bot.get_news_cache_time() == sentinel
        finally:
            news._news_cache_time = original

    def test_enrichment_uses_fastf1_order_as_fallback_when_classification_absent(self):
        """If an episode has no full_classification (edge case: manual creation or
        a failed Jolpica call), FastF1's full_order is used to populate it."""
        from unittest.mock import MagicMock

        episode = {"round": 7, "winner": "VER"}  # no full_classification key
        fake_telemetry = {"full_order": ["VER", "NOR", "PIA"], "source": "fastf1"}
        with patch.object(data_layer, "_safe_ff1_load", return_value=MagicMock()), \
             patch.object(data_layer, "_ff1_session_summary", return_value=fake_telemetry):
            enriched = bot.enrich_episode_with_telemetry(episode.copy(), round_num=7)

        assert enriched.get("full_classification") == ["P1:VER", "P2:NOR", "P3:PIA"], (
            "FastF1 order must fill in full_classification when Jolpica hasn't set it"
        )


# ── data_layer.py extraction regressions (Step 4) ────────────────────────────

class TestDataLayerExtraction:

    def test_build_rich_story_works_without_ask_fn(self):
        """build_rich_story(episode, ask_fn=None) must return a non-empty narrative
        string without raising. Confirms the callable-injection seam is purely additive:
        the function assembles story from episode data and does not require ask_fn."""
        episode = {
            "round": 7,
            "race_name": "Spanish Grand Prix",
            "winner": "ANT",
            "p2": "RUS",
            "p3": "NOR",
            "fastest_lap": "ANT",
            "fastest_lap_time": "1:14.123",
            "champ_after": "ANT 110pts | RUS 88pts",
        }
        story = data_layer.build_rich_story(episode, ask_fn=None)
        assert isinstance(story, str)
        assert len(story) > 0, "build_rich_story must return non-empty string"
        assert "ANT" in story, "story must mention the winner"

    def test_build_rich_story_ask_fn_param_accepted(self):
        """build_rich_story must accept ask_fn as a keyword arg without error,
        so step 5 can wire in an LLM narrative pass without changing call sites."""
        episode = {"round": 1, "winner": "RUS", "p2": "ANT", "p3": "NOR"}
        # Pass a dummy callable — should be silently accepted (not yet used)
        result = data_layer.build_rich_story(episode, ask_fn=lambda x, **kw: "unused")
        assert isinstance(result, str)

    def test_predictor_cache_is_same_object_via_bot_and_data_layer(self):
        """bot._PREDICTOR_CACHE and data_layer._PREDICTOR_CACHE must be the
        same dict object. This ensures bot's _PREDICTOR_CACHE.clear() (called
        in _check_and_run_predictor) actually clears the cache used by
        get_predictor_context in data_layer."""
        assert bot._PREDICTOR_CACHE is data_layer._PREDICTOR_CACHE, (
            "bot imports _PREDICTOR_CACHE by reference — both names must point "
            "to the same dict so mutations in bot affect data_layer and vice versa"
        )

    def test_predictor_cache_mutation_visible_across_modules(self):
        """Mutating data_layer._PREDICTOR_CACHE must be visible through bot's name."""
        original = dict(data_layer._PREDICTOR_CACHE)
        try:
            data_layer._PREDICTOR_CACHE["__test__"] = "sentinel"
            assert bot._PREDICTOR_CACHE.get("__test__") == "sentinel", (
                "mutation via data_layer must be visible through bot's reference"
            )
            bot._PREDICTOR_CACHE.clear()
            assert len(data_layer._PREDICTOR_CACHE) == 0, (
                "bot.clear() must empty data_layer's cache too"
            )
        finally:
            data_layer._PREDICTOR_CACHE.clear()
            data_layer._PREDICTOR_CACHE.update(original)


# ── RACE_KEYWORDS normalization in get_race_replay_context (step 4 live-test fix) ──

class TestRaceReplayNormalization:
    """
    Regression tests for the RACE_KEYWORDS normalization fix.

    Root cause: get_race_replay_context used raw substring matching, so "spain"
    scored 0 against episode data keyed as track="Barcelona" / race_name="Spanish
    Grand Prix" — neither contains "spain" as a literal substring.

    Fix: scoring loop and in-memory guard now use RACE_KEYWORDS to normalise
    country names to the adjective that appears in race_name (e.g. "spain" →
    "Spanish", "mexico" → "Mexico", "austin" → "United States").
    """

    def _ep(self, round_num, race_name, track, winner="HAM", p2="RUS", p3="NOR"):
        return {
            "round": round_num,
            "race_name": race_name,
            "track": track,
            "winner": winner,
            "p2": p2,
            "p3": p3,
            "date": "2026-06-14",
            "story": f"{winner} wins the {race_name}.",
        }

    def _mem(self, *episodes):
        return {"episodic": list(episodes)}

    # ── Spain / Barcelona ──────────────────────────────────────────────────────

    def test_spain_matches_spanish_grand_prix_episode(self):
        """'what happened in qualifying spain' must score R7 Spain, not return ''."""
        mem = self._mem(self._ep(7, "Spanish Grand Prix", "Barcelona"))
        result = context_builder.get_race_replay_context(
            "what happened in qualifying spain", mem)
        assert result != "", (
            "'spain' must match episode with race_name='Spanish Grand Prix'"
        )
        content = result.content if hasattr(result, "content") else str(result)
        assert "Spanish" in content or "Barcelona" in content or "HAM" in content

    def test_barcelona_matches_spanish_grand_prix_episode(self):
        """'what happened in barcelona' must score the Spain episode."""
        mem = self._mem(self._ep(7, "Spanish Grand Prix", "Barcelona"))
        result = context_builder.get_race_replay_context(
            "what happened in barcelona", mem)
        assert result != ""

    def test_spain_not_in_memory_returns_empty(self):
        """If R7 Spain is not in memory, 'spain' query must return '' (no wrong-race fallback)."""
        mem = self._mem(self._ep(5, "Canadian Grand Prix", "Montreal"))
        result = context_builder.get_race_replay_context(
            "what happened in qualifying spain", mem)
        assert result == "", (
            "must not fall back to Canadian GP when Spain isn't in memory"
        )

    # ── Mexico / Mexico City ───────────────────────────────────────────────────

    def test_mexico_matches_mexico_city_grand_prix_episode(self):
        """'explain the strategy in mexico' must match episode with race_name containing 'Mexico'."""
        mem = self._mem(self._ep(17, "Mexico City Grand Prix", "Mexico City"))
        result = context_builder.get_race_replay_context(
            "explain the strategy in mexico", mem)
        assert result != "", (
            "'mexico' must match episode with race_name='Mexico City Grand Prix'"
        )

    def test_mexico_not_in_memory_returns_empty(self):
        """If Mexico isn't in memory, 'mexico' query must not fall back to a different race."""
        mem = self._mem(self._ep(7, "Spanish Grand Prix", "Barcelona"))
        result = context_builder.get_race_replay_context(
            "explain the strategy in mexico", mem)
        assert result == "", (
            "must not return Spanish GP data for a Mexico query"
        )

    # ── Austria / Spielberg ───────────────────────────────────────────────────

    def test_spielberg_matches_austrian_grand_prix_episode(self):
        """'spielberg' (Red Bull Ring locality) must now match Austrian GP episode."""
        mem = self._mem(self._ep(8, "Austrian Grand Prix", "Spielberg"))
        result = context_builder.get_race_replay_context(
            "what happened at spielberg", mem)
        assert result != "", (
            "'spielberg' must match episode with race_name='Austrian Grand Prix'"
        )

    # ── RACE_KEYWORDS is the single source of truth ───────────────────────────

    def test_race_keywords_module_constant_exists(self):
        """RACE_KEYWORDS must be a module-level dict, not a local inside _gather_context."""
        assert hasattr(bot, "RACE_KEYWORDS"), (
            "RACE_KEYWORDS must be a module-level constant"
        )
        assert isinstance(context_builder.RACE_KEYWORDS, dict)
        assert "spain" in context_builder.RACE_KEYWORDS
        assert context_builder.RACE_KEYWORDS["spain"] == "Spanish"
        assert "barcelona" in context_builder.RACE_KEYWORDS
        assert "mexico" in context_builder.RACE_KEYWORDS
        assert "spielberg" in context_builder.RACE_KEYWORDS
        assert "austin" in context_builder.RACE_KEYWORDS
