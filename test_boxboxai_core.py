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
        with patch.object(bot, "_get_circuit_zone_data",
                          return_value=_BARCELONA_ZONE_DATA):
            guide = bot.get_circuit_guide("what about barcelona strategy")
        assert "PRECISE 2026 ZONE DATA" in guide
        assert "Apex T13" in guide
        assert "Entry T14" in guide
        assert "A1" in guide
        assert "45m after T14" in guide

    def test_qualitative_guide_still_present(self):
        with patch.object(bot, "_get_circuit_zone_data",
                          return_value=_BARCELONA_ZONE_DATA):
            guide = bot.get_circuit_guide("barcelona")
        # Qualitative content from CIRCUIT_GUIDES must still be there
        assert "CIRCUIT GUIDE" in guide
        assert "BARCELONA" in guide

    def test_graceful_degradation_when_no_zone_data(self):
        with patch.object(bot, "_get_circuit_zone_data", return_value={}):
            guide = bot.get_circuit_guide("barcelona")
        assert "CIRCUIT GUIDE" in guide
        assert "PRECISE 2026 ZONE DATA" not in guide

    def test_no_zone_data_for_unknown_circuit(self):
        # No circuit match → empty string regardless of zone data
        with patch.object(bot, "_get_circuit_zone_data", return_value={}):
            guide = bot.get_circuit_guide("what is F1?")
        assert guide == ""

    def test_zone_suffix_exception_does_not_propagate(self):
        # If _get_circuit_zone_data raises, guide still returns qualitative text
        with patch.object(bot, "_get_circuit_zone_data",
                          side_effect=RuntimeError("network down")):
            guide = bot.get_circuit_guide("barcelona")
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
        assert bot._resolve_circuit_key("tell me about the Spanish GP") == "barcelona"

    def test_spain_resolves_to_barcelona(self):
        assert bot._resolve_circuit_key("what happened in spain last year") == "barcelona"

    def test_spanish_grand_prix_strategy(self):
        assert bot._resolve_circuit_key("Spanish Grand Prix strategy") == "barcelona"

    def test_literal_spa_still_resolves_to_spa(self):
        assert bot._resolve_circuit_key("tell me about spa") == "spa"

    def test_spa_francorchamps_resolves_to_spa(self):
        assert bot._resolve_circuit_key("spa francorchamps sector 1") == "spa"

    def test_belgium_gp_resolves_to_spa(self):
        assert bot._resolve_circuit_key("belgium gp at spa") == "spa"

    def test_spanner_does_not_match_spa(self):
        assert bot._resolve_circuit_key("spanner in the works") == ""

    def test_spacecraft_does_not_match_spa(self):
        assert bot._resolve_circuit_key("spacecraft trajectory") == ""


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
        prompt = bot.build_system_prompt(
            _minimal_mem(), historical_context=long_history)
        assert long_history in prompt, (
            "HISTORY block was truncated — check the [:N] limit in ctx_blocks")

    def test_session_data_not_truncated_at_300(self):
        # get_practice_context can produce up to 800 chars; old limit was 300.
        long_session = "S" * 750
        prompt = bot.build_system_prompt(
            _minimal_mem(), practice_context=long_session)
        assert long_session in prompt, (
            "SESSION DATA block was truncated — check the [:N] limit in ctx_blocks")

    def test_driver_profile_not_truncated_via_user_block(self):
        # build_driver_profile produces a multi-section block; old path merged it
        # into user_profile ([:150]). New DRIVER PROFILE block allows up to 1500.
        long_profile = "D" * 1200
        prompt = bot.build_system_prompt(
            _minimal_mem(), driver_profile=long_profile)
        assert long_profile in prompt, (
            "DRIVER PROFILE block was truncated — confirm driver_deep_ctx is "
            "routed to driver_profile= kwarg, not merged into user_profile")

    def test_driver_profile_gets_own_block_label(self):
        profile = "VER profile data " * 20  # ~340 chars
        prompt = bot.build_system_prompt(
            _minimal_mem(), driver_profile=profile)
        assert "DRIVER PROFILE:" in prompt

    def test_race_replay_not_truncated_via_user_block(self):
        # get_race_replay_context produces ~597 chars; old path merged into USER[:150]
        long_replay = "R" * 580
        prompt = bot.build_system_prompt(
            _minimal_mem(), race_replay=long_replay)
        assert long_replay in prompt, (
            "RACE REPLAY content was truncated — confirm race_replay_ctx is "
            "routed to race_replay= kwarg, not merged into user_profile")

    def test_race_replay_gets_own_block_label(self):
        prompt = bot.build_system_prompt(
            _minimal_mem(), race_replay="some replay data")
        assert "RACE REPLAY:" in prompt

    def test_champ_scenarios_not_truncated_via_user_block(self):
        # build_championship_scenarios produces ~1000 chars; old path → USER[:150]
        long_champ = "C" * 950
        prompt = bot.build_system_prompt(
            _minimal_mem(), champ_scenarios=long_champ)
        assert long_champ in prompt, (
            "CHAMPIONSHIP SCENARIOS content was truncated — confirm "
            "champ_scenario_ctx is routed to champ_scenarios= kwarg")

    def test_champ_scenarios_gets_own_block_label(self):
        prompt = bot.build_system_prompt(
            _minimal_mem(), champ_scenarios="NOR needs 50 pts")
        assert "CHAMPIONSHIP SCENARIOS:" in prompt

    def test_fan_profile_not_truncated_via_user_block(self):
        # build_fan_context produces ~311 chars; old path → USER[:150]
        long_fan = "F" * 300
        prompt = bot.build_system_prompt(
            _minimal_mem(), fan_profile=long_fan)
        assert long_fan in prompt, (
            "FAN PROFILE content was truncated — confirm fan_ctx is "
            "routed to fan_profile= kwarg, not merged into user_profile")

    def test_fan_profile_gets_own_block_label(self):
        prompt = bot.build_system_prompt(
            _minimal_mem(), fan_profile="Supports McLaren")
        assert "FAN PROFILE:" in prompt

    def test_user_block_still_present_when_only_user_profile_set(self):
        # USER block should still work for its intended short content
        prompt = bot.build_system_prompt(
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
        report = bot._format_debug_context_report("test query", _empty_ctx())
        assert "(no context blocks triggered)" in report

    def test_populated_block_appears_with_label(self):
        ctx = _empty_ctx()
        ctx["circuit_ctx"] = "Barcelona: 4.657km, 16 turns, medium-high speed"
        report = bot._format_debug_context_report("barcelona strategy", ctx)
        assert "CIRCUIT:" in report
        assert "Barcelona" in report

    def test_char_count_reflects_limit(self):
        # NEWS limit is 300 — a 500-char string should report 300 chars
        ctx = _empty_ctx()
        ctx["news_ctx"] = "N" * 500
        report = bot._format_debug_context_report("news query", ctx)
        assert "NEWS: 300 chars" in report

    def test_preview_capped_at_100_chars(self):
        # HISTORY limit is 1200; raw string is 200 chars → sliced stays 200,
        # but the inline preview must be capped at 100.
        ctx = _empty_ctx()
        ctx["historical_ctx"] = "H" * 200
        report = bot._format_debug_context_report("history query", ctx)
        assert "HISTORY: 200 chars" in report
        preview_part = report.split("|")[1]
        assert preview_part.count("H") == 100

    def test_query_appears_in_header(self):
        report = bot._format_debug_context_report("antonelli in spain", _empty_ctx())
        assert "antonelli in spain" in report

    def test_empty_blocks_are_omitted(self):
        ctx = _empty_ctx()
        ctx["circuit_ctx"] = "Some circuit data"
        report = bot._format_debug_context_report("q", ctx)
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
        cb = bot.ContextBlock(content="test", data_age_hours=2.5, completeness="partial")
        assert cb.content == "test"
        assert cb.data_age_hours == 2.5
        assert cb.completeness == "partial"

    def test_context_block_defaults(self):
        cb = bot.ContextBlock(content="x")
        assert cb.data_age_hours is None
        assert cb.completeness == "full"

    def test_unpack_ctx_plain_string_returns_empty_meta(self):
        content, meta = bot._unpack_ctx("plain string")
        assert content == "plain string"
        assert meta == ""

    def test_unpack_ctx_full_block_no_age_returns_empty_meta(self):
        cb = bot.ContextBlock(content="data", completeness="full")
        content, meta = bot._unpack_ctx(cb)
        assert content == "data"
        assert meta == ""

    def test_unpack_ctx_with_age_formats_hours(self):
        cb = bot.ContextBlock(content="data", data_age_hours=3.0)
        content, meta = bot._unpack_ctx(cb)
        assert content == "data"
        assert "3.0h old" in meta

    def test_unpack_ctx_partial_completeness_in_meta(self):
        cb = bot.ContextBlock(content="data", completeness="partial")
        _, meta = bot._unpack_ctx(cb)
        assert "partial" in meta

    def test_unpack_ctx_unknown_completeness_in_meta(self):
        cb = bot.ContextBlock(content="data", completeness="unknown")
        _, meta = bot._unpack_ctx(cb)
        assert "unknown" in meta

    def test_unpack_ctx_age_and_partial_both_present(self):
        cb = bot.ContextBlock(content="data", data_age_hours=72.5, completeness="partial")
        _, meta = bot._unpack_ctx(cb)
        assert "72.5h old" in meta
        assert "partial" in meta

    def test_news_context_block_age_in_system_prompt(self):
        cb = bot.ContextBlock(content="Norris wins sprint", data_age_hours=0.4)
        prompt = bot.build_system_prompt(_minimal_mem(), news_context=cb)
        assert "NEWS" in prompt
        assert "0.4h old" in prompt
        assert "Norris wins sprint" in prompt

    def test_race_replay_partial_label_in_system_prompt(self):
        cb = bot.ContextBlock(content="race replay data", completeness="partial")
        prompt = bot.build_system_prompt(_minimal_mem(), race_replay=cb)
        assert "RACE REPLAY" in prompt
        assert "partial" in prompt

    def test_practice_context_block_age_in_system_prompt(self):
        cb = bot.ContextBlock(content="FP2 classification data", data_age_hours=5.2)
        prompt = bot.build_system_prompt(_minimal_mem(), practice_context=cb)
        assert "SESSION DATA" in prompt
        assert "5.2h old" in prompt

    def test_static_blocks_as_plain_strings_unaffected(self):
        prompt = bot.build_system_prompt(
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
        ctx["news_ctx"] = bot.ContextBlock(content="N" * 200, data_age_hours=1.5)
        report = bot._format_debug_context_report("query", ctx)
        assert "1.5h old" in report
        assert "NEWS" in report

    def test_format_debug_report_surfaces_partial(self):
        ctx = _empty_ctx()
        ctx["race_replay_ctx"] = bot.ContextBlock(content="race data", completeness="partial")
        report = bot._format_debug_context_report("query", ctx)
        assert "partial" in report
        assert "RACE_REPLAY" in report

    def test_format_debug_report_plain_string_unchanged(self):
        ctx = _empty_ctx()
        ctx["circuit_ctx"] = "Barcelona guide"
        report = bot._format_debug_context_report("query", ctx)
        assert "CIRCUIT:" in report
        assert "Barcelona" in report
