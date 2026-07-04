# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Automated LEAPS buy-signal strategy for AAPL (later MSFT, GOOGL), validated by backtesting against historical data with auto-research parameter tuning.

## Strategy specification

The full strategy spec lives in `.claude/skills/buildAaplLeaps/SKILL.md` (invocable as the `buildAaplLeaps` skill). It defines the dual-mode entry signals (Mode A: MACD bottom-divergence + VIX filter; Mode B: trend-pullback continuation), option selection via BSM pricing, tiered exit rules, the parameter search space, and the auto-research protocol. **Read it before touching any strategy logic — it is the living spec that code must follow.**

## Code status

No code yet — the spec precedes implementation. The planned architecture is in the SKILL.md; `options.py`, `metrics.py`, and `portfolio.py` are intended to be ported from the QQQ version at `~/repo/leaps/leaps/`. Once code lands, update this file with build/test/run commands.
