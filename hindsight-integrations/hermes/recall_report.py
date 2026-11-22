"""What one automatic recall did, stage by stage, for the user-facing indicator and the log.

Hermes renders a provider's recall indicator as ``"<glyph> <provider_label> — recalled N
memories"`` on its status channel, which never enters the chat history or the model's context.
The core owns that frame and the count; the label is the provider's, so the breakdown rides in
it: ``Hindsight (76 found · 20 already shown · 31 pruned in 0.6s)``.
"""

from __future__ import annotations

from dataclasses import dataclass

PROVIDER_LABEL = "Hindsight"


@dataclass(frozen=True)
class RecallReport:
    """Counts of one recall. ``already_shown`` is None with session dedup off; ``pruned`` is
    None when the relevance gate did not grade (off, too few items, or skipped — then
    ``prune_skipped`` says why). ``found`` is 0 for a reflect, which has no item count."""

    found: int
    injected: int
    already_shown: int | None = None
    pruned: int | None = None
    prune_ms: float | None = None
    prune_skipped: str | None = None

    def _stages(self) -> list[str]:
        stages = [f"{self.found} found"]
        if self.already_shown:
            stages.append(f"{self.already_shown} already shown")
        if self.pruned is not None:
            timing = f" in {self.prune_ms / 1000:.1f}s" if self.prune_ms is not None else ""
            stages.append(f"{self.pruned} pruned{timing}")
        elif self.prune_skipped:
            stages.append(f"prune skipped ({self.prune_skipped})")
        return stages

    def indicator_label(self) -> str:
        """The provider label for Hermes' recall indicator."""
        if not self.found:
            return PROVIDER_LABEL
        return f"{PROVIDER_LABEL} ({' · '.join(self._stages())})"

    def log_line(self) -> str:
        """One INFO line per recall, including the turns the indicator cannot show (empty block)."""
        if not self.found:
            return "recall: nothing found"
        outcome = f"{self.injected} injected" if self.injected else "nothing injected (block empty)"
        return f"recall: {', '.join(self._stages())} -> {outcome}"

    def empty_after_prune(self) -> bool:
        """Recall found memories but the gate judged none of them worth showing."""
        return self.found > 0 and self.injected == 0 and bool(self.pruned)
