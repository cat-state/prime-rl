"""CPU checks for bounded context projection without modifying replay evidence."""
import copy

from tools.painting_dataset.agentic_episode import EpisodeConfig, next_messages


def main():
    first = {"role": "user", "content": [{"type": "text", "text": "original target contract"}]}
    history = [first, {"role": "assistant", "content": "OLD REASONING"},
               {"role": "user", "content": [{"type": "text", "text": "old feedback"}]}]
    source = "fn paint(uv:vec2f)->vec3f {return vec3f(1.0);}"
    response = {"finish_reason": "stop", "raw_output": "REASONING<|content_text|>```wgsl\n" + source + "\n```<|end_message|>",
                "completion_ids": [1, 2, 3], "routed_experts": {"data": "preserved"}}
    original = copy.deepcopy(response)
    feedback = [{"type": "text", "text": "latest error"}]
    bounded = next_messages(history, response, feedback, EpisodeConfig(history_policy="bounded-v2"))
    assert len(bounded) == 3 and bounded[0] == first
    assert bounded[1]["content"] == "```wgsl\n" + source + "\n```"
    assert bounded[2]["content"] == feedback
    assert response == original  # Structural evidence preservation, not numerical accuracy.
    truncated = dict(response, finish_reason="length")
    assert next_messages(history, truncated, feedback, EpisodeConfig(history_policy="bounded-v2")) == [first, {"role": "user", "content": feedback}]
    full = next_messages(history, response, feedback, EpisodeConfig())
    assert full[:3] == history and full[3]["content"] == response["raw_output"]
    for _ in range(20):
        bounded = next_messages(bounded, response, feedback, EpisodeConfig(history_policy="bounded-v2"))
        assert len(bounded) == 3
    print("PASS: bounded latest action, omitted incomplete reasoning, unchanged raw evidence, stable context")


if __name__ == "__main__":
    main()
