"""How a defense and an attack (or access code) become chat messages.

Tensor Trust adds no prompt text of its own: the model sees the defender's
opening defense (``pre_prompt``), the attack or true access code, and the
closing defense (``post_prompt``), each as its own message, as in the game and
in the paper's GPT-family evaluation (Toyer et al., 2023, Section 5.3 and
Appendix C.2):

* opening defense: ``system``;
* attack or access code: ``user``;
* closing defense: a separate ``user`` message.

Source: ``OpenAIAdapter._get_prediction_messages`` in
https://github.com/HumanCompatibleAI/tensor-trust/blob/f0b055451415e23d444703fbb8a47daf1d4b2c3c/data-pipeline/src/ttdata/openai_adapter.py
Empty opening or closing defenses are sent as empty messages, as there.
"""

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser


def defense_messages(
    pre_prompt: str, attack_or_access_code: str, post_prompt: str
) -> list[ChatMessage]:
    """The three messages the model under test sees, in the paper's roles."""
    return [
        ChatMessageSystem(content=pre_prompt),
        ChatMessageUser(content=attack_or_access_code),
        ChatMessageUser(content=post_prompt),
    ]
