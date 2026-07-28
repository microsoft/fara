"""Fara-1.5 web automation agent (FaraNext browser action set).

Single-class port of agento_next's FaraQwen3NextAgent: the browser
observe-think-act loop, the expanded action space (double/right/triple
click, left_click_drag, hscroll, read_page_answer_question,
ask_user_question), and DataPoint trajectory logging via RunContext.
"""

from __future__ import annotations

import ast
import copy
import io
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

import openai
from PIL import Image
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from ._prompts import get_computer_use_system_prompt
from ...clients.wrapper import ChatCompletionClient, extract_message_text
from ...clients.create_utils import create_client_from_config
from ...clients.messages import (
    LLMMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ImageObj,
    FunctionCall,
)
from ...core.agent import Agent, AgentConfig
from ...core.run_context import RunContext
from ...core.data_point import (
    Action as DPAction,
    AgentState,
    LLMConversation,
    LLMMessage as DPLLMMessage,
    SolverStatus,
    Outcome,
    ComputerObservation,
)
from ..captcha import wait_for_captcha
from ..coord_spaces import FARA_DISPLAY_SIZE
from ..computer_agent.utils import extract_from_page
from ..utils import format_text_observation, get_trimmed_url
from .fara_types import WebSurferEvent


def extract_allowed_actions(system_prompt_text: str) -> frozenset[str]:
    """Extract the allowed action names from the tool schema enum in the system prompt."""
    match = re.search(r'"enum":\s*\[([^\]]+)\]', system_prompt_text)
    if not match:
        raise ValueError("Could not find action enum in system prompt")
    return frozenset(json.loads(f"[{match.group(1)}]"))


class Fara15AgentConfig(AgentConfig):
    """Configuration for Fara15Agent.

    Notable fields:
      extra_create_args: Extra kwargs merged on top of ``temperature: 0`` and
        forwarded to ``chat.completions.create``.
      auto_user_reply: When True, ``ask_user_question`` does NOT halt the run —
        a fixed dummy user response is injected and the agent continues (for
        eval / no-user-simulator settings only).
      captcha_timeout_limit: Consecutive captcha-wait timeouts tolerated in a
        single run before the per-step captcha gate is auto-disabled. ``0``
        skips the gate from the start (appropriate without a Browserbase
        session, which is what triggers the captcha solver).
      raise_on_captcha_timeout: When True, raise on the first captcha-wait
        timeout instead of moving past the unsolved captcha.
    """

    name: str = "fara_next"
    client_config: dict[str, Any] | None = None
    max_rounds: int = 10
    max_n_images: int = 3
    max_observation_chars: int = 1000
    fn_call_template: str = "fara-qwen3vl"
    identity: str | None = "fara_qwen35"
    critical_points: str | None = "fara-1.5"
    computer_use_mode: str = "fara_next_browser"
    save_screenshots: bool = False
    include_input_text_key_args: bool = True
    viewport_width: int = 1440
    viewport_height: int = 900
    min_pixels: int = 3136
    max_pixels: int = 12845056
    extra_create_args: dict[str, Any] | None = None
    auto_user_reply: bool = False
    captcha_timeout_limit: int = 2
    raise_on_captcha_timeout: bool = True
    terminate_on_parse_error: bool = False
    max_parse_retries: int = 2
    image_token_estimate: int = 1500
    image_budget_token_cap: int = 0


@dataclass
class Fara15AgentState:
    """Mutable state that persists across resume calls."""

    chat_history: list[LLMMessage] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    current_step: int = 0
    mlm_width: int = 0
    mlm_height: int = 0


class Fara15Agent(Agent):
    """Web automation agent with the FaraNext browser action set."""

    DISPLAY_SIZE = FARA_DISPLAY_SIZE
    PATCH_SIZE = 16
    MERGE_SIZE = 2
    USER_MESSAGE = "Here is the next screenshot. Think about what to do next."
    MAX_URL_LENGTH = 100

    _TEXT_OBSERVATION_ACTIONS: frozenset[str] = frozenset({"read_page_answer_question"})

    _MODE_CANONICAL_IDENTITY: dict[str, tuple[str, ...]] = {
        "fara_next_browser": ("fara_qwen3vl", "fara_qwen35"),
        "fara_next_windows": ("fara_qwen3vl_windows", "fara_qwen35_windows"),
        "fara_next_windows_core": ("fara_qwen3vl_windows", "fara_qwen35_windows"),
    }

    def __init__(
        self, config: Fara15AgentConfig | dict[str, Any] | None = None, **kwargs: Any
    ):
        super().__init__(config, **kwargs)
        self.config: Fara15AgentConfig
        self.logger = logging.getLogger(__name__)
        self._client: ChatCompletionClient | None = None
        self._state: Fara15AgentState | None = None
        self._pending_observation: str = ""
        self._allowed_actions: frozenset[str] = frozenset()
        self._captcha_timeouts: int = 0
        self._captcha_disabled: bool = self.config.captcha_timeout_limit <= 0

    @property
    def mlm_processor_im_cfg(self) -> dict[str, int]:
        return {
            "min_pixels": self.config.min_pixels,
            "max_pixels": self.config.max_pixels,
            "patch_size": self.PATCH_SIZE,
            "merge_size": self.MERGE_SIZE,
        }

    @classmethod
    def _get_config_class(cls) -> type[Fara15AgentConfig]:
        return Fara15AgentConfig

    async def initialize(self, run_context: RunContext) -> None:
        """Initialize the agent for a run."""
        await super().initialize(run_context)
        self._state = Fara15AgentState(
            mlm_width=self.config.viewport_width,
            mlm_height=self.config.viewport_height,
        )

        if self.config.client_config is not None:
            self._client = create_client_from_config(self.config.client_config)
        elif self.config.client is not None:
            self._client = self.config.client
        else:
            raise ValueError("Either client or client_config must be provided")

        valid_modes = set(self._MODE_CANONICAL_IDENTITY)
        if self.config.computer_use_mode not in valid_modes:
            raise ValueError(
                f"Unknown computer_use_mode '{self.config.computer_use_mode}'. "
                f"Known values: {sorted(valid_modes)}"
            )
        canonical_identities = self._MODE_CANONICAL_IDENTITY[
            self.config.computer_use_mode
        ]
        if (
            self.config.identity is not None
            and self.config.identity not in canonical_identities
        ):
            raise ValueError(
                f"identity={self.config.identity!r} does not match any canonical "
                f"identity in {list(canonical_identities)!r} for "
                f"computer_use_mode={self.config.computer_use_mode!r}. "
                f"Use one of {list(canonical_identities)!r} or set identity=None "
                f"to auto-detect."
            )

    async def run(
        self, run_context: RunContext, input: Any = None
    ) -> Tuple[str, List, List]:
        """Run the agent on a task.

        Supports resuming after WAITING_FOR_USER: if ``_state.chat_history`` is
        already populated the fresh-start init is skipped and the user response
        is appended to the conversation.
        """
        env = run_context.environment

        traj = run_context.solver_log

        pending_user_response = ""
        if self._state.chat_history:
            traj.status = SolverStatus.RUNNING
            pending_user_response = traj.get_last_user_message() or ""
            start_step = self._state.current_step
        else:
            task_instruction = run_context.task.instruction
            scaled_screenshot = await self._get_scaled_screenshot(env)
            await self._save_screenshot(
                env, run_context.output_dir, "screenshot_0_pre.png"
            )
            self._log_initial_observations(run_context, task_instruction)

            self._state.chat_history.append(
                UserMessage(
                    content=[ImageObj.from_pil(scaled_screenshot), task_instruction],
                    metadata={"is_original": True},
                )
            )
            start_step = 0

        all_actions = []
        all_observations = []
        final_answer = "<no_answer>"

        for step in range(start_step + 1, self.config.max_rounds + 1):
            is_first_round = step == 1

            if not self._captcha_disabled:
                if not await wait_for_captcha(env):
                    if self.config.raise_on_captcha_timeout:
                        raise RuntimeError(
                            "Captcha timed out, unable to proceed with web surfing."
                        )
                    self._captcha_timeouts += 1
                    self.logger.warning(
                        "Captcha timed out at step %d (%d/%d); moving past "
                        "unsolved captcha.",
                        step,
                        self._captcha_timeouts,
                        self.config.captcha_timeout_limit,
                    )
                    if self._captcha_timeouts >= self.config.captcha_timeout_limit:
                        self.logger.warning(
                            "Captcha timed out %d times; disabling further "
                            "captcha waits for the rest of this run.",
                            self._captcha_timeouts,
                        )
                        self._captcha_disabled = True

            pre_screenshot_name = f"screenshot_{step}_pre.png"
            await self._save_screenshot(
                env, run_context.output_dir, pre_screenshot_name
            )
            current_url, page_context = await env.get_page_context()

            function_call, raw_response = await self._generate_model_call(
                env,
                is_first_round,
                scaled_screenshot if is_first_round else None,
                user_response=pending_user_response,
            )
            pending_user_response = ""
            all_actions.append(raw_response)

            action_args = function_call[0].arguments
            action_name = action_args.get("action", "unknown")
            thoughts = action_args.get("thoughts", "")

            self.logger.debug(
                f"\nThought #{step}: {thoughts}\nAction #{step}: executing tool '{action_name}' with arguments {json.dumps(action_args)}"
            )

            run_context.add_observation(
                ComputerObservation(
                    screenshot_path=pre_screenshot_name,
                    url=current_url,
                    page_info=page_context,
                )
            )
            dp_action = self._log_action(
                run_context, action_name, action_args, raw_response
            )

            is_stop_action, action_description = await self._execute_action(
                env, function_call
            )
            all_observations.append(action_description)

            self.logger.debug(f"Observation#{step}: {action_description}")

            post_screenshot_name = f"screenshot_{step}_post.png"
            if is_stop_action:
                if run_context.output_dir:
                    shutil.copyfile(
                        run_context.output_dir / pre_screenshot_name,
                        run_context.output_dir / post_screenshot_name,
                    )
                page_context_after = page_context
                url_after = current_url
            else:
                await self._save_screenshot(
                    env, run_context.output_dir, post_screenshot_name
                )
                url_after, page_context_after = await env.get_page_context()

            self._log_observation(
                run_context,
                action_description,
                post_screenshot_name,
                page_info=page_context_after,
                action_id=dp_action.id,
                url=url_after,
            )
            self._state.current_step = step
            run_context.checkpoint()

            if action_name == "ask_user_question":
                if self.config.auto_user_reply:
                    pending_user_response = (
                        "keep going so far as you don't make up information, "
                        "you have my approval"
                    )
                    run_context.checkpoint()
                    continue
                traj.status = SolverStatus.WAITING_FOR_USER
                run_context.checkpoint()
                return action_description, all_actions, all_observations

            if is_stop_action:
                final_answer = self._get_final_answer(thoughts, action_description)
                traj.outcome = Outcome(answer=final_answer)
                traj.status = SolverStatus.COMPLETE
                break

        run_context.checkpoint()
        if traj.status != SolverStatus.COMPLETE:
            final_answer = self._get_final_answer(thoughts, action_description)
            traj.outcome = Outcome(answer=final_answer)
            traj.status = SolverStatus.COMPLETE

        return final_answer, all_actions, all_observations

    async def close(self, run_context: RunContext) -> None:
        """Cleanup after the agent is done."""
        self._state = None
        self._client = None
        self._captcha_timeouts = 0
        self._captcha_disabled = self.config.captcha_timeout_limit <= 0
        await super().close(run_context)

    def _get_final_answer(self, thoughts: str, action_description: str) -> str:
        return action_description

    def _get_observation_prefix(self) -> str:
        obs = self._pending_observation
        self._pending_observation = ""
        return obs

    def _log_action(
        self,
        run_context: RunContext,
        name: str,
        args: dict,
        raw_response: str,
    ) -> DPAction:
        clean_args = {k: v for k, v in args.items() if k not in ("action", "thoughts")}
        thoughts = args.get("thoughts", "")
        action_summary = f"{name}({json.dumps(clean_args)})"
        nl_desc = f"{thoughts}\n{action_summary}" if thoughts else action_summary
        action = DPAction(
            action_name=name,
            content={"action": name, "arguments": dict(args)},
            action_nl_description=nl_desc,
            llm_conversation=LLMConversation(
                messages=[DPLLMMessage(raw_response=raw_response, reasoning=thoughts)]
            ),
            agent_state=AgentState(agent=self.name),
        )
        run_context.add_action(action)
        return action

    async def _save_screenshot(self, env, output_dir, filename: str) -> None:
        if output_dir:
            screenshot_bytes = await env.get_observation()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / filename).write_bytes(screenshot_bytes)

    async def _get_scaled_screenshot(self, env) -> Image.Image:
        """Get current screenshot and scale it for the model."""
        screenshot_bytes = await env.get_observation()
        screenshot = Image.open(io.BytesIO(screenshot_bytes))
        _, scaled_screenshot = self._get_system_message(screenshot)
        return scaled_screenshot

    def _get_system_message(
        self, screenshot: Image.Image
    ) -> Tuple[List[SystemMessage], Image.Image]:
        system_prompt_info = get_computer_use_system_prompt(
            screenshot,
            self.mlm_processor_im_cfg,
            mode=self.config.computer_use_mode,
            include_input_text_key_args=self.config.include_input_text_key_args,
            fn_call_template=self.config.fn_call_template,
            randomize=False,
            display_size=self.DISPLAY_SIZE,
            identity=self.config.identity,
            critical_points=self.config.critical_points,
        )
        self._state.mlm_width, self._state.mlm_height = system_prompt_info["im_size"]
        scaled_screenshot = screenshot.resize(
            (self._state.mlm_width, self._state.mlm_height)
        )

        system_message = []
        full_text = ""
        for msg in system_prompt_info["conversation"]:
            tmp_content = ""
            for content in msg["content"]:
                tmp_content += content["text"]
            system_message.append(SystemMessage(content=tmp_content))
            full_text += tmp_content

        self._allowed_actions = extract_allowed_actions(full_text)

        return system_message, scaled_screenshot

    def _parse_thoughts_and_action(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """Parse a model response without depending on exact whitespace.

        The training format uses ``<tool_call>`` tags, but OpenAI-compatible
        servers sometimes remove a newline, add a Markdown fence, or return the
        JSON object without the tags.  All of those forms are accepted here.
        """
        try:
            if not isinstance(message, str) or not message.strip():
                raise ValueError("the model returned an empty response")

            opening_tag = re.search(r"<tool_call\b[^>]*>", message, re.IGNORECASE)
            if opening_tag:
                thoughts = message[: opening_tag.start()].strip()
                action_text = message[opening_tag.end() :]
                closing_tag = re.search(r"</tool_call\s*>", action_text, re.IGNORECASE)
                if closing_tag:
                    action_text = action_text[: closing_tag.start()]
            else:
                # Be liberal with servers that return a bare/fenced object.
                object_start = message.find("{")
                if object_start < 0:
                    raise ValueError("no <tool_call> tag or JSON object was found")
                thoughts = message[:object_start].strip()
                action_text = message[object_start:]

            action_text = action_text.strip()
            if action_text.startswith("```"):
                action_text = re.sub(
                    r"^```(?:json|python)?\s*",
                    "",
                    action_text,
                    count=1,
                    flags=re.IGNORECASE,
                )
                action_text = re.sub(r"\s*```\s*$", "", action_text, count=1)

            try:
                action, _ = json.JSONDecoder().raw_decode(action_text.lstrip())
            except json.decoder.JSONDecodeError as json_error:
                try:
                    action = ast.literal_eval(action_text)
                except (SyntaxError, ValueError) as literal_error:
                    raise ValueError(
                        f"tool call is not valid JSON: {json_error.msg}"
                    ) from literal_error

            if not isinstance(action, dict):
                raise ValueError("tool call must be a JSON object")
            if not isinstance(action.get("name"), str) or not action["name"]:
                raise ValueError("tool call is missing a non-empty 'name'")
            if not isinstance(action.get("arguments"), dict):
                raise ValueError("tool call is missing an 'arguments' object")
            if not isinstance(action["arguments"].get("action"), str):
                raise ValueError("tool call arguments are missing a string 'action'")
            return thoughts, action
        except (TypeError, ValueError) as error:
            if self.config.terminate_on_parse_error:
                self.logger.warning(
                    "Could not parse model response (%s); ending trajectory because "
                    "terminate_on_parse_error=true",
                    error,
                )
                raw_message = message.strip() if isinstance(message, str) else ""
                return raw_message, {
                    "name": "computer_use",
                    "arguments": {
                        "action": "terminate",
                        "answer": raw_message,
                    },
                }
            raise ValueError(f"Could not parse model tool call: {error}") from error

    def convert_resized_coords_to_original(
        self, coords: List[float], rsz_w: int, rsz_h: int, og_w: int, og_h: int
    ) -> List[float]:
        scale_x = og_w / rsz_w
        scale_y = og_h / rsz_h
        return [coords[0] * scale_x, coords[1] * scale_y]

    def proc_coords(
        self,
        coords: List[float] | None,
        im_w: int,
        im_h: int,
        og_im_w: int | None = None,
        og_im_h: int | None = None,
    ) -> List[float] | None:
        if not coords:
            return coords
        if og_im_w is None:
            og_im_w = im_w
        if og_im_h is None:
            og_im_h = im_h
        tgt_x, tgt_y = coords
        return self.convert_resized_coords_to_original(
            [tgt_x, tgt_y], im_w, im_h, og_im_w, og_im_h
        )

    def _proc(self, coordinate):
        """Shorthand for proc_coords with standard display/viewport args."""
        return self.proc_coords(
            coordinate,
            self.DISPLAY_SIZE,
            self.DISPLAY_SIZE,
            self.config.viewport_width,
            self.config.viewport_height,
        )

    @retry(
        retry=retry_if_not_exception_type(openai.BadRequestError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=5.0, min=5.0, max=60),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    async def _make_model_call(
        self,
        history: List[LLMMessage],
        extra_create_args: Dict[str, Any] | None = None,
    ) -> str:
        """Make a model call using the client."""
        if self._client is None:
            raise ValueError(
                "No client configured for Fara15Agent. Call initialize() first."
            )
        result = await self._client.create(
            messages=history,
            extra_create_args=extra_create_args or {},
        )
        return extract_message_text(result.content)

    def remove_screenshot_from_message(self, msg: LLMMessage) -> LLMMessage | None:
        """Remove the screenshot from the message content."""
        if isinstance(msg.content, list):
            msg.content = [c for c in msg.content if not isinstance(c, ImageObj)]
            return msg
        elif isinstance(msg.content, ImageObj):
            return None
        return msg

    def maybe_remove_old_screenshots(
        self, history: List[LLMMessage], includes_current: bool = False
    ) -> List[LLMMessage]:
        """Remove old screenshots from the chat history, keeping the most recent
        ``max_n_images``. Original task and user-response messages keep their text."""
        if self.config.max_n_images <= 0:
            return history

        max_n_images = (
            self.config.max_n_images
            if includes_current
            else self.config.max_n_images - 1
        )
        new_history: List[LLMMessage] = []
        n_images = 0

        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            meta = (msg.metadata or {}) if isinstance(msg, UserMessage) else {}
            preserve_text = meta.get("is_original", False) or meta.get(
                "is_user_response", False
            )

            if i == 0 and n_images >= max_n_images:
                msg = self.remove_screenshot_from_message(msg)
                if msg is None:
                    continue

            if isinstance(msg.content, list):
                has_image = any(isinstance(c, ImageObj) for c in msg.content)
                if has_image:
                    if n_images < max_n_images:
                        new_history.append(msg)
                    elif preserve_text:
                        msg = self.remove_screenshot_from_message(msg)
                        if msg is not None:
                            raw = meta.get("user_response")
                            if raw is not None:
                                msg.content = [raw]
                            new_history.append(msg)
                    n_images += 1
                else:
                    new_history.append(msg)
            elif isinstance(msg.content, ImageObj):
                if n_images < max_n_images:
                    new_history.append(msg)
                n_images += 1
            else:
                new_history.append(msg)

        return new_history[::-1]

    def _estimate_prompt_tokens(self, history: List[LLMMessage]) -> int:
        total = 0
        for msg in history:
            items = msg.content if isinstance(msg.content, list) else [msg.content]
            for c in items:
                if isinstance(c, ImageObj):
                    total += self.config.image_token_estimate
                elif isinstance(c, str):
                    total += len(c) // 4 + 1
        return total

    @staticmethod
    def _msg_has_image(msg: LLMMessage) -> bool:
        c = msg.content
        if isinstance(c, list):
            return any(isinstance(x, ImageObj) for x in c)
        return isinstance(c, ImageObj)

    def _fit_images_to_budget(self, history: List[LLMMessage]) -> List[LLMMessage]:
        """Drop oldest screenshots (keeping the most recent) until the estimated
        prompt is within ``image_budget_token_cap``. Returns a new list."""
        cap = self.config.image_budget_token_cap
        if cap <= 0 or self._estimate_prompt_tokens(history) <= cap:
            return history
        out = list(history)
        img_idxs = [i for i, m in enumerate(out) if self._msg_has_image(m)]
        for i in img_idxs[:-1]:
            msg = copy.copy(out[i])
            msg.content = [x for x in msg.content if not isinstance(x, ImageObj)]
            out[i] = msg
            if self._estimate_prompt_tokens(out) <= cap:
                break
        return out

    async def _generate_model_call(
        self,
        env,
        is_first_round: bool,
        first_screenshot: Image.Image | None = None,
        user_response: str = "",
    ) -> Tuple[List[FunctionCall], str]:
        history = self.maybe_remove_old_screenshots(self._state.chat_history)

        screenshot_for_system = first_screenshot
        if not is_first_round:
            scaled_screenshot = await self._get_scaled_screenshot(env)
            screenshot_for_system = scaled_screenshot

            curr_url = (await env.get_page_context()).url
            if curr_url:
                trimmed_url = get_trimmed_url(curr_url, self.MAX_URL_LENGTH)
                url_prefix = f"Current URL: {trimmed_url}\n"
            else:
                url_prefix = ""
            observation_prefix = self._get_observation_prefix()
            if user_response:
                text_prompt = f"{url_prefix}{user_response}"
            elif observation_prefix:
                text_prompt = f"{url_prefix}{observation_prefix}\n{self.USER_MESSAGE}"
            else:
                text_prompt = f"{url_prefix}{self.USER_MESSAGE}"

            metadata = (
                {"is_user_response": True, "user_response": user_response}
                if user_response
                else None
            )
            curr_message = UserMessage(
                content=[ImageObj.from_pil(scaled_screenshot), text_prompt],
                metadata=metadata,
            )
            self._state.chat_history.append(curr_message)
            history.append(curr_message)

        system_message, _ = self._get_system_message(screenshot_for_system)
        history = system_message + history
        history = self._fit_images_to_budget(history)

        call_args: dict[str, Any] = {"temperature": 0}
        if self.config.extra_create_args:
            call_args.update(self.config.extra_create_args)
        max_attempts = max(1, self.config.max_parse_retries + 1)
        for attempt in range(1, max_attempts + 1):
            message = await self._make_model_call(history, extra_create_args=call_args)
            assistant_message = AssistantMessage(content=message or "")
            self._state.chat_history.append(assistant_message)
            try:
                thoughts, action = self._parse_thoughts_and_action(message)
                break
            except ValueError as error:
                if attempt == max_attempts:
                    raise ValueError(
                        f"Model returned an invalid tool call after {max_attempts} "
                        f"attempts: {error}"
                    ) from error
                self.logger.warning(
                    "Invalid model tool call (attempt %d/%d): %s; retrying",
                    attempt,
                    max_attempts,
                    error,
                )
                correction = UserMessage(
                    content=(
                        "Your previous response could not be parsed. Reply with exactly "
                        "one valid JSON tool call inside <tool_call> and </tool_call> "
                        "tags, following the provided tool schema."
                    )
                )
                self._state.chat_history.append(correction)
                history.extend([assistant_message, correction])

        action["arguments"]["thoughts"] = thoughts
        function_call = [FunctionCall(id="dummy", **action)]
        return function_call, message

    async def _execute_action(
        self, env, function_call: list[FunctionCall]
    ) -> tuple[bool, str]:
        """Execute an action. Only actions in the current schema are allowed."""
        name = function_call[0].name
        args = function_call[0].arguments
        action_type = args.get("action", "")

        if action_type not in self._allowed_actions:
            if self.config.terminate_on_parse_error:
                self.logger.warning(
                    "terminate_on_parse_error=true: invalid action %r — "
                    "ending trajectory with thoughts as final answer",
                    action_type,
                )
                return True, args.get("thoughts", "") or args.get("answer", "")
            raise ValueError(
                f"Action '{action_type}' is not allowed in mode "
                f"'{self.config.computer_use_mode}'. Allowed: {sorted(self._allowed_actions)}"
            )

        url = (await env.get_page_context()).url
        self.logger.debug(
            WebSurferEvent(
                source="Fara15Agent",
                url=url,
                action=name,
                arguments=args,
                message=f"{name}( {json.dumps(args)} )",
            )
        )

        if "coordinate" in args:
            args["coordinate"] = self._proc(args["coordinate"])

        is_stop_action, action_description = await self._dispatch_action(
            env, action_type, args
        )

        if action_type in self._TEXT_OBSERVATION_ACTIONS:
            self._pending_observation = format_text_observation(
                action_type,
                action_description,
                self.config.max_observation_chars,
            )

        if not is_stop_action:
            if hasattr(env, "wait_for_load"):
                await env.wait_for_load()

        return is_stop_action, action_description

    async def _dispatch_action(
        self, env, action_type: str, args: dict
    ) -> tuple[bool, str]:
        """Dispatch an action using the environment's canonical methods."""
        if action_type == "left_click":
            tgt_x, tgt_y = args["coordinate"]
            await env.left_click(tgt_x, tgt_y)
            return False, f"I clicked at coordinates ({tgt_x}, {tgt_y})."

        elif action_type == "key":
            keys = args.get("keys", [])
            await env.key(keys)
            return False, f"I pressed the following keys: {keys}"

        elif action_type == "mouse_move":
            tgt_x, tgt_y = args["coordinate"]
            await env.mouse_move(tgt_x, tgt_y)
            return False, f"I moved the cursor to ({tgt_x}, {tgt_y})."

        elif action_type == "type":
            text = str(args.get("text", args.get("text_value", "")))
            await env.type(text)
            return False, f"I typed '{text}'."

        elif action_type == "scroll":
            pixels = int(args.get("pixels", 0))
            pixels = int(pixels * self.config.viewport_height / self.DISPLAY_SIZE)
            await env.scroll(pixels)
            direction = "up" if pixels > 0 else "down"
            return False, f"I scrolled {direction}."

        elif action_type == "double_click":
            tgt_x, tgt_y = args["coordinate"]
            await env.double_click(tgt_x, tgt_y)
            return False, f"I double-clicked at coordinates ({tgt_x}, {tgt_y})."

        elif action_type == "right_click":
            tgt_x, tgt_y = args["coordinate"]
            await env.right_click(tgt_x, tgt_y)
            return False, f"I right-clicked at coordinates ({tgt_x}, {tgt_y})."

        elif action_type == "triple_click":
            tgt_x, tgt_y = args["coordinate"]
            await env.triple_click(tgt_x, tgt_y)
            return False, f"I triple-clicked at coordinates ({tgt_x}, {tgt_y})."

        elif action_type == "left_click_drag":
            tgt_x, tgt_y = args["coordinate"]
            await env.left_click_drag(tgt_x, tgt_y)
            return False, f"I dragged to ({tgt_x}, {tgt_y})."

        elif action_type == "hscroll":
            pixels = int(args.get("pixels", 0))
            pixels = int(pixels * self.config.viewport_width / self.DISPLAY_SIZE)
            await env.hscroll(pixels)
            return False, f"I scrolled horizontally by {pixels} pixels."

        elif action_type == "visit_url":
            url = args.get("url", "")
            if url.startswith(("https://", "http://", "file://", "about:")):
                target = url
            elif " " in url:
                target = f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH"
            else:
                target = "https://" + url
            try:
                await env.goto_url(target)
            except Exception as e:
                msg = f"visit_url to {url} failed: {type(e).__name__}: {e}"
                self._pending_observation = msg
                return False, f"I tried to navigate to {url} but it failed: {e}"
            return False, f"I navigated to {url}."

        elif action_type == "history_back":
            await env.go_back()
            return False, "I clicked the browser back button."

        elif action_type == "web_search":
            query = args.get("query", "")
            await env.goto_url(
                f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH"
            )
            return False, f"I searched for '{query}'."

        elif action_type == "read_page_answer_question":
            question = str(args.get("question", ""))
            markdown = await env.get_page_markdown()
            answer = await extract_from_page(markdown, question, self._client)
            return False, f"I read the page to answer: {question}\nAnswer: {answer}"

        elif action_type == "ask_user_question":
            question = str(args.get("question", ""))
            return True, f"I asked the user: {question}"

        elif action_type == "wait":
            duration = args.get("time", args.get("duration", 3.0))
            await env.wait(duration)
            return False, f"I waited {duration}s."

        elif action_type == "pause_and_memorize_fact":
            fact = str(args.get("fact", ""))
            self._state.facts.append(fact)
            return False, f"I memorized the following fact: {fact}"

        elif action_type == "terminate":
            answer = args.get("answer")
            if answer is None:
                raise ValueError("terminate action requires 'answer' argument")
            return True, answer

        else:
            raise ValueError(f"Unknown action: {action_type}")
