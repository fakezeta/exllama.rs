//! Chat message type and Qwen3.5 chat-template rendering.
//!
//! Implements the text-only subset of `chat_template.jinja` (system / user /
//! assistant / tool messages, reasoning-content splitting, multi-step tool
//! detection, and the thinking-mode generation prompt).

use crate::json::Json;

#[derive(Debug, Clone, Default)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Multi-turn reasoning trace (OpenAI-compatible `reasoning_content`).
    pub reasoning_content: Option<String>,
    /// Assistant tool calls to render into the prompt.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Optional speaker name from the OpenAI API.
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON object text with the function arguments.
    pub arguments: String,
}

/// The system preamble Qwen3.5 emits when `tools` are supplied.
pub fn tools_system_prompt(tools: &Json) -> String {
    let mut out = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    if let Some(arr) = tools.as_arr() {
        for tool in arr {
            out.push('\n');
            out.push_str(&tool.to_string());
        }
    }
    out.push_str(
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
    );
    out
}

/// Length of the longest suffix of `s` that is also a prefix of `prefix`.
/// Used to hold back markers split across token boundaries so they can be
/// reassembled on the next chunk instead of leaking into content.
pub(crate) fn trailing_prefix_len(s: &str, prefix: &str) -> usize {
    let max = s.len().min(prefix.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&n| {
            let start = s.len() - n;
            s.is_char_boundary(start) && prefix.starts_with(&s[start..])
        })
        .unwrap_or(0)
}

/// Split model-generated `<tool_call>` blocks out of `text`. Returns the text
/// with those blocks removed plus one `ToolCall` per block. An unfinished
/// block (generation stopped at `finish_reason=length`) is dropped rather
/// than left in the content as raw XML.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut clean = String::with_capacity(text.len());
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let Some(end_rel) = rest[start..].find(CLOSE) else {
            // Keep only the text before the incomplete block.
            rest = &rest[..start];
            break;
        };
        let block_end = start + end_rel + CLOSE.len();
        clean.push_str(&rest[..start]);
        let block = &rest[start..block_end];
        if let Some(call) = parse_one_tool_call(block, calls.len()) {
            calls.push(call);
        }
        rest = &rest[block_end..];
    }
    // Also withhold a trailing partial `<tool_call` opener.
    let keep = rest.len() - trailing_prefix_len(rest, OPEN);
    clean.push_str(&rest[..keep]);
    (clean, calls)
}

pub(crate) fn parse_one_tool_call(block: &str, index: usize) -> Option<ToolCall> {
    let name_start = block.find("<function=")? + "<function=".len();
    let name_end = block[name_start..].find('>')? + name_start;
    let name = block[name_start..name_end].trim().to_string();
    let mut pairs: Vec<(String, Json)> = Vec::new();
    let mut rest = &block[name_end + 1..];
    while let Some(p_start) = rest.find("<parameter=") {
        let after = p_start + "<parameter=".len();
        let header = &rest[after..];
        // Qwen3.5 emits `<parameter=name>`. Some clients inject object-valued
        // pseudo-parameters (e.g. `__labels__`) that the model writes as
        // `<parameter=name\n{...}` without the closing `>`. Ending the name at
        // a line break too keeps the following parameter from being swallowed
        // into this one's value.
        let name_end = header.find(['>', '\n', '\r']).unwrap_or(header.len());
        let key = header[..name_end].trim();
        // Consume the `>` itself when present; for the newline form the
        // value starts at the line break and `trim()` removes it.
        let value_at = after + name_end + usize::from(header[name_end..].starts_with('>'));
        let value_region = &rest[value_at..];
        let Some(close) = value_region.find("</parameter>") else {
            break;
        };
        let raw = value_region[..close].trim();
        if !key.is_empty() && !key.contains('<') {
            let value = crate::json::parse(raw).unwrap_or_else(|_| Json::Str(raw.to_string()));
            pairs.push((key.to_string(), value));
        }
        rest = &value_region[close + "</parameter>".len()..];
    }
    Some(ToolCall {
        id: format!("call_{}", index),
        name,
        arguments: Json::Obj(pairs).to_string(),
    })
}

/// Render a conversation into the Qwen3.5 chat format.
///
/// `enable_thinking` selects the generation-prompt suffix:
///   - true  -> `<|im_start|>assistant\n<think>\n`  (9B / 4B default)
///   - false -> `<|im_start|>assistant\n<think>\n\n</think>\n\n`
///     (0.8B / 2B default, or `enable_thinking: false`)
pub fn apply_chat_template(
    messages: &[Message],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut out = String::new();
    if messages.is_empty() {
        return out;
    }

    // System message must be first (template raises otherwise). Emit it here
    // and skip it in the main loop.
    let mut start = 0;
    if messages[0].role == "system" {
        let content = messages[0].content.trim();
        out.push_str("<|im_start|>system\n");
        out.push_str(content);
        out.push_str("<|im_end|>\n");
        start = 1;
    }

    // Find the last real user query (not a tool_response), scanning in reverse.
    let mut last_query_index = messages.len().saturating_sub(1);
    let mut multi_step_tool = true;
    for (idx, m) in messages.iter().enumerate().rev() {
        if multi_step_tool && m.role == "user" {
            let c = m.content.trim();
            if !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")) {
                multi_step_tool = false;
                last_query_index = idx;
            }
        }
    }
    // If no user query was found, the template raises. We fall back to the
    // last index so we still produce something usable.
    let _ = multi_step_tool;

    for (idx, m) in messages.iter().enumerate().skip(start) {
        let content = m.content.trim();
        match m.role.as_str() {
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                // The upstream template prefers an explicit `reasoning_content`
                // field; only when it is absent does it split `</think>` out of
                // the message content.
                let (reasoning, body) = if let Some(rc) = m.reasoning_content.as_deref() {
                    (rc.trim().to_string(), content.to_string())
                } else if let Some(pos) = content.find("</think>") {
                    let before = &content[..pos];
                    let after = &content[pos + "</think>".len()..];
                    let r = before
                        .trim_end_matches('\n')
                        .rsplit("<think>")
                        .next()
                        .unwrap_or("")
                        .trim_start_matches('\n')
                        .trim();
                    (r.to_string(), after.trim_start_matches('\n').to_string())
                } else {
                    (String::new(), content.to_string())
                };

                out.push_str("<|im_start|>assistant\n");
                if idx > last_query_index {
                    out.push_str("<think>\n");
                    out.push_str(&reasoning);
                    out.push_str("\n</think>\n\n");
                    out.push_str(&body);
                } else {
                    out.push_str(&body);
                }
                // Qwen3.5 tool calls are XML blocks nested after the content.
                if let Some(calls) = &m.tool_calls {
                    for (i, call) in calls.iter().enumerate() {
                        if i == 0 {
                            if !body.trim().is_empty() {
                                out.push_str("\n\n<tool_call>\n");
                            } else {
                                out.push_str("<tool_call>\n");
                            }
                        } else {
                            out.push_str("\n<tool_call>\n");
                        }
                        out.push_str(&format!("<function={}>\n", call.name));
                        if let Ok(Json::Obj(args)) = crate::json::parse(&call.arguments) {
                            for (key, value) in args {
                                out.push_str(&format!("<parameter={}>\n", key));
                                match value {
                                    Json::Str(s) => out.push_str(&s),
                                    other => out.push_str(&other.to_string()),
                                }
                                out.push_str("\n</parameter>\n");
                            }
                        }
                        out.push_str("</function>\n</tool_call>");
                    }
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                // Consecutive tool messages are wrapped in one user turn.
                let prev_is_tool = idx > 0 && messages[idx - 1].role == "tool";
                let next_is_tool = idx + 1 < messages.len() && messages[idx + 1].role == "tool";
                if !prev_is_tool {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if !next_is_tool {
                    out.push_str("<|im_end|>\n");
                }
            }
            _ => {
                // Unknown roles: emit as plain text, like the template's
                // fallback for string content.
                out.push_str("<|im_start|>");
                out.push_str(&m.role);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if enable_thinking {
            out.push_str("<think>\n");
        } else {
            out.push_str("<think>\n\n</think>\n\n");
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    #[test]
    fn simple_chat() {
        let m = vec![msg("user", "Hello")];
        let s = apply_chat_template(&m, true, true);
        assert_eq!(
            s,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn with_system() {
        let m = vec![msg("system", "Be helpful"), msg("user", "Hi")];
        let s = apply_chat_template(&m, true, false);
        assert_eq!(
            s,
            "<|im_start|>system\nBe helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn assistant_history() {
        let m = vec![msg("user", "Q1"), msg("assistant", "A1"), msg("user", "Q2")];
        let s = apply_chat_template(&m, true, true);
        assert_eq!(
            s,
            "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn assistant_after_last_query_thinking() {
        let m = vec![
            msg("user", "Q"),
            msg("assistant", "<think>\nreasoning\n</think>\nanswer"),
        ];
        let s = apply_chat_template(&m, false, true);
        assert_eq!(
            s,
            "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\nreasoning\n</think>\n\nanswer<|im_end|>\n"
        );
    }

    /// Committed golden prompts (no `/tmp` dependency) lock down the rendered
    /// Qwen3.5 template, including the thinking-mode generation suffix.
    #[test]
    fn prompts_match_committed_golden_files() {
        // `.gitattributes` forces LF for these files, but a checkout with
        // core.autocrlf=true can still yield CRLF on Windows. Normalize both
        // sides so the assertion is about template content, not line endings.
        fn lf(s: &str) -> String {
            s.replace("\r\n", "\n")
        }
        let simple = apply_chat_template(&[msg("user", "Hello")], true, true);
        assert_eq!(
            lf(&simple),
            lf(include_str!("../tests/golden/chat_simple.txt"))
        );
        let with_system = apply_chat_template(
            &[msg("system", "Be helpful"), msg("user", "Hi")],
            true,
            false,
        );
        assert_eq!(
            lf(&with_system),
            lf(include_str!("../tests/golden/chat_with_system.txt"))
        );
    }

    #[test]
    fn reasoning_content_round_trip() {
        let m = vec![
            msg("user", "Q"),
            Message {
                role: "assistant".into(),
                content: "answer".into(),
                reasoning_content: Some("why".into()),
                ..Default::default()
            },
        ];
        let s = apply_chat_template(&m, true, true);
        assert_eq!(
            s,
            "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\nwhy\n</think>\n\nanswer<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn assistant_tool_calls_render_as_xml() {
        let m = vec![
            msg("user", "weather?"),
            Message {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Paris","days":2}"#.into(),
                }]),
                ..Default::default()
            },
        ];
        let s = apply_chat_template(&m, false, false);
        assert!(s.contains(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"
        ), "{}", s);
    }

    #[test]
    fn generated_tool_calls_are_parsed_and_stripped() {
        let text = "Sure\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(clean, "Sure\n\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn malformed_parameter_header_does_not_swallow_next_argument() {
        // Real-client trace: the client injects an object-valued `__labels__`
        // pseudo-parameter into every tool schema, and Qwen3.5 emitted its
        // opener without the closing `>` (`<parameter=__labels__\n{...}`).
        let text = "<tool_call>\n<function=view_chat>\n<parameter=__labels__\n{\"running\": \"Searching for flights.\"\n\"completed\": \"Searched for flights.\"}\n</parameter>\n<parameter=chat_id>\nchat_001\n</parameter>\n</function>\n</tool_call>\n";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(clean, "\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "view_chat");
        let args = crate::json::parse(&calls[0].arguments).unwrap();
        assert_eq!(args.get("chat_id").unwrap().as_str(), Some("chat_001"));
        // The client's `__labels__` payload itself is not valid JSON (the
        // model omits the comma), so it falls back to a string — but it must
        // stay in its own argument instead of swallowing `chat_id`.
        assert!(args.get("__labels__").is_some());
        assert!(!calls[0].arguments.contains("</parameter"));
        assert!(!calls[0].arguments.contains("<parameter="));
    }

    #[test]
    fn incomplete_tool_call_is_not_left_in_content() {
        let text = "Sure\n<tool_call>\n<function=view_chat>\n<parameter=chat_id>\nchat_001\n";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(clean, "Sure\n");
        assert!(calls.is_empty());
    }

    #[test]
    fn tools_preamble_includes_definitions_and_format() {
        let tools =
            crate::json::parse(r#"[{"type":"function","function":{"name":"f","parameters":{}}}]"#)
                .unwrap();
        let s = tools_system_prompt(&tools);
        assert!(
            s.starts_with("# Tools\n\nYou have access to the following functions:\n\n<tools>\n")
        );
        assert!(s.contains("\"name\":\"f\""));
        assert!(s.contains("<function=example_function_name>"));
    }
}

#[cfg(test)]
mod prompt_debug {
    use super::*;
    #[test]
    #[ignore = "requires /tmp/unstable_tok.json (manual prompt dump); see tests/README.md"]
    fn print_chat_prompt_tokens() {
        let t = match crate::tokenizer::Tokenizer::load("/tmp/unstable_tok.json") {
            Ok(t) => t,
            Err(_) => return,
        };
        let m = vec![Message {
            role: "user".into(),
            content: "The capital of France is".into(),
            ..Default::default()
        }];
        let prompt = apply_chat_template(&m, true, false);
        eprintln!("PROMPT {:?}", prompt);
        let ids = t.encode(&prompt);
        eprintln!("TOKENS {:?}", ids);
        eprintln!("DECODED {:?}", t.decode(&ids, false));
    }
}
