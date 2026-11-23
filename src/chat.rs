//! Chat message type and Qwen3.5 chat-template rendering.
//!
//! Implements the text-only subset of `chat_template.jinja` (system / user /
//! assistant / tool messages, reasoning-content splitting, multi-step tool
//! detection, and the thinking-mode generation prompt).

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
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
                // Split reasoning_content out of the content if a </think>
                // marker is present.
                let (reasoning, body) = if let Some(pos) = content.find("</think>") {
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
}

#[cfg(test)]
mod prompt_debug {
    use super::*;
    #[test]
    fn print_chat_prompt_tokens() {
        let t = match crate::tokenizer::Tokenizer::load("/tmp/unstable_tok.json") {
            Ok(t) => t,
            Err(_) => return,
        };
        let m = vec![Message {
            role: "user".into(),
            content: "The capital of France is".into(),
        }];
        let prompt = apply_chat_template(&m, true, false);
        eprintln!("PROMPT {:?}", prompt);
        let ids = t.encode(&prompt);
        eprintln!("TOKENS {:?}", ids);
        eprintln!("DECODED {:?}", t.decode(&ids, false));
    }
}
