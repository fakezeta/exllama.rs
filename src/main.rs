//! exllama-server: minimal, dependency-free OpenAI-compatible inference server
//! for ExLlamaV3 (EXL3) quantized models.
//!
//! Focus: the `Qwen3_5ForConditionalGeneration` architecture (text decoder).

// Several helpers are intentionally kept for future use / API completeness.
#![allow(dead_code)]

mod chat;
mod compute;
mod exl3;
mod http;
mod json;
mod model;
mod regex;
mod rng;
mod server;
mod tensor;
mod tokenizer;

use std::env;
use std::net::{TcpListener, ToSocketAddrs};
use std::process::ExitCode;

use http::{respond, Request};
use server::{json_error, Engine};

struct Cli {
    model: String,
    context_size: usize,
    host: String,
    port: u16,
    tokenizer: Option<String>,
    max_tokens: usize,
    threads: usize,
    name: Option<String>,
    /// Weight storage mode: "f32" (fast, ~4x disk), "f16" (fast-ish, ~2x),
    /// or "packed" (lowest RAM, on-the-fly decode, slow).
    weight_mode: String,
    /// Compute backend: "auto" (best available), "blas", "gpu", "scalar".
    compute: String,
}

fn print_usage() {
    eprintln!(
        "exllama-server \u{2014} minimal OpenAI-compatible server for EXL3 models (Qwen3.5)\n\
         \n\
         USAGE:\n\
           exllama-server --model <model-dir> [options]\n\
         \n\
         OPTIONS:\n\
           -m, --model <PATH>        Path to the EXL3 model directory (required)\n\
           -c, --context-size <N>    Maximum context size in tokens (default 4096)\n\
           -p, --port <PORT>         Listen port (default 8080)\n\
               --host <HOST>         Bind address (default 0.0.0.0)\n\
               --tokenizer <PATH>    Path to tokenizer.json (default: next to model)\n\
               --max-tokens <N>      Default max generation tokens (default 512)\n\
               --threads <N>         Worker threads for matmul (default: all CPUs)\n\
               --name <NAME>         Model name exposed by the API\n\
               --weight-mode <MODE>  f32 (fast, default) | f16 | packed (lowest RAM, slow)\n\
               --compute <BACKEND>   auto (best, default) | blas | gpu | scalar\n\
           -h, --help                Show this help"
    );
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli {
        model: String::new(),
        context_size: 4096,
        host: "0.0.0.0".into(),
        port: 8080,
        tokenizer: None,
        max_tokens: 512,
        threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
        name: None,
        weight_mode: "f32".into(),
        compute: "auto".into(),
    };
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        let mut next = || {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| "missing value for flag".to_string())
        };
        match a.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "-m" | "--model" => cli.model = next()?,
            "-c" | "--context-size" => {
                cli.context_size = next()?.parse().map_err(|_| "invalid --context-size")?
            }
            "-p" | "--port" => cli.port = next()?.parse().map_err(|_| "invalid --port")?,
            "--host" => cli.host = next()?,
            "--tokenizer" => cli.tokenizer = Some(next()?),
            "--max-tokens" => {
                cli.max_tokens = next()?.parse().map_err(|_| "invalid --max-tokens")?
            }
            "--threads" => cli.threads = next()?.parse().map_err(|_| "invalid --threads")?,
            "--name" => cli.name = Some(next()?),
            "--weight-mode" => {
                cli.weight_mode = next()?;
                if !matches!(cli.weight_mode.as_str(), "f32" | "f16" | "packed") {
                    return Err("--weight-mode must be f32, f16, or packed".into());
                }
            }
            "--compute" => {
                cli.compute = next()?;
                if !matches!(cli.compute.as_str(), "auto" | "blas" | "gpu" | "scalar") {
                    return Err("--compute must be auto, blas, gpu, or scalar".into());
                }
            }
            _ => return Err(format!("unknown argument: {}", a)),
        }
        i += 1;
    }
    if cli.model.is_empty() {
        print_usage();
        return Err("--model is required".into());
    }
    if cli.context_size == 0 {
        return Err("--context-size must be > 0".into());
    }
    if cli.threads == 0 {
        return Err("--threads must be > 0".into());
    }
    if cli.max_tokens == 0 {
        return Err("--max-tokens must be > 0".into());
    }
    Ok(cli)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let cli = match parse_cli(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Select compute backend (best available for "auto"). Pass the configured
    // thread count so the auto probe benchmarks the actual inference setup.
    compute::init(&cli.compute, cli.threads);

    // Load model.
    eprintln!("[init] loading model: {}", cli.model);
    let model =
        match model::Model::load(&cli.model, cli.context_size, cli.threads, &cli.weight_mode) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: failed to load model: {}", e);
                return ExitCode::FAILURE;
            }
        };
    eprintln!(
        "[init] architecture={} layers={} hidden={} vocab={} ctx={}",
        model.config().architecture,
        model.config().num_layers,
        model.config().hidden_size,
        model.config().vocab_size,
        cli.context_size
    );

    // Load tokenizer.
    let tok_path = cli.tokenizer.clone().unwrap_or_else(|| {
        // Default: tokenizer.json next to the model/shard.
        let parent = std::path::Path::new(&cli.model)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        parent.join("tokenizer.json").to_string_lossy().into_owned()
    });
    let mut tokenizer = match tokenizer::Tokenizer::load(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: failed to load tokenizer from {}: {}", tok_path, e);
            return ExitCode::FAILURE;
        }
    };
    // config.json's eos_token_id is authoritative when present (the
    // tokenizer.json fallback is chat-specific).
    if let Some(id) = model.config().eos_token_id {
        eprintln!("[init] eos_token_id from config: {}", id);
        tokenizer.set_eos_id(id);
    }
    eprintln!("[init] tokenizer loaded (vocab={})", tokenizer.vocab_size());

    let name = cli.name.unwrap_or_else(|| {
        std::path::Path::new(&cli.model)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into())
    });

    let default_enable_thinking = model.config().default_enable_thinking();
    let engine = Engine::new(
        model,
        tokenizer,
        name.clone(),
        cli.max_tokens,
        cli.threads,
        default_enable_thinking,
    );

    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = match addr
        .to_socket_addrs()
        .ok()
        .and_then(|mut it| it.next())
        .and_then(|a| TcpListener::bind(a).ok())
    {
        Some(l) => l,
        None => {
            eprintln!("error: failed to bind {}", addr);
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "[init] serving OpenAI API on http://{}  (model: {})",
        addr, name
    );

    let engine = std::sync::Arc::new(engine);
    http::serve(listener, move |stream, req| route(stream, req, &engine));

    ExitCode::SUCCESS
}

pub(crate) fn route(
    mut stream: std::net::TcpStream,
    req: Request,
    engine: &std::sync::Arc<Engine>,
) {
    let method = req.method.clone();
    let path = req.path.clone();

    // CORS preflight.
    if method == "OPTIONS" {
        respond(&mut stream, 204, "application/json", "");
        return;
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/models") | ("GET", "/models") => {
            let body = models_json(engine);
            respond(&mut stream, 200, "application/json", &body);
        }
        ("GET", "/health") | ("GET", "/") => {
            respond(&mut stream, 200, "application/json", "{\"status\":\"ok\"}");
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            server::handle_chat(stream, req, engine);
        }
        ("POST", "/v1/completions") | ("POST", "/completions") => {
            server::handle_completions(stream, req, engine);
        }
        _ => {
            let (status, body) = json_error(
                404,
                &format!("unknown route {} {}", method, path),
                "invalid_request_error",
            );
            respond(&mut stream, status, "application/json", &body);
        }
    }
}

fn models_json(engine: &Engine) -> String {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as f64)
        .unwrap_or(0.0);
    json::Json::Obj(vec![
        ("object".into(), json::Json::Str("list".into())),
        (
            "data".into(),
            json::Json::Arr(vec![json::Json::Obj(vec![
                ("id".into(), json::Json::Str(engine.model_name.clone())),
                ("object".into(), json::Json::Str("model".into())),
                ("created".into(), json::Json::Num(created)),
                ("owned_by".into(), json::Json::Str("exllama.rs".into())),
            ])]),
        ),
    ])
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(rest: &[&str]) -> Vec<String> {
        let mut a = vec!["prog".to_string(), "-m".to_string(), "model".to_string()];
        a.extend(rest.iter().map(|s| s.to_string()));
        a
    }

    #[test]
    fn cli_rejects_zero_sizes() {
        assert!(parse_cli(&args(&["-c", "0"])).is_err());
        assert!(parse_cli(&args(&["--threads", "0"])).is_err());
        assert!(parse_cli(&args(&["--max-tokens", "0"])).is_err());
    }

    #[test]
    fn cli_accepts_valid_options() {
        let cli = parse_cli(&args(&["-c", "1024", "--weight-mode", "f16"])).unwrap();
        assert_eq!(cli.context_size, 1024);
        assert_eq!(cli.weight_mode, "f16");
        assert_eq!(cli.compute, "auto");
    }
}
