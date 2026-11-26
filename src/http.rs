//! Minimal HTTP/1.1 server over std::net. One OS thread per connection.
//! Supports plain responses and chunked Server-Sent-Events streaming.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Request size limits. Without these, a client can OOM the server with a
/// huge Content-Length or an unterminated header block.
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
/// Generations are serialized; more waiting connections only waste threads.
const MAX_CONNECTIONS: usize = 64;

pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Read a full HTTP request (request line, headers, body).
pub fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream);
    // Cap the *whole* header block while reading it. `read_line` on an
    // unbounded reader would happily buffer a multi-gigabyte line before the
    // header-size check below ever ran (a slow-client OOM vector).
    let mut limited = (&mut reader).take(MAX_HEADER_BYTES as u64 + 1);
    let mut request_line = String::new();
    if limited
        .read_line(&mut request_line)
        .map_err(|e| e.to_string())?
        == 0
    {
        return Err("empty request".into());
    }
    if request_line.len() > MAX_HEADER_BYTES {
        return Err("request line too large".into());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or("bad request line")?.to_string();
    let target = parts.next().ok_or("bad request line")?.to_string();
    let (path, query) = match target.find('?') {
        Some(i) => (target[..i].to_string(), target[i + 1..].to_string()),
        None => (target, String::new()),
    };

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut content_length: Option<usize> = None;
    let mut transfer_encoding: Option<String> = None;
    let mut header_bytes = request_line.len();
    loop {
        let mut line = String::new();
        if limited.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        header_bytes += line.len();
        if header_bytes > MAX_HEADER_BYTES {
            return Err("request headers too large".into());
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim().to_string();
            let v = v.trim().to_string();
            if k.eq_ignore_ascii_case("content-length") {
                if content_length.is_some() {
                    return Err("duplicate Content-Length".into());
                }
                let n: usize = v.parse().map_err(|_| "invalid Content-Length")?;
                if n > MAX_BODY_BYTES {
                    return Err("request body too large".into());
                }
                content_length = Some(n);
            } else if k.eq_ignore_ascii_case("transfer-encoding") {
                if transfer_encoding.is_some() || content_length.is_some() {
                    return Err("ambiguous message framing".into());
                }
                transfer_encoding = Some(v.to_ascii_lowercase());
            }
            headers.push((k, v));
        }
    }
    // Chunked bodies are not supported. Rejecting them (especially together
    // with Content-Length) avoids a request-smuggling ambiguity if a reverse
    // proxy sits in front of this server.
    if let Some(te) = transfer_encoding {
        if content_length.is_some() {
            return Err("both Content-Length and Transfer-Encoding".into());
        }
        return Err(format!("unsupported Transfer-Encoding: {}", te));
    }
    // `limited` is no longer used; NLL releases the header borrow here so the
    // body can be read from the underlying reader below.
    let content_length = content_length.unwrap_or(0);

    let mut body = vec![0u8; content_length];
    let mut got = 0;
    while got < content_length {
        let n = reader.read(&mut body[got..]).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("connection closed while reading body".into());
        }
        got += n;
    }

    Ok(Request {
        method,
        path,
        query,
        headers,
        body,
    })
}

/// Write a complete, non-streaming HTTP response and close the connection.
pub fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &str) {
    let reason = match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n",
        status,
        reason,
        content_type,
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

/// Streaming (chunked) response writer used for SSE.
pub struct Streamer {
    writer: BufWriter<TcpStream>,
    /// Set once a write fails; callers should stop generating immediately.
    failed: bool,
}

impl Streamer {
    pub fn new(stream: TcpStream, status: u16, content_type: &str) -> Self {
        let reason = match status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            _ => "OK",
        };
        let head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
            status, reason, content_type
        );
        let mut writer = BufWriter::new(stream);
        let failed = writer.write_all(head.as_bytes()).is_err() || writer.flush().is_err();
        Streamer { writer, failed }
    }

    pub fn failed(&self) -> bool {
        self.failed
    }

    fn put(&mut self, bytes: &[u8]) -> bool {
        if self.writer.write_all(bytes).is_err() {
            self.failed = true;
            return false;
        }
        true
    }

    /// Write one raw chunk (SSE frame already formatted by caller).
    /// Returns false if the connection is gone.
    pub fn write_chunk(&mut self, data: &str) -> bool {
        if data.is_empty() {
            return !self.failed;
        }
        if self.failed {
            return false;
        }
        let head = format!("{:X}\r\n", data.len());
        if !self.put(head.as_bytes()) {
            return false;
        }
        if !self.put(data.as_bytes()) {
            return false;
        }
        if !self.put(b"\r\n") {
            return false;
        }
        if self.writer.flush().is_err() {
            self.failed = true;
            return false;
        }
        true
    }

    /// Terminate the chunked body. Takes `&mut self` so callers that share a
    /// mutable borrow with generation callbacks can still finish the stream.
    pub fn end(&mut self) {
        if !self.failed {
            self.put(b"0\r\n\r\n");
            let _ = self.writer.flush();
        }
    }
}

/// Accept loop: spawn a thread per connection, invoking `handler`.
///
/// The handler is cloned for each connection, so it may capture owned state
/// (e.g. an `Arc`) by value. The handler takes ownership of the stream and is
/// responsible for writing the response and closing it.
pub fn serve(
    listener: TcpListener,
    handler: impl Fn(TcpStream, Request) + Clone + Send + Sync + 'static,
) {
    let active = Arc::new(AtomicUsize::new(0));
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if active.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
                    active.fetch_sub(1, Ordering::AcqRel);
                    let body = "{\"error\":{\"message\":\"too many connections\",\"type\":\"server_error\"}}";
                    let head = format!(
                        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(head.as_bytes());
                    continue;
                }
                let handler = handler.clone();
                let active = active.clone();
                std::thread::spawn(move || {
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(600)));
                    let _ = stream.set_write_timeout(Some(Duration::from_secs(600)));
                    let _ = stream.set_nodelay(true);
                    match read_request(&mut stream) {
                        Ok(req) => handler(stream, req),
                        Err(e) => {
                            eprintln!("[http] read error: {}", e);
                            let _ = stream.shutdown(std::net::Shutdown::Both);
                        }
                    }
                    active.fetch_sub(1, Ordering::AcqRel);
                });
            }
            Err(e) => eprintln!("[http] accept error: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A newline-free flood must be rejected by the header cap, not buffered
    /// into an unbounded String before the cap is checked.
    #[test]
    fn request_line_is_bounded() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            read_request(&mut stream)
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client.write_all(&vec![b'A'; 2 * MAX_HEADER_BYTES]).unwrap();
        drop(client); // EOF so read_line returns with the truncated line
        let err = match server.join().unwrap() {
            Ok(_) => panic!("overlong request line was accepted"),
            Err(e) => e,
        };
        assert!(
            err.contains("too large"),
            "expected the header cap to reject the line, got {:?}",
            err
        );
    }

    #[test]
    fn transfer_encoding_is_rejected() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            read_request(&mut stream)
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n")
            .unwrap();
        let err = server.join().unwrap().err().unwrap();
        assert!(
            err.contains("Transfer-Encoding"),
            "unexpected error: {}",
            err
        );
    }

    /// A normal request still parses correctly through the bounded reader.
    #[test]
    fn bounded_reader_parses_normal_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            read_request(&mut stream)
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .write_all(b"POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 4\r\n\r\nbody")
            .unwrap();
        let req = server.join().unwrap().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions");
        assert_eq!(req.query, "x=1");
        assert_eq!(req.body, b"body");
        assert_eq!(req.header("host"), Some("localhost"));
    }
}
