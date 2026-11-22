//! Minimal HTTP/1.1 server over std::net. One OS thread per connection.
//! Supports plain responses and chunked Server-Sent-Events streaming.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

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
    let mut request_line = String::new();
    if reader
        .read_line(&mut request_line)
        .map_err(|e| e.to_string())?
        == 0
    {
        return Err("empty request".into());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or("bad request line")?.to_string();
    let target = parts.next().ok_or("bad request line")?.to_string();
    let (path, query) = match target.find('?') {
        Some(i) => (target[..i].to_string(), target[i + 1..].to_string()),
        None => (target, String::new()),
    };

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim().to_string();
            let v = v.trim().to_string();
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }

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
    started: bool,
}

impl Streamer {
    pub fn new(stream: TcpStream, status: u16, content_type: &str) -> Self {
        let reason = if status == 200 {
            "OK"
        } else {
            "Internal Server Error"
        };
        let head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
            status, reason, content_type
        );
        let mut writer = BufWriter::new(stream);
        let _ = writer.write_all(head.as_bytes());
        let _ = writer.flush();
        Streamer {
            writer,
            started: true,
        }
    }

    /// Write one raw chunk (SSE frame already formatted by caller).
    pub fn write_chunk(&mut self, data: &str) {
        if data.is_empty() {
            return;
        }
        let head = format!("{:X}\r\n", data.len());
        let _ = self.writer.write_all(head.as_bytes());
        let _ = self.writer.write_all(data.as_bytes());
        let _ = self.writer.write_all(b"\r\n");
        let _ = self.writer.flush();
    }

    pub fn end(mut self) {
        let _ = self.writer.write_all(b"0\r\n\r\n");
        let _ = self.writer.flush();
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
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let handler = handler.clone();
                std::thread::spawn(move || {
                    let mut stream = stream;
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(600)));
                    let _ = stream.set_nodelay(true);
                    match read_request(&mut stream) {
                        Ok(req) => handler(stream, req),
                        Err(e) => {
                            eprintln!("[http] read error: {}", e);
                            let _ = stream.shutdown(std::net::Shutdown::Both);
                        }
                    }
                });
            }
            Err(e) => eprintln!("[http] accept error: {}", e),
        }
    }
}
