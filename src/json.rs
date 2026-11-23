//! Minimal JSON value type with a recursive-descent parser and serializer.
//! No external dependencies.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Ordered map (preserves insertion order for stable output).
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn obj() -> Self {
        Json::Obj(Vec::new())
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            // JSON booleans are not numbers; callers that want a numeric field
            // must validate the variant explicitly (e.g. via `json_int`).
            Json::Num(n) if n.is_finite() => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64().and_then(|v| usize::try_from(v).ok())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) if n.is_finite() => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&Vec<(String, Json)>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }

    /// True when the value is an object or array (i.e. "structured").
    pub fn is_structured(&self) -> bool {
        matches!(self, Json::Obj(_) | Json::Arr(_))
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

/// Nesting limit for the recursive-descent parser. Without it, an 8 MiB HTTP
/// body of `[[[[...` would overflow the stack and abort the whole process.
const MAX_DEPTH: usize = 128;

pub fn parse(input: &str) -> Result<Json, String> {
    let b = input.as_bytes();
    let mut p = Parser { b, i: 0, depth: 0 };
    let v = p.value()?;
    p.skip_ws();
    if p.i != p.b.len() {
        return Err(format!("trailing data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.i += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn err(&self, msg: &str) -> String {
        format!("{} at byte {}", msg, self.i)
    }

    fn value(&mut self) -> Result<Json, String> {
        if self.depth >= MAX_DEPTH {
            return Err(self.err("maximum nesting depth exceeded"));
        }
        self.depth += 1;
        let result = self.value_impl();
        self.depth -= 1;
        result
    }

    fn value_impl(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("unexpected character")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn literal(&mut self, s: &str) -> Result<(), String> {
        let sb = s.as_bytes();
        if self.i + sb.len() > self.b.len() {
            return Err(self.err("unexpected end of input"));
        }
        if &self.b[self.i..self.i + sb.len()] != sb {
            return Err(self.err("invalid literal"));
        }
        self.i += sb.len();
        Ok(())
    }

    fn object(&mut self) -> Result<Json, String> {
        self.next(); // consume '{'
        let mut items: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.next();
            return Ok(Json::Obj(items));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key"));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.next() != Some(b':') {
                return Err(self.err("expected ':'"));
            }
            let val = self.value()?;
            items.push((key, val));
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
        Ok(Json::Obj(items))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.next(); // consume '['
        let mut items: Vec<Json> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.next();
            return Ok(Json::Arr(items));
        }
        loop {
            let val = self.value()?;
            items.push(val);
            self.skip_ws();
            match self.next() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.next(); // consume opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = self.next().ok_or_else(|| self.err("unterminated string"))?;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.next().ok_or_else(|| self.err("bad escape"))?;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let cp = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&cp) {
                                // Expect a low surrogate.
                                if self.next() == Some(b'\\') && self.next() == Some(b'u') {
                                    let lo = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                                    } else {
                                        return Err(self.err("invalid low surrogate"));
                                    }
                                } else {
                                    return Err(self.err("expected low surrogate"));
                                }
                            } else {
                                cp
                            };
                            let ch =
                                char::from_u32(cp).ok_or_else(|| self.err("invalid unicode"))?;
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                // Raw control characters are forbidden inside JSON strings.
                c if c < 0x20 => return Err(self.err("control character in string")),
                _ => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("invalid utf-8 in string"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut v: u32 = 0;
        for _ in 0..4 {
            let c = self.next().ok_or_else(|| self.err("bad unicode escape"))?;
            let d = (c as char)
                .to_digit(16)
                .ok_or_else(|| self.err("bad hex digit"))?;
            v = (v << 4) | d;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.next();
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' || c == b'+' || c == b'-' {
                self.next();
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| self.err("bad number"))?;
        s.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| self.err("invalid number"))
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Json::Null => write!(f, "null"),
            Json::Bool(b) => write!(f, "{}", b),
            Json::Num(n) => {
                if !n.is_finite() {
                    // NaN / Infinity are not valid JSON tokens.
                    write!(f, "null")
                } else if n.fract() == 0.0 && n.abs() < 9.007199254740992e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Json::Str(s) => write_json_string(f, s),
            Json::Arr(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Json::Obj(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write_json_string(f, k)?;
                    write!(f, ":")?;
                    write!(f, "{}", v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

pub fn write_json_string(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    write!(f, "\"")?;
    for c in s.chars() {
        match c {
            '"' => write!(f, "\\\"")?,
            '\\' => write!(f, "\\\\")?,
            '\n' => write!(f, "\\n")?,
            '\r' => write!(f, "\\r")?,
            '\t' => write!(f, "\\t")?,
            '\u{0008}' => write!(f, "\\b")?,
            '\u{000C}' => write!(f, "\\f")?,
            c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
            c => write!(f, "{}", c)?,
        }
    }
    write!(f, "\"")
}

/// Build a JSON object from key/value string pairs, preserving order.
pub fn obj(pairs: &[(&str, Json)]) -> Json {
    Json::Obj(
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_excessive_nesting() {
        let deep = format!("{}0{}", "[".repeat(200), "]".repeat(200));
        assert!(
            parse(&deep).is_err(),
            "deeply nested input must be rejected"
        );
        let ok = format!("{}0{}", "[".repeat(100), "]".repeat(100));
        assert!(parse(&ok).is_ok());
        // Deeply nested objects are guarded by the same limit.
        let deep_obj = format!("{{\"a\":{}}}", "{".repeat(200) + &"}".repeat(200));
        assert!(parse(&deep_obj).is_err());
    }

    #[test]
    fn parser_rejects_control_chars_in_strings() {
        assert!(parse("\"a\nb\"").is_err());
        assert!(parse("\"a\u{0001}b\"").is_err());
        assert_eq!(parse("\"a\\nb\"").unwrap(), Json::Str("a\nb".into()));
    }

    #[test]
    fn non_finite_numbers_serialize_as_null() {
        assert_eq!(Json::Num(f64::NAN).to_string(), "null");
        assert_eq!(Json::Num(f64::INFINITY).to_string(), "null");
        assert_eq!(Json::Num(f64::NEG_INFINITY).to_string(), "null");
        assert_eq!(Json::Num(1.5).to_string(), "1.5");
    }

    #[test]
    fn booleans_are_not_numbers() {
        assert!(Json::Bool(true).as_i64().is_none());
        assert!(Json::Bool(true).as_f64().is_none());
        assert!(Json::Bool(true).as_usize().is_none());
        assert_eq!(
            parse("{\"n\":-1}").unwrap().get("n").unwrap().as_usize(),
            None
        );
    }
}
