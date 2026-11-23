//! Minimal backtracking regex engine (no external dependencies).
//!
//! Supports the subset of syntax used by HuggingFace pre-tokenizer patterns:
//! alternation, concatenation, `(?:...)` groups, lookaheads `(?=...)` /
//! `(?!...)`, quantifiers `*` `+` `?` `{n}` `{n,m}`, character classes with
//! ranges, escapes, and Unicode property classes `\p{L}` `\p{M}` `\p{N}`
//! `\p{P}` `\p{S}` `\p{Z}` `\p{C}` (approximated with `std::char` where the
//! exact property tables are unavailable).
//!
//! Semantics are leftmost-first (Perl/Python-like) via backtracking.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UProp {
    L, // letter
    M, // mark
    N, // number
    P, // punctuation
    S, // symbol
    Z, // separator
    C, // control/other
}

#[derive(Debug, Clone)]
enum ClassMember {
    Char(char),
    Range(char, char),
    Prop(UProp),
}

#[derive(Debug, Clone)]
struct Class {
    negated: bool,
    members: Vec<ClassMember>,
}

impl Class {
    fn matches(&self, c: char) -> bool {
        let m = self.members.iter().any(|mem| match mem {
            ClassMember::Char(x) => c == *x,
            ClassMember::Range(lo, hi) => c >= *lo && c <= *hi,
            ClassMember::Prop(p) => char_prop(c, *p),
        });
        if self.negated {
            !m
        } else {
            m
        }
    }
}

#[derive(Debug, Clone)]
enum Re {
    Char(char),
    Class(Box<Class>),
    Alt(Vec<Re>),
    Seq(Vec<Re>),
    Lookahead(Box<Re>, bool), // (pattern, positive)
    /// inner, min, max (None = unbounded), lazy (prefer fewer repetitions).
    Quant(Box<Re>, usize, Option<usize>, bool),
    Ci(Box<Re>), // case-insensitive group `(?i:...)`
}

pub struct Regex {
    seq: Vec<Re>,
}

impl Regex {
    /// Compile a pattern. Returns an error string on malformed input.
    pub fn compile(pattern: &str) -> Result<Regex, String> {
        let chars: Vec<char> = pattern.chars().collect::<Vec<char>>();
        let mut p = Parser { chars, pos: 0 };
        let re = p.parse_alternation()?;
        if p.pos != p.chars.len() {
            return Err(format!(
                "unexpected '{}' at position {}",
                p.chars[p.pos], p.pos
            ));
        }
        Ok(Regex { seq: vec![re] })
    }

    /// Try to match the pattern at exactly `pos` (in chars). Returns the end
    /// position of the match, if any.
    pub fn match_at(&self, text: &[char], pos: usize) -> Option<usize> {
        match_seq(&self.seq, text, pos, false)
    }

    /// Split `text` into consecutive non-overlapping matches (pre-tokenize).
    /// This implements HuggingFace `Split(behavior="isolated")`: matches are
    /// kept and unmatched gaps are **dropped**. The Qwen3.5 pattern covers all
    /// normal input, so this only affects unusual Unicode.
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect::<Vec<char>>();
        let mut words: Vec<String> = Vec::new();
        let mut pos = 0;
        while pos < chars.len() {
            match self.match_at(&chars, pos) {
                Some(end) if end > pos => {
                    words.push(chars[pos..end].iter().collect());
                    pos = end;
                }
                // Isolated split: skip the unmatched character entirely.
                _ => pos += 1,
            }
        }
        words
    }
}

fn match_seq(items: &[Re], text: &[char], pos: usize, ci: bool) -> Option<usize> {
    all_seq_ends(items, text, pos, ci).into_iter().next()
}

/// Every end position reachable by matching a whole sequence, in
/// leftmost-first priority order. Returning only the *first* end here used to
/// discard backtracking alternatives for grouped subexpressions (e.g.
/// `(?:ab?|c)bb` on `"abb"` failed because the group never tried end=1).
fn all_seq_ends(items: &[Re], text: &[char], pos: usize, ci: bool) -> Vec<usize> {
    if items.is_empty() {
        return vec![pos];
    }
    let mut out = Vec::new();
    for end in possible_ends(&items[0], text, pos, ci) {
        out.extend(all_seq_ends(&items[1..], text, end, ci));
    }
    out
}

/// All end positions reachable by matching `re` at `pos`, in greedy
/// leftmost-first order. `ci` enables case-insensitive literal matching.
fn possible_ends(re: &Re, text: &[char], pos: usize, ci: bool) -> Vec<usize> {
    match re {
        Re::Char(c) => {
            if pos < text.len() && (text[pos] == *c || (ci && ci_eq(text[pos], *c))) {
                vec![pos + 1]
            } else {
                Vec::new()
            }
        }
        Re::Class(cls) => {
            if pos < text.len() && cls.matches(text[pos]) {
                vec![pos + 1]
            } else {
                Vec::new()
            }
        }
        Re::Alt(alts) => {
            let mut out = Vec::new();
            for a in alts {
                out.extend(possible_ends(a, text, pos, ci));
            }
            out
        }
        Re::Seq(items) => all_seq_ends(items, text, pos, ci),
        Re::Lookahead(inner, positive) => {
            let matched = match_seq(&[(**inner).clone()], text, pos, ci).is_some();
            if matched == *positive {
                vec![pos]
            } else {
                Vec::new()
            }
        }
        Re::Ci(inner) => possible_ends(inner, text, pos, true),
        Re::Quant(inner, min, max, lazy) => {
            let min = *min;
            let max = *max;
            let inner_re = &**inner;
            let max_count = max.unwrap_or(text.len() + 1);
            // reachable[k] = end positions after matching `inner` k times.
            let mut reachable: Vec<Vec<usize>> = vec![vec![pos]];
            for k in 0..=max_count {
                let cur = reachable[k].clone();
                if cur.is_empty() {
                    break;
                }
                let mut next = Vec::new();
                for p in cur {
                    next.extend(possible_ends(inner_re, text, p, ci));
                }
                reachable.push(next);
            }
            let hi = max_count.min(reachable.len() - 1);
            let mut out = Vec::new();
            if *lazy {
                // Lazy: try the smallest count first.
                for entry in reachable.iter().take(hi + 1).skip(min) {
                    out.extend(entry.clone());
                }
            } else {
                // Greedy: try the largest count first.
                for k in (min..=hi).rev() {
                    out.extend(reachable[k].clone());
                }
            }
            out
        }
    }
}

/// Case-insensitive single-char equality (Unicode-aware via lowercasing).
fn ci_eq(a: char, b: char) -> bool {
    a == b || a.to_lowercase().next() == b.to_lowercase().next()
}

fn char_prop(c: char, p: UProp) -> bool {
    match p {
        UProp::L => c.is_alphabetic(),
        UProp::N => c.is_numeric(),
        UProp::M => is_mark(c),
        UProp::P | UProp::S => is_punct_or_symbol(c),
        UProp::Z => c.is_whitespace(),
        UProp::C => c.is_control() || is_format(c),
    }
}

fn is_mark(c: char) -> bool {
    let cp = c as u32;
    (0x0300..=0x036F).contains(&cp)
        || (0x0483..=0x0489).contains(&cp)
        || (0x0591..=0x05BD).contains(&cp)
        || (0x0610..=0x061A).contains(&cp)
        || (0x064B..=0x065F).contains(&cp)
        || (0x0670..=0x0670).contains(&cp)
        || (0x06D6..=0x06DC).contains(&cp)
        || (0x0900..=0x0903).contains(&cp)
        || (0x093A..=0x094A).contains(&cp)
        || (0x094E..=0x094F).contains(&cp)
        || (0x0951..=0x0954).contains(&cp)
        || (0x0A01..=0x0A03).contains(&cp)
        || (0x0A3C..=0x0A3C).contains(&cp)
        || (0x0A3E..=0x0A42).contains(&cp)
        || (0x0A47..=0x0A48).contains(&cp)
        || (0x0A4B..=0x0A4D).contains(&cp)
        || (0x0B00..=0x0B03).contains(&cp)
        || (0x0B3C..=0x0B3C).contains(&cp)
        || (0x0B3E..=0x0B44).contains(&cp)
        || (0x0B47..=0x0B48).contains(&cp)
        || (0x0B4B..=0x0B4D).contains(&cp)
        || (0x0C00..=0x0C03).contains(&cp)
        || (0x0C3C..=0x0C3C).contains(&cp)
        || (0x0C3E..=0x0C44).contains(&cp)
        || (0x0C46..=0x0C48).contains(&cp)
        || (0x0C4A..=0x0C4D).contains(&cp)
        || (0x0C55..=0x0C56).contains(&cp)
        || (0x0C62..=0x0C63).contains(&cp)
        || (0x0D00..=0x0D03).contains(&cp)
        || (0x0D3C..=0x0D3C).contains(&cp)
        || (0x0D3E..=0x0D44).contains(&cp)
        || (0x0D46..=0x0D48).contains(&cp)
        || (0x0D4A..=0x0D4D).contains(&cp)
        || (0x0D62..=0x0D63).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE00..=0xFE0F).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
}

fn is_format(c: char) -> bool {
    let cp = c as u32;
    (0x200B..=0x200F).contains(&cp)
        || (0x202A..=0x202E).contains(&cp)
        || (0x2060..=0x2064).contains(&cp)
        || cp == 0xFEFF
        || (0xFFF9..=0xFFFB).contains(&cp)
}

fn is_punct_or_symbol(c: char) -> bool {
    !(c.is_alphabetic() || c.is_numeric() || c.is_whitespace() || c.is_control() || is_format(c))
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, off: usize) -> Option<char> {
        self.chars.get(self.pos + off).copied()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    /// Parse a concatenation until end of pattern (top level), `)` (group),
    /// or `|` (alternation).
    fn parse_concat(&mut self) -> Result<Vec<Re>, String> {
        let mut items: Vec<Re> = Vec::new();
        loop {
            match self.peek() {
                None => break,
                Some(')') => break,
                Some('|') => break,
                _ => {}
            }
            items.push(self.parse_quantified()?);
        }
        Ok(items)
    }

    /// Parse a `|`-separated alternation. Returns a single `Re` (a lone
    /// alternative, or `Re::Alt` when there are several).
    fn parse_alternation(&mut self) -> Result<Re, String> {
        let mut alts: Vec<Re> = Vec::new();
        loop {
            let items = self.parse_concat()?;
            let re = if items.len() == 1 {
                items.into_iter().next().unwrap()
            } else {
                Re::Seq(items)
            };
            alts.push(re);
            if self.peek() == Some('|') {
                self.next();
            } else {
                break;
            }
        }
        if alts.len() == 1 {
            Ok(alts.into_iter().next().unwrap())
        } else {
            Ok(Re::Alt(alts))
        }
    }

    fn parse_quantified(&mut self) -> Result<Re, String> {
        let re = self.parse_atom()?;
        match self.peek() {
            Some('*') => {
                self.next();
                let lazy = self.consume_greediness();
                Ok(Re::Quant(Box::new(re), 0, None, lazy))
            }
            Some('+') => {
                self.next();
                let lazy = self.consume_greediness();
                Ok(Re::Quant(Box::new(re), 1, None, lazy))
            }
            Some('?') => {
                self.next();
                let lazy = self.consume_greediness();
                Ok(Re::Quant(Box::new(re), 0, Some(1), lazy))
            }
            Some('{') => {
                let (min, max) = self.parse_braces()?;
                let lazy = self.consume_greediness();
                Ok(Re::Quant(Box::new(re), min, max, lazy))
            }
            _ => Ok(re),
        }
    }

    /// Consume a trailing `?` quantifier modifier; returns true for lazy.
    fn consume_greediness(&mut self) -> bool {
        if self.peek() == Some('?') {
            self.next();
            true
        } else {
            false
        }
    }

    fn parse_braces(&mut self) -> Result<(usize, Option<usize>), String> {
        self.next(); // consume '{'
        let mut min_str = String::new();
        while let Some(c) = self.peek() {
            if c == ',' || c == '}' {
                break;
            }
            min_str.push(c);
            self.next();
        }
        let min: usize = min_str
            .parse()
            .map_err(|_| "invalid quantifier".to_string())?;
        let mut max = None;
        if self.peek() == Some(',') {
            self.next();
            let mut max_str = String::new();
            while let Some(c) = self.peek() {
                if c == '}' {
                    break;
                }
                max_str.push(c);
                self.next();
            }
            max = Some(
                max_str
                    .parse()
                    .map_err(|_| "invalid quantifier".to_string())?,
            );
        }
        if self.next() != Some('}') {
            return Err("unterminated quantifier".into());
        }
        if let Some(m) = max {
            if m < min {
                return Err("invalid quantifier range".into());
            }
        }
        Ok((min, max))
    }

    fn parse_atom(&mut self) -> Result<Re, String> {
        match self.next().ok_or("unexpected end of pattern")? {
            '(' => {
                if self.peek() == Some('?') {
                    self.next();
                    // Inline flags: `(?i:...)` (case-insensitive group).
                    if self.peek() == Some('i') {
                        self.next();
                        if self.next() != Some(':') {
                            return Err("expected ':' after (?i".into());
                        }
                        let inner = self.parse_alternation()?;
                        self.expect_close()?;
                        return Ok(Re::Ci(Box::new(inner)));
                    }
                    match self.next() {
                        Some('=') => {
                            let inner = self.parse_alternation()?;
                            self.expect_close()?;
                            Ok(Re::Lookahead(Box::new(inner), true))
                        }
                        Some('!') => {
                            let inner = self.parse_alternation()?;
                            self.expect_close()?;
                            Ok(Re::Lookahead(Box::new(inner), false))
                        }
                        Some(':') => {
                            let inner = self.parse_alternation()?;
                            self.expect_close()?;
                            Ok(inner)
                        }
                        _ => Err("unsupported group".into()),
                    }
                } else {
                    let inner = self.parse_alternation()?;
                    self.expect_close()?;
                    Ok(inner)
                }
            }
            '[' => self.parse_class(),
            '\\' => self.parse_escape(),
            c => Ok(Re::Char(c)),
        }
    }

    fn expect_close(&mut self) -> Result<(), String> {
        if self.next() == Some(')') {
            Ok(())
        } else {
            Err("expected ')'".into())
        }
    }

    fn parse_escape(&mut self) -> Result<Re, String> {
        let c = self.next().ok_or("dangling escape")?;
        match c {
            's' => Ok(prop_class(UProp::Z, false)),
            'S' => Ok(prop_class(UProp::Z, true)),
            'd' => Ok(prop_class(UProp::N, false)),
            'D' => Ok(prop_class(UProp::N, true)),
            'w' => Ok(Re::Class(Box::new(Class {
                negated: false,
                members: vec![
                    ClassMember::Prop(UProp::L),
                    ClassMember::Prop(UProp::N),
                    ClassMember::Char('_'),
                ],
            }))),
            'W' => Ok(Re::Class(Box::new(Class {
                negated: true,
                members: vec![
                    ClassMember::Prop(UProp::L),
                    ClassMember::Prop(UProp::N),
                    ClassMember::Char('_'),
                ],
            }))),
            'n' => Ok(Re::Char('\n')),
            'r' => Ok(Re::Char('\r')),
            't' => Ok(Re::Char('\t')),
            'f' => Ok(Re::Char('\u{0c}')),
            'v' => Ok(Re::Char('\u{0b}')),
            'p' | 'P' => {
                let neg = c == 'P';
                let prop = self.parse_prop()?;
                Ok(prop_class(prop, neg))
            }
            other => Ok(Re::Char(other)),
        }
    }

    fn parse_prop(&mut self) -> Result<UProp, String> {
        if self.peek() == Some('{') {
            self.next();
            let mut s = String::new();
            while let Some(c) = self.peek() {
                if c == '}' {
                    break;
                }
                s.push(c);
                self.next();
            }
            if self.next() != Some('}') {
                return Err("unterminated \\p{}".into());
            }
            parse_prop_name(&s)
        } else {
            let c = self.next().ok_or("dangling \\p")?;
            parse_prop_name(&c.to_string())
        }
    }

    fn parse_class(&mut self) -> Result<Re, String> {
        // peek is '['
        let mut negated = false;
        if self.peek() == Some('^') {
            self.next();
            negated = true;
        }
        let mut members: Vec<ClassMember> = Vec::new();
        loop {
            match self.next() {
                None => return Err("unterminated character class".into()),
                Some(']') => break,
                Some(c) => {
                    if c == '\\' {
                        let ec = self.next().ok_or("dangling escape in class")?;
                        match ec {
                            'n' => members.push(ClassMember::Char('\n')),
                            'r' => members.push(ClassMember::Char('\r')),
                            't' => members.push(ClassMember::Char('\t')),
                            's' => members.push(ClassMember::Prop(UProp::Z)),
                            'd' => members.push(ClassMember::Prop(UProp::N)),
                            'w' => {
                                members.push(ClassMember::Prop(UProp::L));
                                members.push(ClassMember::Prop(UProp::N));
                                members.push(ClassMember::Char('_'));
                            }
                            'p' | 'P' => {
                                let prop = self.parse_prop()?;
                                members.push(ClassMember::Prop(prop));
                            }
                            other => members.push(ClassMember::Char(other)),
                        }
                    } else if self.peek() == Some('-') && self.peek_at(1) != Some(']') {
                        // Range: c - next.
                        self.next(); // consume '-'
                        let hi = self.next().ok_or("dangling range in class")?;
                        members.push(ClassMember::Range(c, hi));
                    } else {
                        members.push(ClassMember::Char(c));
                    }
                }
            }
        }
        Ok(Re::Class(Box::new(Class { negated, members })))
    }
}

fn prop_class(prop: UProp, negated: bool) -> Re {
    Re::Class(Box::new(Class {
        negated,
        members: vec![ClassMember::Prop(prop)],
    }))
}

fn parse_prop_name(s: &str) -> Result<UProp, String> {
    match s {
        "L" | "Letter" => Ok(UProp::L),
        "M" | "Mark" => Ok(UProp::M),
        "N" | "Number" => Ok(UProp::N),
        "P" | "Punctuation" => Ok(UProp::P),
        "S" | "Symbol" => Ok(UProp::S),
        "Z" | "Separator" => Ok(UProp::Z),
        "C" | "Other" => Ok(UProp::C),
        _ => Err(format!("unsupported unicode property: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_split() {
        let re = Regex::compile(r"\s+|[^\s]+").unwrap();
        assert_eq!(
            re.pre_tokenize("hello world foo"),
            vec!["hello", " ", "world", " ", "foo"]
        );
    }

    #[test]
    fn test_quantifier() {
        let re = Regex::compile(r"ab{1,3}").unwrap();
        assert_eq!(
            re.match_at(&"abbb".chars().collect::<Vec<char>>(), 0),
            Some(4)
        );
        assert_eq!(
            re.match_at(&"ab".chars().collect::<Vec<char>>(), 0),
            Some(2)
        );
        assert_eq!(re.match_at(&"a".chars().collect::<Vec<char>>(), 0), None);
    }

    #[test]
    fn test_group_backtracking() {
        // The group `(?:ab?|c)` must try `a` (end 1) after `ab` (end 2) fails.
        let re = Regex::compile(r"(?:ab?|c)bb").unwrap();
        let text: Vec<char> = "abb".chars().collect();
        assert_eq!(re.match_at(&text, 0), Some(3));
        let re = Regex::compile(r"(?:a|ab)c").unwrap();
        let text: Vec<char> = "abc".chars().collect();
        assert_eq!(re.match_at(&text, 0), Some(3));
    }

    #[test]
    fn test_lazy_quantifier() {
        let lazy = Regex::compile(r"a+?a").unwrap();
        let greedy = Regex::compile(r"a+a").unwrap();
        let text: Vec<char> = "aaa".chars().collect();
        assert_eq!(lazy.match_at(&text, 0), Some(2), "lazy should match 'aa'");
        assert_eq!(
            greedy.match_at(&text, 0),
            Some(3),
            "greedy should match all"
        );
    }

    #[test]
    fn test_isolated_split_drops_gaps() {
        let re = Regex::compile(r"a+").unwrap();
        // `b` and `c` are unmatched gaps: HF isolated split discards them.
        assert_eq!(re.pre_tokenize("baac"), vec!["aa"]);
        assert_eq!(re.pre_tokenize("a--a"), vec!["a", "a"]);
    }

    #[test]
    fn test_lookahead() {
        let re = Regex::compile(r"\s+(?!\s)").unwrap();
        assert_eq!(
            re.match_at(&"a  b".chars().collect::<Vec<char>>(), 1),
            Some(3)
        );
        assert_eq!(
            re.match_at(&"a b".chars().collect::<Vec<char>>(), 1),
            Some(2)
        );
    }

    #[test]
    fn test_alt_order() {
        let re = Regex::compile(r"abc|ab").unwrap();
        assert_eq!(
            re.match_at(&"ab".chars().collect::<Vec<char>>(), 0),
            Some(2)
        );
        assert_eq!(
            re.match_at(&"abcd".chars().collect::<Vec<char>>(), 0),
            Some(3)
        );
    }

    #[test]
    fn test_group_concat() {
        let re = Regex::compile(r"(ab)+c").unwrap();
        assert_eq!(
            re.match_at(&"ababc".chars().collect::<Vec<char>>(), 0),
            Some(5)
        );
        assert_eq!(
            re.match_at(&"abc".chars().collect::<Vec<char>>(), 0),
            Some(3)
        );
        assert_eq!(re.match_at(&"ac".chars().collect::<Vec<char>>(), 0), None);
    }

    #[test]
    fn test_qwen_style() {
        // The real Qwen3.5 pre-tokenizer pattern (top-level alternation,
        // inline `(?i:...)` group, unicode property classes, lookaheads).
        // Expected outputs verified against the Python `regex` module.
        let re = Regex::compile(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )
        .unwrap();
        let cases: &[(&str, &[&str])] = &[
            (
                "Hello, world! 123 it's don't",
                &[
                    "Hello", ",", " world", "!", " ", "1", "2", "3", " it", "'s", " don", "'t",
                ],
            ),
            ("foo bar\nbaz", &["foo", " bar", "\n", "baz"]),
            ("a  b\tc", &["a", " ", " b", "\tc"]),
            ("x123y", &["x", "1", "2", "3", "y"]),
            ("café naïve", &["café", " naïve"]),
        ];
        for (text, expected) in cases {
            let words = re.pre_tokenize(text);
            let got: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
            assert_eq!(&got, expected, "mismatch for input {:?}", text);
        }
    }
}
