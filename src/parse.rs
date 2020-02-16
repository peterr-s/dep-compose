struct Dependency {
    name: String,
}

struct Token {
    surface: String,
    children: Vec<(&Token, Dependency)>,
    idx: Option<u64>,
}
impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        if (let Some(n) = self.idx) && (let Some(o) = other.idx) {
            n == m
        } else {
            (self.surface == other.surface) && (self.children == other.children)
        }
    }
}
impl Eq for Token {}

struct Parse {
    tokens: Vec<Token>,
    root: Token,
}
impl Parse {
    fn read_parse(file: &mut BufReader) -> Result<Parse> {
        let mut tokens: Vec<Token> = Vec::new();
        let mut root: Option<Token> = None;
        let mut deps: Vec<(int, Dependency)> = Vec::new();

        loop {
            // get line
            let mut line: String = String::new();
            match file.read_line(&mut line) {
                Ok(n) => {
                    if n == 0 {
                        return Err("Reached EOF unexpectedly");
                    }
                }
                Err => return Err("File read error"),
            };

            // empty string denotes the end of a sentence
            if line.trim().is_empty() {
                break;
            }

            // # denotes a comment; ignore lines marked as such
            if *(line
                .chars()
                .peekable()
                .peek()
                .expect("Input line both empty and nonempty"))
                == '#'
            {
                continue;
            }

            // create an indexable representation of the line
            let line: Vec<&str> = line.split_whitespace().collect();

            // populate fields
            let token: Token = Token {
                surface: String::from(line[1]),
                children: Vec::new(),
                idx: from_str::<int>(line[0]).unwrap(),
            };

            tokens.push(token);
            tokens.push((
                from_str::<int>(line[6]).unwrap(),
                Dependency {
                    name: String::from(line[7]),
                },
            ));
        }

        // TODO map dependencies
        for token in &tokens {}

        // finalize the parse
        match root {
            Some(r) => {
                if !tokens.is_empty() {
                    return Ok(Parse { tokens, r });
                }
            }
            None => Err("Parse has no root"),
        }
    }
    fn get_parent(&self, child: Token) -> Option<Token> {
        for token in tokens {
            if token.children.contains(child) {
                return Some(token);
            }
        }

        None
    }
}
