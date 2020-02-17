use std::io::BufReader;

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
        if let (Some(n), Some(o)) = (self.idx, other.idx) {
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
    fn read_parse(file: &mut BufReader) -> Result<Parse, &str> {
        let mut tokens: Vec<Token> = Vec::new();
        let mut root: Option<Token> = None;
        let mut deps: Vec<(u64, Dependency)> = Vec::new();

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
                idx: line[0].unwrap().parse::<u64>(),
            };

            tokens.push(token);
            deps.push((
                line[6].unwrap().parse::<u64>()?,
                Dependency {
                    name: String::from(line[7]),
                },
            ));
        }

        // map dependencies
        for (i, dep) in deps.iter().enumerate() {
            &tokens[i].children.push((&(&tokens[dep.0]), dep.1));
        }

        // finalize the parse
        match root {
            Some(r) => {
                if !tokens.is_empty() {
                    Ok(Parse { tokens, r })
                } else {
                    Err("Parse is empty")
                }
            }
            None => Err("Parse has no root"),
        }
    }
    fn get_parent(&self, child: Token) -> Option<Token> {
        for token in &self.tokens {
            if token.children.contains(child) {
                return Some(token);
            }
        }

        None
    }
}
