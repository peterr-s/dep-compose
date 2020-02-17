use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

#[derive(PartialEq, Eq)]
struct Dependency {
    name: String,
}

struct Token<'a> {
    surface: String,
    children: Vec<(&'a Token<'a>, Dependency)>,
    idx: Option<usize>,
}
impl PartialEq for Token<'_> {
    fn eq(&self, other: &Self) -> bool {
        if let (Some(n), Some(o)) = (self.idx, other.idx) {
            n == o
        } else {
            (self.surface == other.surface) && (self.children == other.children)
        }
    }
}
impl Eq for Token<'_> {}

struct Parse<'a> {
    tokens: Vec<Token<'a>>,
    root: Token<'a>,
}
impl Parse<'_> {
    fn read_parse(file: &mut BufReader<File>) -> Result<Parse, &str> {
        let mut tokens: Vec<Token> = Vec::new();
        let mut root: Option<Token> = None;
        let mut deps: Vec<(usize, Dependency)> = Vec::new();

        loop {
            // get line
            let mut line: String = String::new();
            match file.read_line(&mut line) {
                Ok(n) => {
                    if n == 0 {
                        return Err("Reached EOF unexpectedly");
                    }
                }
                Err(_) => return Err("File read error"),
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
                idx: Some(line[0].parse::<usize>().expect("Invalid token index")),
            };

            tokens.push(token);
            deps.push((
                line[6].parse::<usize>().expect("Invalid head index"),
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
                    Ok(Parse { tokens, root: r })
                } else {
                    Err("Parse is empty")
                }
            }
            None => Err("Parse has no root"),
        }
    }
    fn get_parent(&self, child: Token) -> Option<&Token> {
        for token in &self.tokens {
            for s_child in &token.children {
                if *(s_child.0) == child {
                    return Some(&token);
                }
            }
        }

        None
    }
}
