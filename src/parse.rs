struct Dependency {
    name: String,
}

struct Token {
    surface: String,
    children: Vec<(Token, Dependency)>,
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
    fn get_parent(&self, child: Token) -> Option<Token> {
        for token in tokens {
            if token.children.contains(child) {
                return Some(token);
            }
        }

        None
    }
}

