use std::io;
use std::collections::HashSet;

fn main() {
    // get all sentences in ConLL format
    let mut parses: Vec<Parse> = Vec::new();
    let mut current_parse: Parse = Parse { tokens: Vec::new() };
    let mut dependencies: HashSet<String> = HashSet::new();
    loop {
        // get line and handle EOF case
        let mut line: String = String::new();
        if io::stdin()
            .read_line(&mut line)
            .expect("Could not read line")
            == 0
        {
            break;
        }

        // empty string denotes the end of a sentence
        if line.trim().is_empty() {
            // don't redo work if last line was also empty
            if !current_parse.tokens.is_empty() {
                // consider the last sentence complete and start the next
                parses.push(current_parse);
                current_parse = Parse { tokens: Vec::new() };
            }

            // do not bother processing this line
            continue;
        }

        // # denotes a comment; ignore lines marked as such
        if *(line.chars().peekable().peek().expect("Empty input line")) == '#' {
            continue;
        }

        // create an indexable representation of the line
        let line: Vec<&str> = line.split_whitespace().collect();

        // populate fields
        let token: Token = Token {
            surface: String::from(line[1]),
            head: line[6].parse().unwrap(),
            dependency: String::from(line[7]),
        };

        // add dependency to set
        dependencies.insert(String::from(token.dependency.as_str()));

        // add word to sentence
        current_parse.tokens.push(token);
    }
    println!("End of input reached");

    // DEBUG echo the parses
    //for sentence in sentences {
    //    for token in sentence.tokens {
    //        println!("\"{}\" is a {} of [{}]", token.surface, token.dependency, token.head);
    //    }
    //    println!("--- EOS ---");
    //}

    // TODO train embeddings
    //println!("Word embeddings trained");

    // find dependencies between adjacent tokens
    let mut bigrams: HashSet<Bigram> = HashSet::new();
    for parse in &parses {
        for token in &parse.tokens {
            // get adjacent head before
            if (token.head > 0)  // ignore tokens deriving from root
                && (token.head < parse.tokens.len())
                && (parse.tokens[token.head] == *token)
            // indices are offset by 1 due to 0 root
            {
                bigrams.insert(Bigram {
                    tail: String::from(token.surface.as_str()),
                    head: String::from(parse.tokens[token.head - 1].surface.as_str()),
                    dependency: String::from(token.dependency.as_str()),
                });
            }

            // get adjacent head after
            else if (token.head > 2) && (parse.tokens[token.head - 2] == *token) {
                // indices are offset by 1
                bigrams.insert(Bigram {
                    tail: String::from(token.surface.as_str()),
                    head: String::from(parse.tokens[token.head - 1].surface.as_str()),
                    dependency: String::from(token.dependency.as_str()),
                });
            }
        }
    }
    println!("Trainable pairs found");

    // TODO verify that each dependency type is used

    // TODO train each phrase embedding separately and add all to dictionary
    //println!("Phrase embeddings trained");
}

#[derive(PartialEq, Eq)]
struct Token {
    surface: String,
    head: usize,
    dependency: String,
}

// this only supports one direction of dependency
// fine because we will be rewriting the sentences and treating the bigram as one token
// this means we can just re-order to always be head-final
#[derive(PartialEq, Eq, Hash)]
struct Bigram {
    tail: String,
    head: String,
    dependency: String,
}

struct Parse {
    tokens: Vec<Token>,
}
