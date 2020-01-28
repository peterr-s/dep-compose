use std::io;

fn main() {
    // get all sentences in ConLL format
    let mut sentences: Vec<Parse> = Vec::new();
    let mut current_sentence: Parse = Parse { tokens: Vec::new() };
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
            if !current_sentence.tokens.is_empty() {
                // consider the last sentence complete and start the next
                sentences.push(current_sentence);
                current_sentence = Parse { tokens: Vec::new() };
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

        // add word to sentence
        current_sentence.tokens.push(token);
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

    // TODO find dependencies between adjacent tokens
    // note: all such will include exactly one leaf, but not all leaves will be used
    // can this be stretched to all leaves? will include longer, less trainable ngrams
    //println!("Trainable pairs found");

    // TODO verify that each dependency type is used

    // TODO train each phrase embedding separately and add all to dictionary
    //println!("Phrase embeddings trained");
}

struct Token {
    surface: String,
    head: usize,
    dependency: String,
}

struct Parse {
    tokens: Vec<Token>,
}
