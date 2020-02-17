// DEBUG suppress warnings
#![allow(deprecated)]
#![allow(unused)]

use std::collections::HashSet;
use std::io;
use std::io::Read;
use std::fs::File;

extern crate tensorflow;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Operation;
use tensorflow::OutputToken;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;

mod parse;

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

    // verify that each dependency type is used
    for bigram in &bigrams {
        dependencies.remove(&bigram.dependency);
    }
    if dependencies.is_empty() {
        println!("Trainable instances of all dependencies found");
    } else {
        println!("Warning: No trainable instances found for:");
        for dependency in dependencies {
            println!("{}", dependency);
        }
    }

    // TODO train each phrase embedding separately and add all to dictionary
    //println!("Phrase embeddings trained");
    
    // load graph
    let mut graph: Graph = Graph::new();
    let mut proto: Vec<u8> = Vec::new();
    {
        let mut graph_file: File = File::open("graph.pb").expect("Could not open graph file");
        graph_file.read_to_end(&mut proto)
            .expect("Could not read graph file");
    }
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())
        .expect("Could not import graph");
    
    // TODO run graph
    let mut session: Session = Session::new(&SessionOptions::new(), &graph).expect("Could not start session");
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
