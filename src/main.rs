use std::io;

fn main() {
    // get all sentences in ConLL format
    let mut sentences: Vec<String> = Vec::new();
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
            // TODO sentence finalization things
            continue;
        }

        // create an indexable representation of the line
        let line: Vec<&str> = line.split_whitespace().collect();

        // TODO populate fields
    
        // DEBUG
        print!("Echo:");
        for token in line {
            print!(" {}.", token);
        }
        println!("");
    }
    println!("End of input reached");

    // TODO train embeddings
    println!("Word embeddings trained");
    
    // TODO find dependencies between adjacent tokens
    // note: all such will include exactly one leaf, but not all leaves will be used
    // can this be stretched to all leaves? will include longer, less trainable ngrams
    println!("Trainable pairs found");

    // TODO verify that each dependency type is used

    // TODO train each phrase embedding separately and add all to dictionary
    println!("Phrase embeddings trained");
}
