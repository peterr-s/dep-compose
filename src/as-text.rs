use std::io;

fn main() {
    loop {
        let mut line: String = String::new();
        if io::stdin()
            .read_line(&mut line)
            .expect("Could not read line")
            == 0
        {
            break;
        }

        if line.trim().is_empty() {
            println!("");
            continue;
        }
        
        if *(line.chars().peekable().peek().expect("Empty input line")) == '#' {
            continue;
        }

        print!(
            "{} ",
            match line.split_whitespace().nth(1) {
                Some(s) => s,
                _ => "",
            }
        );
    }
}
