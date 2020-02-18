use std::result::Result;
use std::Vec;

struct Parse {
    head: String,
    children: Vec<(Parse, String)>,
}

fn compose_embedding(
    input: &Parse,
    embeddings: &Embeddings,
) -> Result<(Vec<Variable>, Output), Status> {
}

fn main() {
    // hyperparameters
    let learning_rate: f32 = 0.02;
    let sent_embedding_dim: u64 = 4000;
    let epoch_ct: u16 = 200;
    let sigmoid_cutoff: f32 = 8.0;

    // set up tensorflow environment
    let mut scope: Scope = Scope::new_root_scope();
    let scope: &mut Scope = &mut scope;

    // build graph

    // set up model
    let options: SessionOptions = SessionOptions::new();
    let graph: DerefMut<Graph> = scope.graph_mut();
    let session: Session = Session::new(&options, &graph);
    let mut run_args: SessionRunArgs = SessionRunArgs::new();

    // run training
}
