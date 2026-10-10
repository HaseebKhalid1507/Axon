mod gguf;
mod viz;

use std::env;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("usage: axon <model.gguf>        # interactive TUI");
        eprintln!("       axon info <model.gguf>    # print summary and exit");
        std::process::exit(1);
    }

    let (cmd, path) = if args.len() >= 3 && args[1] == "info" {
        ("info", &args[2])
    } else {
        ("viz", &args[1])
    };

    let model = gguf::load(path)?;

    match cmd {
        "info" => {
            let a = &model.arch;
            println!("axon v{}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("{} | {}L / {}d / {}h (kv {}) | ctx {}",
                a, model.n_layers(), model.embed_dim(), model.n_heads(), model.n_kv_heads(), model.ctx_len());
            println!("{} tensors | {:.2}B params | {}",
                model.tensors.len(), model.total_params() as f64 / 1e9, human_bytes(model.total_bytes()));
        }
        _ => {
            viz::run(&model)?;
        }
    }

    Ok(())
}

fn human_bytes(b: u64) -> String {
    if b >= 1 << 30 { format!("{:.2} GB", b as f64 / (1u64 << 30) as f64) }
    else if b >= 1 << 20 { format!("{:.1} MB", b as f64 / (1u64 << 20) as f64) }
    else { format!("{b} B") }
}
