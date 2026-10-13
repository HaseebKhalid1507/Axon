use crate::gguf::Model;
use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::io::stdout;

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Arch,
    Tensors,
    Weights,
    Meta,
}

const TABS: &[Tab] = &[Tab::Arch, Tab::Tensors, Tab::Weights, Tab::Meta];

struct App {
    tab: usize,
    scroll: usize,
    selected: usize,
}

fn qtype_color(id: u32) -> Color {
    match id {
        0 => Color::White,        // F32
        1 => Color::LightBlue,    // F16
        2 => Color::Green,        // Q4_0
        6 => Color::Cyan,         // Q5_0
        8 => Color::Yellow,       // Q8_0
        12 => Color::Red,         // Q4_K
        13 => Color::Magenta,     // Q5_K
        14 => Color::Blue,        // Q6_K
        15 => Color::LightYellow, // Q8_K
        _ => Color::DarkGray,
    }
}

fn human_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

fn draw_arch(f: &mut Frame, area: Rect, model: &Model, scroll: usize) {
    let mut lines: Vec<Line> = Vec::new();
    let a = &model.arch;
    let nl = model.n_layers();
    let ed = model.embed_dim();
    let hd = model.n_heads();
    let kv = model.n_kv_heads();
    let ctx = model.ctx_len();
    let ffn = model.ffn_dim();
    let head_dim = ed.checked_div(hd).unwrap_or(0);

    lines.push(Line::from(vec![
        Span::styled("MODEL: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            model.meta_str("general.name").unwrap_or(a).to_string(),
            Style::default().fg(Color::Cyan).bold(),
        ),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::styled(
        format!("  Architecture:    {a}"),
        Style::default().fg(Color::White),
    ));
    lines.push(Line::styled(
        format!(
            "  Parameters:      {:.2}B ({} total)",
            model.total_params() as f64 / 1e9,
            model.total_params()
        ),
        Style::default().fg(Color::White),
    ));
    lines.push(Line::styled(
        format!(
            "  Size on disk:    {} (avg {:.1} bits/param)",
            human_bytes(model.total_bytes()),
            model.total_bytes() as f64 * 8.0 / model.total_params() as f64
        ),
        Style::default().fg(Color::White),
    ));
    lines.push(Line::styled(
        format!(
            "  vs F32:          {} → {:.1}× compression",
            human_bytes(model.total_params() * 4),
            model.total_params() as f64 * 4.0 / model.total_bytes() as f64
        ),
        Style::default().fg(Color::Green),
    ));
    lines.push(Line::from(""));

    // the forward pass
    lines.push(Line::styled(
        "─── FORWARD PASS ───",
        Style::default().fg(Color::Cyan).bold(),
    ));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  INPUT ", Style::default().fg(Color::DarkGray)),
        Span::styled("\"hello world\"", Style::default().fg(Color::Yellow)),
    ]));
    lines.push(Line::styled("    │", Style::default().fg(Color::DarkGray)));

    // tokenizer
    lines.push(Line::from(vec![
        Span::styled("  ┌─", Style::default().fg(Color::DarkGray)),
        Span::styled(" TOKENIZER ", Style::default().fg(Color::White).bold()),
        Span::styled(
            "─────────────────────────────────────────────┐",
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "text → token IDs (integers from a 151K vocabulary)",
            Style::default().fg(Color::Gray),
        ),
        Span::styled("  │", Style::default().fg(Color::DarkGray)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "\"hello\" → [15339]  \"world\" → [1917]",
            Style::default().fg(Color::Yellow),
        ),
        Span::styled("                 │", Style::default().fg(Color::DarkGray)),
    ]));
    lines.push(Line::styled(
        "  └────────────────────────────────────────────────────┘",
        Style::default().fg(Color::DarkGray),
    ));
    lines.push(Line::styled("    │", Style::default().fg(Color::DarkGray)));

    // embedding
    lines.push(Line::from(vec![
        Span::styled("  ┌─", Style::default().fg(Color::Cyan)),
        Span::styled(" TOKEN EMBEDDING ", Style::default().fg(Color::Cyan).bold()),
        Span::styled(
            format!("─── [{ed} dims] "),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(
            format!("(151K × {ed} lookup table)"),
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  │ ", Style::default().fg(Color::Cyan)),
        Span::styled(
            format!("each token ID → a vector of {ed} floats"),
            Style::default().fg(Color::Gray),
        ),
    ]));
    lines.push(Line::styled(
        "  └─────────────────────────────────────────────",
        Style::default().fg(Color::Cyan),
    ));
    lines.push(Line::styled("    │", Style::default().fg(Color::DarkGray)));

    // layers
    lines.push(Line::from(vec![
        Span::styled(
            format!("  ×{nl} TRANSFORMER LAYERS"),
            Style::default().fg(Color::White).bold(),
        ),
        Span::styled(
            "  (the same block, repeated)",
            Style::default().fg(Color::DarkGray),
        ),
    ]));

    for i in 0..nl.min(4) {
        lines.push(Line::styled(
            format!("    │  layer {i}"),
            Style::default().fg(Color::DarkGray),
        ));
        // attention
        lines.push(Line::from(vec![
            Span::styled("    ├─", Style::default().fg(Color::Red)),
            Span::styled(" ATTENTION ", Style::default().fg(Color::Red).bold()),
            Span::styled(
                format!("{hd} heads × {head_dim}d"),
                Style::default().fg(Color::LightRed),
            ),
            Span::styled(
                format!("  (KV: {kv} heads — GQA {hd}/{kv}×)",),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("    │ ", Style::default().fg(Color::Red)),
            Span::styled("  Q: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("[{ed}→{ed}]"), Style::default().fg(Color::DarkGray)),
            Span::styled("  K: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("[{ed}→{}]", head_dim * kv),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("  V: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("[{ed}→{}]", head_dim * kv),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("  Out: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("[{ed}→{ed}]"), Style::default().fg(Color::DarkGray)),
        ]));
        if i == 0 {
            lines.push(Line::from(vec![
                Span::styled("    │ ", Style::default().fg(Color::Red)),
                Span::styled(
                    "  \"what should I pay attention to?\"",
                    Style::default().fg(Color::DarkGray).italic(),
                ),
            ]));
        }
        // FFN
        lines.push(Line::from(vec![
            Span::styled("    ├─", Style::default().fg(Color::Green)),
            Span::styled(" FFN ", Style::default().fg(Color::Green).bold()),
            Span::styled(
                format!("{ed} → {ffn} → {ed}"),
                Style::default().fg(Color::LightGreen),
            ),
            Span::styled(
                "  (SwiGLU: gate × up, then down)",
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        if i == 0 {
            lines.push(Line::from(vec![
                Span::styled("    │ ", Style::default().fg(Color::Green)),
                Span::styled(
                    "  \"now think about it\" — most of the compute is here",
                    Style::default().fg(Color::DarkGray).italic(),
                ),
            ]));
        }
        if i == 1 {
            lines.push(Line::styled(
                "    │  ...",
                Style::default().fg(Color::DarkGray),
            ));
            lines.push(Line::styled(
                format!("    │  (layers 2–{} identical)", nl - 1),
                Style::default().fg(Color::DarkGray),
            ));
            break;
        }
    }
    lines.push(Line::styled("    │", Style::default().fg(Color::DarkGray)));

    // output
    lines.push(Line::from(vec![
        Span::styled("  ┌─", Style::default().fg(Color::Magenta)),
        Span::styled(" RMS NORM ", Style::default().fg(Color::Magenta).bold()),
        Span::styled(
            "normalize → stable gradients",
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    lines.push(Line::styled(
        "  └──────────────────",
        Style::default().fg(Color::Magenta),
    ));
    lines.push(Line::styled("    │", Style::default().fg(Color::DarkGray)));
    lines.push(Line::from(vec![
        Span::styled("  ┌─", Style::default().fg(Color::Yellow)),
        Span::styled(" OUTPUT HEAD ", Style::default().fg(Color::Yellow).bold()),
        Span::styled(
            format!("[{ed} → 151,936]"),
            Style::default().fg(Color::LightYellow),
        ),
        Span::styled(
            "  one score per word in vocab",
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    lines.push(Line::styled(
        "  └──────────────────",
        Style::default().fg(Color::Yellow),
    ));
    lines.push(Line::styled("    │", Style::default().fg(Color::DarkGray)));
    lines.push(Line::from(vec![
        Span::styled("  OUTPUT ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "softmax → sample → next token → feed back ↻",
            Style::default().fg(Color::Cyan),
        ),
    ]));
    lines.push(Line::from(""));

    // context
    lines.push(Line::styled(
        "─── CONTEXT ───",
        Style::default().fg(Color::Cyan).bold(),
    ));
    lines.push(Line::styled(
        format!(
            "  Max context:   {ctx} tokens (~{:.0}K words)",
            ctx as f64 / 1.3 / 1000.0
        ),
        Style::default().fg(Color::White),
    ));
    lines.push(Line::styled(
        format!(
            "  KV cache/tok:  {} per layer × {nl} layers = {} per token",
            human_bytes((head_dim * kv * 2 * 2) as u64),
            human_bytes((head_dim * kv * 2 * 2 * nl) as u64)
        ),
        Style::default().fg(Color::White),
    ));
    lines.push(Line::styled(
        format!(
            "  Full KV cache: {} at max context",
            human_bytes((head_dim * kv * 2 * 2 * nl) as u64 * ctx as u64)
        ),
        Style::default().fg(Color::Yellow),
    ));

    let skip = scroll.min(lines.len().saturating_sub(area.height as usize));
    let para =
        Paragraph::new(lines[skip..].to_vec()).block(Block::default().borders(Borders::NONE));
    f.render_widget(para, area);
}

fn draw_tensors(f: &mut Frame, area: Rect, model: &Model, scroll: usize, selected: usize) {
    let mut rows: Vec<Row> = Vec::new();
    for (i, t) in model.tensors.iter().enumerate() {
        let shape: String = t
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("×");
        let style = if i == selected {
            Style::default().bg(Color::DarkGray)
        } else {
            Style::default()
        };
        rows.push(Row::new(vec![
            Cell::from(Span::styled(t.name.clone(), style.fg(Color::White))),
            Cell::from(Span::styled(
                t.qtype.name,
                style.fg(qtype_color(t.qtype.id)),
            )),
            Cell::from(Span::styled(
                format!("{:.1}b", t.qtype.bits),
                style.fg(Color::DarkGray),
            )),
            Cell::from(Span::styled(shape.clone(), style.fg(Color::Gray))),
            Cell::from(Span::styled(
                format!("{:.2}M", t.n_elements as f64 / 1e6),
                style.fg(Color::DarkGray),
            )),
            Cell::from(Span::styled(
                human_bytes(t.n_bytes),
                style.fg(Color::Yellow),
            )),
        ]));
    }

    let header = Row::new(vec!["NAME", "QUANT", "BITS", "SHAPE", "PARAMS", "SIZE"])
        .style(Style::default().fg(Color::Cyan).bold())
        .bottom_margin(1);

    let widths = [
        Constraint::Min(40),
        Constraint::Length(6),
        Constraint::Length(5),
        Constraint::Length(18),
        Constraint::Length(10),
        Constraint::Length(10),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::NONE))
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    let mut state = TableState::default();
    state.select(Some(selected));
    state.scroll_down_by(scroll as u16);

    f.render_stateful_widget(table, area, &mut state);
}

fn draw_weights(f: &mut Frame, area: Rect, model: &Model, selected: usize) {
    let t = &model.tensors[selected.min(model.tensors.len() - 1)];
    let data = model.tensor_data(t);
    let w = area.width as usize;
    let h = (area.height as usize).saturating_sub(4);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(t.name.clone(), Style::default().fg(Color::Cyan).bold()),
        Span::styled(
            format!("  {}  ", t.qtype.name),
            Style::default().fg(qtype_color(t.qtype.id)),
        ),
        Span::styled(
            format!(
                "{}  {}",
                t.dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("×"),
                human_bytes(t.n_bytes)
            ),
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    lines.push(Line::from(""));

    // show raw bytes as colored blocks
    let block_chars = ['░', '▒', '▓', '█'];
    let bytes_per_row = w.min(data.len());
    for row in 0..h.min(data.len() / bytes_per_row.max(1)) {
        let offset = row * bytes_per_row;
        let mut spans: Vec<Span> = Vec::new();
        for col in 0..bytes_per_row {
            if offset + col >= data.len() {
                break;
            }
            let b = data[offset + col];
            let ch = block_chars[(b / 64) as usize];
            let color = match b / 51 {
                // 0-4 range
                0 => Color::Blue,
                1 => Color::Cyan,
                2 => Color::Green,
                3 => Color::Yellow,
                _ => Color::Red,
            };
            spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
        }
        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));
    lines.push(Line::styled(
        "each char = 1 byte │ ░ low  ▒▓ mid  █ high │ color = byte value range",
        Style::default().fg(Color::DarkGray),
    ));

    let para = Paragraph::new(lines).block(Block::default().borders(Borders::NONE));
    f.render_widget(para, area);
}

fn draw_meta(f: &mut Frame, area: Rect, model: &Model, scroll: usize) {
    let mut lines: Vec<Line> = Vec::new();
    for (k, v) in &model.metadata {
        if k.starts_with("tokenizer.ggml.tokens")
            || k.starts_with("tokenizer.ggml.scores")
            || k.starts_with("tokenizer.ggml.token_type")
        {
            continue;
        }
        let vs = format!("{v}");
        let display = if vs.len() > 200 {
            format!("{}...", &vs[..200])
        } else {
            vs
        };
        lines.push(Line::from(vec![
            Span::styled(k.clone(), Style::default().fg(Color::Cyan)),
            Span::styled(": ", Style::default().fg(Color::DarkGray)),
            Span::styled(display.clone(), Style::default().fg(Color::White)),
        ]));
    }
    let skip = scroll.min(lines.len().saturating_sub(area.height as usize));
    let para =
        Paragraph::new(lines[skip..].to_vec()).block(Block::default().borders(Borders::NONE));
    f.render_widget(para, area);
}

pub fn run(model: &Model) -> std::io::Result<()> {
    terminal::enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut app = App {
        tab: 0,
        scroll: 0,
        selected: 0,
    };

    loop {
        term.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(1),
                ])
                .split(f.area());

            // tab bar
            let tab_titles: Vec<&str> = vec![
                "Architecture [1]",
                "Tensors [2]",
                "Weights [3]",
                "Metadata [4]",
            ];
            let tabs = Tabs::new(tab_titles)
                .select(app.tab)
                .style(Style::default().fg(Color::DarkGray))
                .highlight_style(Style::default().fg(Color::Cyan).bold())
                .divider("│")
                .block(
                    Block::default()
                        .title(Span::styled(
                            " ⚡ AXON ",
                            Style::default().fg(Color::Cyan).bold(),
                        ))
                        .borders(Borders::BOTTOM)
                        .border_style(Style::default().fg(Color::DarkGray)),
                );
            f.render_widget(tabs, chunks[0]);

            // content
            let content = chunks[1];
            match TABS[app.tab] {
                Tab::Arch => draw_arch(f, content, model, app.scroll),
                Tab::Tensors => draw_tensors(f, content, model, app.scroll, app.selected),
                Tab::Weights => draw_weights(f, content, model, app.selected),
                Tab::Meta => draw_meta(f, content, model, app.scroll),
            }

            // status bar
            let status = Line::from(vec![
                Span::styled(" q", Style::default().fg(Color::Yellow)),
                Span::styled(" quit  ", Style::default().fg(Color::DarkGray)),
                Span::styled("1-4", Style::default().fg(Color::Yellow)),
                Span::styled(" tabs  ", Style::default().fg(Color::DarkGray)),
                Span::styled("↑↓", Style::default().fg(Color::Yellow)),
                Span::styled(" scroll  ", Style::default().fg(Color::DarkGray)),
                Span::styled("Enter", Style::default().fg(Color::Yellow)),
                Span::styled(" inspect weights", Style::default().fg(Color::DarkGray)),
            ]);
            f.render_widget(Paragraph::new(status), chunks[2]);
        })?;

        if event::poll(std::time::Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Char('1') => {
                    app.tab = 0;
                    app.scroll = 0;
                }
                KeyCode::Char('2') => {
                    app.tab = 1;
                    app.scroll = 0;
                }
                KeyCode::Char('3') => {
                    app.tab = 2;
                }
                KeyCode::Char('4') => {
                    app.tab = 3;
                    app.scroll = 0;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if app.tab == 1 {
                        app.selected = app.selected.saturating_sub(1);
                    } else {
                        app.scroll = app.scroll.saturating_sub(1);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if app.tab == 1 {
                        app.selected = (app.selected + 1).min(model.tensors.len() - 1);
                    } else {
                        app.scroll += 1;
                    }
                }
                KeyCode::PageUp => {
                    if app.tab == 1 {
                        app.selected = app.selected.saturating_sub(20);
                    } else {
                        app.scroll = app.scroll.saturating_sub(20);
                    }
                }
                KeyCode::PageDown => {
                    if app.tab == 1 {
                        app.selected = (app.selected + 20).min(model.tensors.len() - 1);
                    } else {
                        app.scroll += 20;
                    }
                }
                KeyCode::Enter if app.tab == 1 => {
                    app.tab = 2;
                } // jump to weights view
                _ => {}
            }
        }
    }

    terminal::disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
