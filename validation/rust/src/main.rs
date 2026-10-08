//! Native plskit-rs fit timings, same grid, timing rule and CSV schema as bench.py.
//!
//!   plskit-bench --out <csv> --when <iso>

use std::fs::File;
use std::io::Write;
use std::time::Instant;

use cpu_time::ProcessTime;
use plskit::{pls1_fit, Col, FitOpts, KSpec, Mat};

const FIELDS: &str = "bench,lib,version,routine,n,p,k,wall_ms,cpu_ms,threads,when";
const MIN_BATCH_S: f64 = 0.2;
const REPEATS: usize = 5;

fn fit_grid() -> Vec<(usize, usize, usize)> {
    let mut g = Vec::new();
    for n in [100, 1_000, 10_000] {
        for p in [10, 100, 1_000, 10_000, 100_000] {
            for k in [1, 5, 10] {
                if n * p <= 20_000_000 && k <= n.min(p) {
                    g.push((n, p, k));
                }
            }
        }
    }
    g
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// (wall_ms, cpu_ms) per call: medians over REPEATS batches of >= MIN_BATCH_S (as common.time_call).
fn time_call(mut f: impl FnMut()) -> (f64, f64) {
    f();
    let mut number: usize = 1;
    loop {
        let t0 = Instant::now();
        for _ in 0..number {
            f();
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt >= MIN_BATCH_S || number >= 1_000_000 {
            break;
        }
        number *= if dt == 0.0 { 2 } else { 2.max((MIN_BATCH_S / dt * 1.2) as usize) };
    }
    let (mut walls, mut cpus) = (Vec::new(), Vec::new());
    for _ in 0..REPEATS {
        let (w0, c0) = (Instant::now(), ProcessTime::now());
        for _ in 0..number {
            f();
        }
        walls.push(w0.elapsed().as_secs_f64() / number as f64);
        cpus.push(c0.elapsed().as_secs_f64() / number as f64);
    }
    (median(walls) * 1e3, median(cpus) * 1e3)
}

/// SplitMix64 + Box-Muller: dependency-free normals for the fit grid.
struct Normal(u64);
impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
}

fn fit_data(n: usize, p: usize) -> (Mat<f64>, Col<f64>) {
    let mut rng = Normal(0);
    let x = Mat::from_fn(n, p, |_, _| rng.next());
    let y = Col::from_fn(n, |i| (0..p.min(3)).map(|j| x[(i, j)]).sum::<f64>() + rng.next());
    (x, y)
}

struct Out {
    file: File,
    when: String,
}

impl Out {
    fn create(path: &str, when: &str) -> Self {
        let mut file = File::create(path).unwrap();
        writeln!(file, "{FIELDS}").unwrap();
        Out { file, when: when.into() }
    }
    fn put(&mut self, bench: &str, routine: &str, (n, p, k): (usize, usize, usize), (wall, cpu): (f64, f64)) {
        let version = plskit::version();
        writeln!(self.file, "{bench},plskit-rs,{version},{routine},{n},{p},{k},{wall:.6},{cpu:.6},1,{}", self.when).unwrap();
    }
}

fn run_fit(out: &mut Out) {
    for (n, p, k) in fit_grid() {
        let (x, y) = fit_data(n, p);
        let t = time_call(|| {
            pls1_fit(x.as_ref(), y.as_ref(), KSpec::Fixed(k), None, FitOpts::default()).unwrap();
        });
        out.put("pls1", "pls1_fit", (n, p, k), t);
        eprintln!("{n:>6} {p:>6} {k:>3}  plskit-rs/pls1_fit          wall {:10.3} ms  cpu {:10.3} ms", t.0, t.1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    if std::env::var("BENCH_SINGLE_CORE").as_deref() != Ok("1") {
        eprintln!("source single.env first (or use ./run.sh): runs must be single-core");
        std::process::exit(2);
    }
    let out_path = flag("--out").expect("--out <csv>");
    let mut out = Out::create(&out_path,&flag("--when").unwrap_or_default());
    run_fit(&mut out);
}
