// Install a plskit wheel into Pyodide under Node and run one seeded pls1_fit.
// Usage: node pyodide-smoke.mjs <wheel>, run from a directory where the
// `pyodide` npm package is installed: it is resolved from the working
// directory, not from this file, so the repository needs no node_modules.
// Exits non-zero when the install or the fit fails.
import { createRequire } from "node:module";
import path from "node:path";

const { loadPyodide } = createRequire(path.join(process.cwd(), "/"))("pyodide");
const wheel = path.resolve(process.argv[2]);
const py = await loadPyodide();
py.FS.mkdirTree("/wheel");
py.FS.mount(py.FS.filesystems.NODEFS, { root: path.dirname(wheel) }, "/wheel");
await py.loadPackage(["numpy", "micropip"]);
py.setStdout({ batched: (s) => console.log(s) });
await py.runPythonAsync(`
import micropip
await micropip.install("emfs:/wheel/${path.basename(wheel)}")
import numpy as np, plskit
# 1500 x 300 with k = 4 puts n·d·sweeps above the 1,000,000 threshold at
# which the kernel takes its parallel path, the code that runs through
# atomic-wait.
rng = np.random.default_rng(0)
X = rng.normal(size=(1500, 300))
y = X[:, :5] @ rng.normal(size=5) + rng.normal(size=1500)
m = plskit.pls1_fit(X, y, 4, seed=1)
beta = np.asarray(m.beta)
assert beta.shape == (300,) and np.isfinite(beta).all(), beta
print("plskit", plskit.__version__, "beta[:3]", beta[:3])
`);
