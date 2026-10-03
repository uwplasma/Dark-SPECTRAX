import json, hashlib, os, sys, platform, subprocess, numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SPECTRAX_COMMIT = "6781d8054b08834e477e3b47c3f8e5711129177c"
HERE = os.path.dirname(os.path.abspath(__file__))

def code_hash():
    h = hashlib.sha256()
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py"):
            h.update(open(os.path.join(HERE, f), "rb").read())
    return h.hexdigest()[:16]

def versions():
    import jax, jaxlib, diffrax, scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                jax=jax.__version__, jaxlib=jaxlib.__version__, diffrax=diffrax.__version__,
                backend=jax.default_backend(), machine=platform.machine(), dtype="float64/complex128")

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, complex) or isinstance(o, np.complexfloating): return [float(o.real), float(o.imag)]
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    return o

def save(case, record, arrays):
    d = os.path.join(OUT, case); os.makedirs(d, exist_ok=True)
    record = dict(case_id=case, spectrax_commit=SPECTRAX_COMMIT,
                  spectrax_source="uwplasma/SPECTRAX integration/dark-baseline-a1",
                  reference_code="c-refs/code (independent NumPy semi-Lagrangian + scipy wofz)",
                  reference_code_sha256_16=code_hash(), gkeyll_changeset="ce77f9f8a366+ (local build, unmodified)",
                  command=" ".join(["python"] + sys.argv), versions=versions(), **record)
    json.dump(jsonable(record), open(os.path.join(d, "run.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(d, "data.npz"), **arrays)
    print("saved", d)
