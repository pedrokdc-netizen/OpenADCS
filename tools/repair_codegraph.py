"""Repair the missing-root visualization adapter in CodeGraph 0.21.0."""

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "0.21.0"
ORIGINAL_SHA256 = "c846f80ee0d412d401bbbe1593caaf9df99c397a3f11370dc9ef93d40dbb614a"
ADAPTER_START = "convertToGraphData(e,t){"
ADAPTER_END = "getWebviewContent(){"
ADAPTER = """convertToGraphData(e,t){
    this.currentGraphType=e;
    const nodes=[...t.nodes];
    const root=e==="call"?t.root:null;
    if(root&&typeof root==="object"&&!nodes.some(n=>String(n.id)===String(root.id))){
        nodes.unshift(root);
    }
    return{
        nodes:nodes.map(n=>({
            id:String(n.id),
            label:e==="dependency"?(n.label??n.name??String(n.id)):n.name,
            type:e==="dependency"?n.type:"function",
            language:n.language
        })),
        edges:t.edges.map(edge=>({
            from:String(edge.from),
            to:String(edge.to),
            type:e==="dependency"?edge.type:(edge.isRecursive?"recursive":"call")
        }))
    };
}"""


def repair(extension: Path) -> bool:
    manifest = json.loads((extension / "package.json").read_text(encoding="utf-8"))
    if manifest["version"] != VERSION or manifest["publisher"].lower() != "astudioplus":
        raise ValueError("This repair only supports aStudioPlus CodeGraph 0.21.0")
    bundle = extension / "out" / "extension.js"
    original = bundle.read_bytes()
    text = original.decode("utf-8")
    if ADAPTER + ADAPTER_END in text:
        return False
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Extension bundle differs from the verified original; refusing to overwrite it")
    start = text.index(ADAPTER_START)
    end = text.index(ADAPTER_END, start)
    repaired = (text[:start] + ADAPTER + text[end:]).encode("utf-8")
    backup = bundle.with_name("extension.js.openadcs-original")
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("Existing repair backup does not match this extension")
    else:
        backup.write_bytes(original)
    temporary = bundle.with_name("extension.js.openadcs-tmp")
    with temporary.open("xb") as output:
        output.write(repaired)
    temporary.replace(bundle)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "extension", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".tools" / "vscode" / "data" / "extensions" / f"astudioplus.codegraph-{VERSION}",
        help="CodeGraph extension directory (defaults to this project's portable editor)",
    )
    args = parser.parse_args()
    try:
        changed = repair(args.extension)
    except (OSError, ValueError) as error:
        parser.exit(1, f"CodeGraph repair failed: {error}\n")
    print("CodeGraph visualization repaired." if changed else "CodeGraph repair is already applied.")
    print("Run Developer: Reload Window in VS Code, then reopen the graph.")


if __name__ == "__main__":
    main()
