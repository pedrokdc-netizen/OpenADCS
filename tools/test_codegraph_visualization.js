const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const bundle = fs.readFileSync(process.argv[2], "utf8");
const start = bundle.indexOf("convertToGraphData(e,t){");
const end = bundle.indexOf("getWebviewContent(){", start);
assert(start >= 0 && end > start, "Graph adapter must exist");
const adapter = vm.runInNewContext("({" + bundle.slice(start, end) + "})");

const response = {
    root: {id: "root", name: "utc_time", language: "python"},
    nodes: [
        {id: 1, name: "main", language: "python"},
        {id: "2", name: "test_utc_normalization", language: "python"},
    ],
    edges: [
        {from: "1", to: "root"},
        {from: 2, to: "root"},
    ],
};
const graph = adapter.convertToGraphData("call", response);
assert.equal(graph.nodes.length, 3);
assert.equal(graph.nodes[0].label, "utc_time");
assert.equal(adapter.currentGraphType, "call");
assert.equal(response.nodes.length, 2, "Do not mutate the engine response");
const ids = new Set(graph.nodes.map(node => node.id));
assert(graph.edges.every(edge => ids.has(edge.from) && ids.has(edge.to)));

const duplicate = adapter.convertToGraphData("call", {
    ...response,
    nodes: [response.root, ...response.nodes],
});
assert.equal(duplicate.nodes.length, 3, "Do not duplicate an existing root");
const isolated = adapter.convertToGraphData("call", {root: response.root, nodes: [], edges: []});
assert.equal(isolated.nodes.length, 1);
assert.equal(isolated.edges.length, 0, "Do not invent calls for isolated functions");
const recursive = adapter.convertToGraphData("call", {
    root: response.root, nodes: [],
    edges: [{from: "root", to: "root", isRecursive: true}],
});
assert.equal(recursive.edges[0].type, "recursive");
const dependency = adapter.convertToGraphData("dependency", {
    nodes: [{id: 1, label: "module", type: "file", language: "python"}],
    edges: [{from: 1, to: 1, type: "import"}],
});
assert.equal(dependency.nodes[0].label, "module");
assert.equal(dependency.edges[0].type, "import");
assert.equal(adapter.currentGraphType, "dependency");

class Element {
    constructor(tag) {
        this.tag = tag;
        this.children = [];
        this.attributes = {};
        this.clientWidth = 1000;
        this.clientHeight = 700;
    }
    appendChild(child) { this.children.push(child); }
    setAttribute(key, value) { this.attributes[key] = String(value); }
    addEventListener() {}
    set innerHTML(value) { this.children = []; this.html = value; }
    querySelectorAll(selector) {
        const results = [];
        for (const child of this.children) {
            if (child.tag === selector || child.attributes.class === selector.slice(1)) {
                results.push(child);
            }
            results.push(...child.querySelectorAll(selector));
        }
        return results;
    }
}

const elements = new Map();
const messages = new Map();
const document = {
    getElementById(id) {
        if (!elements.has(id)) elements.set(id, new Element("div"));
        return elements.get(id);
    },
    createElementNS: (_, tag) => new Element(tag),
    createElement: tag => new Element(tag),
    addEventListener() {},
};
const scriptStart = bundle.indexOf('<script nonce="${e}">', end);
const scriptEnd = bundle.indexOf("</script>", scriptStart);
assert(scriptStart >= 0 && scriptEnd > scriptStart, "Webview script must exist");
const script = bundle.slice(scriptStart + '<script nonce="${e}">'.length, scriptEnd)
    .replaceAll("\\`", "`").replaceAll("\\\\", "\\");
vm.runInNewContext(script, {
    document,
    window: {
        innerWidth: 1000, innerHeight: 700,
        addEventListener: (name, handler) => messages.set(name, handler),
    },
    acquireVsCodeApi: () => ({postMessage() {}}),
    requestAnimationFrame() {},
});
messages.get("message")({data: {command: "renderGraph", graphType: "call", data: graph}});
const root = elements.get("root");
const nodes = root.querySelectorAll(".node");
const lines = root.querySelectorAll("line");
assert.equal(nodes.length, 3, "SVG must include the selected function and both callers");
assert.equal(lines.length, 2, "SVG must draw both real call relationships");
for (const line of lines) {
    assert.equal(line.attributes["marker-end"], "url(#arrow-call)");
    for (const attribute of ["x1", "y1", "x2", "y2"]) {
        assert(Number.isFinite(Number(line.attributes[attribute])));
    }
}
console.log("Verified graph normalization and actual webview SVG: 3 nodes, 2 directed edges.");
