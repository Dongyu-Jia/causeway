"""The 智能体推演 tab's static modules (no LLM, no build needed): routing, module exports, the storyboard's art rules."""
import re

from fastapi.testclient import TestClient

from scenariolab.server import app

client = TestClient(app, raise_server_exceptions=False)
MODULES = ("agents.js", "_agent_common.js", "_agent_data.js", "_agent_tree.js", "_agent_story.js", "_agent_art.js")


def test_agents_tab_sits_between_the_llm_lab_and_the_portfolio():
    app_js = client.get("/js/app.js").text
    assert "{ id: 'agents', name: '智能体推演' }" in app_js
    assert app_js.index("id: 'lab'") < app_js.index("id: 'agents'") < app_js.index("id: 'portfolio'")


def test_agents_modules_are_served_and_revalidated():
    for name in MODULES:
        r = client.get(f"/js/tabs/{name}")
        assert r.status_code == 200, name
        assert r.headers.get("cache-control") == "no-cache", name
    assert "export async function render" in client.get("/js/tabs/agents.js").text
    story = client.get("/js/tabs/_agent_story.js").text
    assert "export async function renderStory" in story
    assert "export async function renderBrowser" in client.get("/js/tabs/_agent_tree.js").text


def test_storyboard_uses_only_inline_svg_and_no_animation_library():
    """Art is drawn in code: no external images, fonts or scripts, no animation libraries; reduced motion honoured."""
    for name in MODULES:
        src = client.get(f"/js/tabs/{name}").text
        urls = set(re.findall(r"https?://[^\s'\"`)]+", src))
        assert urls <= {"http://www.w3.org/2000/svg"}, (name, urls)
        assert not re.search(r"\bimport\s[^;]*from\s+['\"](?!\./|\.\./)", src), name      # only the app's own modules
        assert "<img" not in src and "url(http" not in src, name
    story = client.get("/js/tabs/_agent_story.js").text
    assert "prefers-reduced-motion" in story and "reducedMotion()" in story
    art = client.get("/js/tabs/_agent_art.js").text
    assert "s('animate'" not in art and "<animate" not in art          # CSS and JS animations only (no SMIL)
    for emblem in ("columns", "dome", "podium", "vault", "crane", "derrick", "chip", "voter", "courts", "ship", "cloud"):
        assert f"  {emblem}: [" in art, emblem


def test_every_catalogue_actor_has_a_colour_and_an_emblem():
    """A new actor in the catalogue must get a colour and an emblem in the tab (no logos, flags or likenesses)."""
    cfg = client.get("/api/agents/config")
    assert cfg.status_code == 200
    common = client.get("/js/tabs/_agent_common.js").text
    ids = [a["id"] for a in cfg.json()["catalogue"]["actors"]] + ["nature"]
    for aid in ids:
        assert re.search(rf"\n  {aid}: {{ zh: '[^']+', color: '#[0-9a-f]{{6}}', emblem: '\w+' }}", common), aid
    # actor colours never reuse the K-line up / down hexes or the probability accent
    colours = re.findall(r"color: '(#[0-9a-f]{6})', emblem", common)
    assert colours and not {"#d9363e", "#1a9c5b", "#3b5bdb"} & set(colours)


def test_real_runs_need_an_explicit_confirmation_in_the_page():
    """With the local Claude Code the start button first shows the estimate and a tick box; a snapshot's on-demand
    branches are generated without a dialog (the user's decision), while a saved run's branch asks first (it spends
    usage on a run, like start, resume, extend, 假如 and 补写叙事)."""
    agents = client.get("/js/tabs/agents.js").text
    assert "我确认：这会计入我的 Claude 订阅用量。" in agents and "confirm: !!confirm" in agents
    assert "env.mock ? start(false) : showConfirm(est)" in agents
    tree = client.get("/js/tabs/_agent_tree.js").text
    gen = tree.split("export function generate")[1].split("\n}\n")[0]
    assert "export const needsConfirm = (env) => env.source?.kind === 'run' && !env.mock" in tree
    assert "if (needsConfirm(env) && !confirmed)" in gen and "我确认：这会计入我的 Claude 订阅用量" in gen
    assert "快照分支不需要确认" in tree


def test_views_stop_when_left_and_the_comic_path_is_durable():
    """Review: switching views inside the tab aborts the previous view (its own signal); the comic's path is kept
    as its leaf in the state and the route, and an interior start plays to the best leaf of its subtree."""
    agents = client.get("/js/tabs/agents.js").text
    assert "viewCtl?.abort();" in agents and "venv.signal = ctl.signal;" in agents
    assert "if (AG.view === 'story' && AG.leaf) p.set('leaf', AG.leaf);" in agents
    story = client.get("/js/tabs/_agent_story.js").text
    assert "async descend(" not in story and "this.src.bestLeaf(key" in story
    assert "this.env.AG.leaf = this.path[this.path.length - 1];" in story
    assert "keepPlaying: false" in story.split("btn('◀'")[1].split("\n")[0]       # stepping back pauses playback
    data = client.get("/js/tabs/_agent_data.js").text
    assert "bestLeaf(key, opts) { return get(`${this.base}/best_leaf/${key}`" in data
