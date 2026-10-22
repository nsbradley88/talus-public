"""Agentic coding tasks on charmbracelet/bubbles v0.21.1: SWE-bench style, graded by tests the agent never sees.

Each task: pkg (the package graded), setup(repo) (applied before the agent runs: injected bugs or a stub), prompt, and
hidden (a test file written into pkg AFTER the agent finishes). protect_tests: the agent must not edit *_test.go.
"""
import re

import os, subprocess
def _default_bubbles_src():
    if os.environ.get("BUBBLES_SRC"):
        return os.environ["BUBBLES_SRC"]
    try:
        mod = subprocess.check_output(["go", "env", "GOMODCACHE"], text=True).strip()
        if mod:
            return os.path.join(mod, "github.com/charmbracelet/bubbles@v0.21.1")
    except Exception:
        pass
    return os.path.expanduser("~/go/pkg/mod/github.com/charmbracelet/bubbles@v0.21.1")
SRC = _default_bubbles_src()


def sub(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, old)
    open(path, "w").write(s.replace(old, new))


def stub_visible_lines(repo):
    p = f"{repo}/viewport/viewport.go"
    s = open(p).read()
    s2 = re.sub(r"(func \(m Model\) visibleLines\(\) \(lines \[\]string\) \{\n).*?\n\}\n",
                r'\1\tpanic("not implemented")\n}\n', s, count=1, flags=re.S)
    assert s2 != s
    open(p, "w").write(s2)


TASKS = {
    "progress-downward": dict(
        pkg="progress", protect_tests=False,
        setup=lambda repo: None,
        prompt="Bug report for the progress package: when the target percentage is lowered with SetPercent (so the "
               "bar animates downward), IsAnimating() reports false while the bar is still visibly moving, so callers "
               "stop rendering frames too early. Please find and fix the bug in progress/progress.go with a minimal "
               "change.",
        hidden='''package progress

import "testing"

func TestHiddenIsAnimatingDownward(t *testing.T) {
	m := New()
	m.percentShown, m.targetPercent, m.velocity = 0.8, 0.2, -0.5
	if !m.IsAnimating() {
		t.Fatal("moving downward toward the target must count as animating")
	}
	// passing through the target while still moving fast downward: the spring will overshoot and come back
	m.percentShown, m.targetPercent, m.velocity = 0.2003, 0.2, -0.5
	if !m.IsAnimating() {
		t.Fatal("moving fast downward through the target must count as animating")
	}
	m.percentShown, m.targetPercent, m.velocity = 0.2, 0.8, 0.5
	if !m.IsAnimating() {
		t.Fatal("moving upward toward the target must count as animating")
	}
	m.percentShown, m.targetPercent, m.velocity = 0.2, 0.2, 0
	if m.IsAnimating() {
		t.Fatal("at rest on the target must not be animating")
	}
	m.percentShown, m.targetPercent, m.velocity = 0.2005, 0.2, -0.00001
	if m.IsAnimating() {
		t.Fatal("settled within tolerance must not be animating")
	}
}
'''),
    "paginator-broken": dict(
        pkg="paginator", protect_tests=True,
        setup=lambda repo: (
            sub(f"{repo}/paginator/paginator.go", "\tstart = m.Page * m.PerPage\n", "\tstart = m.Page*m.PerPage + 1\n"),
            sub(f"{repo}/paginator/paginator.go", "return m.Page == m.TotalPages-1", "return m.Page == m.TotalPages")),
        prompt="`go test ./paginator/` is failing. Find and fix the bug(s) in paginator/paginator.go. Do not modify "
               "the test files.",
        hidden=None),
    "viewport-impl": dict(
        pkg="viewport", protect_tests=True,
        setup=stub_visible_lines,
        prompt="The method `visibleLines` in viewport/viewport.go is not implemented yet (it panics). Implement it "
               "according to its doc comment and the rest of the package (vertical offset, frame sizes from the "
               "style, and horizontal scrolling via xOffset) so that `go test ./viewport/` passes. Do not modify "
               "the test files.",
        hidden=None),
    "viewport-ensurevisible": dict(
        pkg="viewport", protect_tests=False,
        setup=lambda repo: None,
        prompt="Add a method `func (m *Model) EnsureVisible(line int)` to viewport.Model. It scrolls the viewport by "
               "the minimum amount so that the given 0-based content line is visible: if the line is above the view "
               "it becomes the top line, if it is below the view it becomes the bottom line, and if it is already "
               "visible nothing changes. Out-of-range values are clamped to the first/last content line, and the "
               "offset must stay within the valid range (as SetYOffset does). Add a doc comment.",
        hidden='''package viewport

import (
	"fmt"
	"strings"
	"testing"
)

func TestHiddenEnsureVisible(t *testing.T) {
	m := New(10, 5)
	lines := make([]string, 20)
	for i := range lines {
		lines[i] = fmt.Sprintf("line %d", i)
	}
	m.SetContent(strings.Join(lines, "\\n"))
	steps := []struct{ line, want int }{
		{3, 0},    // already visible
		{12, 8},   // below: becomes the bottom line
		{9, 8},    // visible in [8,12]: unchanged
		{2, 2},    // above: becomes the top line
		{100, 15}, // clamped to the last line
		{-5, 0},   // clamped to the first line
	}
	for _, s := range steps {
		m.EnsureVisible(s.line)
		if m.YOffset != s.want {
			t.Fatalf("EnsureVisible(%d): YOffset = %d, want %d", s.line, m.YOffset, s.want)
		}
	}
}
'''),
    "paginator-wrap": dict(
        pkg="paginator", protect_tests=False,
        setup=lambda repo: None,
        prompt="Add wrap-around navigation to the paginator package: a new exported field `Wrap bool` on Model and "
               "an option `WithWrap(wrap bool) Option`. When Wrap is true, NextPage on the last page goes to the "
               "first page and PrevPage on the first page goes to the last page. With Wrap false (the default) the "
               "behaviour must not change. Document the new field and option.",
        hidden='''package paginator

import "testing"

func TestHiddenWrap(t *testing.T) {
	m := New(WithTotalPages(3), WithWrap(true))
	if !m.Wrap {
		t.Fatal("WithWrap(true) must set Wrap")
	}
	m.Page = 2
	m.NextPage()
	if m.Page != 0 {
		t.Fatalf("wrap next from last: page %d, want 0", m.Page)
	}
	m.PrevPage()
	if m.Page != 2 {
		t.Fatalf("wrap prev from first: page %d, want 2", m.Page)
	}
	m.PrevPage()
	if m.Page != 1 {
		t.Fatalf("plain prev: page %d, want 1", m.Page)
	}
	n := New(WithTotalPages(3))
	n.Page = 2
	n.NextPage()
	if n.Page != 2 {
		t.Fatalf("no wrap by default (next): page %d", n.Page)
	}
	n.Page = 0
	n.PrevPage()
	if n.Page != 0 {
		t.Fatalf("no wrap by default (prev): page %d", n.Page)
	}
	w := New(WithTotalPages(1), WithWrap(true))
	w.NextPage()
	w.PrevPage()
	if w.Page != 0 {
		t.Fatalf("single page with wrap: page %d", w.Page)
	}
}
'''),
}
