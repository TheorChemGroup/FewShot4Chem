# -*- coding: utf-8 -*-
"""A config that delegates its code must pin the SAME revision as the repo it delegates to.

MoLFormer-c3-1.1B executes ibm/MoLFormer-XL-both-10pct's modeling code: all six of its
`auto_map` entries are cross-repo (`repo--module.Class`), and transformers inherits the
`revision` pin only when the delegated repo IS the one you asked for
(dynamic_module_utils.py:601). So c3 needs `code_revision`, and that SHA names the same
files that `molformer-xl.yaml`'s own `revision` names.

If one config is bumped and not the other, the two models run different versions of the same
repository's code while each config still looks correct; this test keeps them in step.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "models", "configs", "molformer")

# The delegating config -> the config for the repo whose code it actually executes.
DELEGATIONS = {"molformer-c3-1.1b.yaml": "molformer-xl.yaml"}


def _scalar(path, key):
    """Read one `key: value` out of a YAML without a YAML parser (keeps the test free of a YAML dependency)."""
    if not os.path.exists(path):
        pytest.skip("%s absent" % os.path.basename(path))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^\s*%s:\s*(\S+)\s*$" % re.escape(key), line)
            if m:
                return m.group(1)
    return None


def test_delegated_code_revision_matches_the_repo_it_delegates_to():
    for delegating, owner in DELEGATIONS.items():
        code_rev = _scalar(os.path.join(CFG, delegating), "code_revision")
        owner_rev = _scalar(os.path.join(CFG, owner), "revision")
        assert code_rev, (
            "%s has no `code_revision`, but its auto_map delegates to another repo. "
            "`revision` alone does not bind delegated code." % delegating)
        assert owner_rev, "%s has no `revision` to compare against" % owner
        assert code_rev == owner_rev, (
            "%s executes the code in %s, but the two name different revisions:\n"
            "    %s code_revision = %s\n    %s revision      = %s\n"
            "One of them was bumped without the other; they are the same files."
            % (delegating, owner, delegating, code_rev, owner, owner_rev))


def test_every_delegating_config_is_listed_here():
    """The map above is hand-written, so it goes stale silently when a new delegating repo
    is added. A config that pins `code_revision` is announcing that it delegates -- if one
    appears that this test does not know about, say so rather than passing."""
    pinned = set()
    if not os.path.isdir(CFG):
        pytest.skip("config dir absent")
    for root, _dirs, files in os.walk(os.path.join(ROOT, "models", "configs")):
        for fn in files:
            if fn.endswith((".yaml", ".yml")) and _scalar(os.path.join(root, fn), "code_revision"):
                pinned.add(fn)
    unknown = sorted(pinned - set(DELEGATIONS))
    assert not unknown, (
        "these configs pin `code_revision`, so they delegate their code, but DELEGATIONS "
        "does not say to which repo: %s" % ", ".join(unknown))
