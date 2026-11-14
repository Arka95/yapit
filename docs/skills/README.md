# Reusable agent skill

[Project README](../../README.md) | [Skill instructions](yapit/SKILL.md)

`yapit\SKILL.md` is the marketplace-neutral customer skill. It covers local
Windows setup and the five existing MCP workflows without introducing new
tools, commands or a hosted service. The runtime, model assets and personal
voice profiles are not bundled.

## MCP setup and workflows

This is the canonical customer guide for YapIt's MCP integration. The
[skill instructions](yapit/SKILL.md) contain the detailed setup, inputs,
examples, execution limits and output-review steps:

- [Local installation](yapit/SKILL.md#set-up-only-if-needed)
- [Client configuration](yapit/SKILL.md#connect-the-local-mcp-server)
- [The five workflows](yapit/SKILL.md#pick-the-requested-workflow)
- [File paths, synchronous execution and output review](yapit/SKILL.md#files-execution-and-handoff)

The local stdio server delegates to existing workflow modules; it is not a
hosted model service. See [MCP development](../mcp/dev.readme.md) for internals
and tests.

## Package

From the repository root, with an existing Python environment:

```powershell
@'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path.cwd()
version = (root / "VERSION").read_text(encoding="utf-8").strip()
output = root.parent / f"yapit-skill-{version}.zip"
with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
    archive.write(root / "docs" / "skills" / "yapit" / "SKILL.md", "yapit/SKILL.md")
    archive.write(root / "LICENSE", "yapit/LICENSE.txt")
print(output)
'@ | py -3.12 -
```

The explicit two-file allowlist excludes local configuration, credentials,
recordings, models, environments, tests and application binaries. The ZIP is
written beside the repository (`C:\projects` for this checkout), with no duplicate
inside the repository. Update the skill's frontmatter version with the application version
when preparing a new release. Keep its instructions consistent with the MCP
guide; this is a skill ZIP, not the application wheel/source distribution.

## Listing and review

Describe the listing as local narration and portrait **edit guides**, not hosted
generation or finished-footage reels. Declare access to the customer's installed
YapIt local MCP server (or a local terminal for setup), Windows, Python, model
assets and microphone/desktop access for enrollment. Cloud writing is optional
and needs its own dependency, credentials and opt-in.

Before sharing, inspect the actual ZIP for its two expected text files,
frontmatter, version and readable license. Exercise adapter behavior with the
existing mocked MCP units. Live installation, microphone use, model inference
and marketplace acceptance are separate checks, not proven by a valid archive.
