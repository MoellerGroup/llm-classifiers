<div align="center">
  <a href="https://ai.pydantic.dev/">
    <h1>
        LLM Classifiers
    </h1>
  </a>
</div>
<div align="center">
<p align="center">
    <em>Framework to build production-grade LLM-based classifiers, super fast</em>
</p>
</div>
<div align="center">
  <a href="https://github.com/MoellerGroup/llm-classifiers/actions/workflows/publish.yml?query=branch%3Amain"><img src="https://img.shields.io/github/actions/workflow/status/moellergroup/llm-classifiers/publish.yml" alt="CI"></a>
  <a href="https://codecov.io/gh/MoellerGroup/llm-classifiers" > <img src="https://codecov.io/gh/MoellerGroup/llm-classifiers/graph/badge.svg?token=M48LMSM1S4"/></a>  
  <a href="https://pypi.org/project/llm-classifiers" target="_blank"> <img src="https://img.shields.io/pypi/v/llm-classifiers" alt="Package version"></a>  
  <a href="https://github.com/MoellerGroup/llm-classifiers/blob/main/LICENSE"><img src="https://img.shields.io/github/license/moellergroup/llm-classifiers" alt="license"></a>
</div>

---

**Source Code
**: <a href="https://github.com/MoellerGroup/llm-classifiers" target="_blank">https://github.com/MoellerGroup/llm-classifiers</a>

---

`LLM-Classifiers` is a framework to build production-grade LLM-based classifiers, superfast. It is built on top of
Pydantic and Pydantic-AI to provide a simple and easy-to-use interface to build and deploy LLM-based classifiers.

## :cop: How to contribute

To contribute to this project, you should comply with the following guidelines.

### :book: Commit Messages

We follow the `Semantic Commit Messages` convention. This means that each commit message should be prefixed with a type,
followed by a scope and a message. The message should be in the imperative mood.

The following types are allowed:

- `[feat]`: A new feature.
- `[fix]`: A bug fix.
- `[docs]`: Documentation only changes.
- `[style]`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).
- `[refactor]`: A code change that neither fixes a bug nor adds a feature.
- `[perf]`: A code change that improves performance.
- `[test]`: Adding missing or correcting existing tests.
- `[chore]`: Changes to the build process or auxiliary tools and libraries such as documentation generation.
- `[revert]`: Reverts a previous commit.
- `[ci]`: Changes to our CI configuration files and scripts.

An example of a commit message is:

```bash
[feat] description of the scope of the feature.

Description of the changes in the imperative mood.

Changes:
 - Change 1
 - Change 2
```