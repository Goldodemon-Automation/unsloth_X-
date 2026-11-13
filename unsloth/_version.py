# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The single source of truth for this package's version.
# Keep it a plain literal in a module with no imports. pyproject reads it through
# `attr:`, which setuptools resolves by static AST parse, so a build never imports torch;
# and the torch-free MLX path in __init__.py imports it directly, which it cannot do for
# models/_utils.py (torch, transformers, trl, peft, Triton). That is why the MLX branch
# used to borrow unsloth_zoo's version, reporting a different package's number.
# The build segment is not decoration: it is the only thing that separates this fork's backend
# wheel from the upstream package of the same name and release. PEP 440 local versions cannot be
# published to PyPI, so "unsloth==2026.10.6+x8.1" can only ever be satisfied by a file we shipped --
# pip and uv cannot silently substitute upstream's 2026.10.6 for it. Keep the base version equal to
# the upstream release this tree is cut from; bump the +x8.N counter for every fork release that
# changes backend routes, so an installer can tell the two apart at all.
__version__ = "2026.10.6+x8.1"
