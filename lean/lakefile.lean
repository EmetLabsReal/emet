import Lake
open Lake DSL

package «emet» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.29.0"

@[default_target]
lean_lib Emet where
  globs := #[.submodules `Emet]
