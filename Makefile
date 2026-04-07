.PHONY: build test test-rust test-python oracle examples clean lean cert squeeze mexican-hat 4d-construction quantum-channel condensation semantic-threshold rg-flow return-bound

build:
	cargo build --release

test: test-rust test-python

test-rust:
	cargo test

test-python:
	PYTHONPATH=python python3 -m pytest python/tests/ -v

oracle:
	PYTHONPATH=python python3 oracle/generate.py

check-oracle:
	PYTHONPATH=python python3 oracle/check.py

examples:
	PYTHONPATH=python python3 examples/condensation.py
	PYTHONPATH=python python3 examples/rg_flow.py
	PYTHONPATH=python python3 examples/return_bound.py
	PYTHONPATH=python python3 examples/yang_mills_mass_gap.py
	PYTHONPATH=python python3 examples/yang_mills_4d_construction.py
	PYTHONPATH=python python3 examples/quantum_channel.py
	PYTHONPATH=python python3 examples/mexican_hat_theorem.py
	PYTHONPATH=python python3 examples/semantic_threshold.py
	PYTHONPATH=python python3 examples/prediction.py
	PYTHONPATH=python python3 examples/squeeze_argument.py

quantum-channel:
	PYTHONPATH=python python3 examples/quantum_channel.py

4d-construction:
	PYTHONPATH=python python3 examples/yang_mills_4d_construction.py

mexican-hat:
	PYTHONPATH=python python3 examples/mexican_hat_theorem.py

squeeze:
	PYTHONPATH=python python3 examples/squeeze_argument.py

condensation:
	PYTHONPATH=python python3 examples/condensation.py

semantic-threshold:
	PYTHONPATH=python python3 examples/semantic_threshold.py

rg-flow:
	PYTHONPATH=python python3 examples/rg_flow.py

return-bound:
	PYTHONPATH=python python3 examples/return_bound.py

lean:
	cd lean && lake build

cert: lean
	@echo "Certificate hash (Emet .olean files):"
	@find lean/.lake/build/lib/lean/Emet -name '*.olean' -print0 | sort -z | xargs -0 sha256sum | sha256sum

clean:
	cargo clean || true
