# regex
## Table of Contents
<!-- TOC -->

- [regex](#regex)
	- [Table of Contents](#table-of-contents)
	- [General](#general)
		- [Examples](#examples)
			- [Building](#building)
	- [Usage](#usage)
	- [Grammar](#grammar)
	- [Acknowledgements](#acknowledgements)

<!-- /TOC -->

## General
A regex engine based on PikeVMs.

### Examples
- [re](./compiler/examples/re/) - A grep-like example program.

```bash
$> echo 'hello\ngoodbye' | target/debug/examples/re '\Bll'
hello
```

#### Building
```bash
cargo build --release --example re
```

## Usage
```rust
// Parsing and compilation of a regular expression into a runnable program is
// accomplished by two functions exposed in the `regex_compiler` crate.
use regex_compiler::{compile, parse};

// Evaluating a given input against a pattern is accomplished via a single
/// exposed function in the `regex_runtime` crate.
use regex_runtime::run;

// Matches are exposed as `SaveGroupSlots`
use regex_runtime::SaveGroupSlot;

// A standard regex pattern to be parsed.
let pattern = "(ll)";

// Using the above `compile` and `parse` methods, the regular expresion is 
// parsed into an evaluatable program.
let program = parse(pattern)
    .map_err(|e| format!("{:?}", e))
    .and_then(compile).expect("failed to parse or compile");

let input = "hello\nworld";

// A given program can be ran, taking the number of save groups as an function
// parameter. In this case one group for the pattern `ll` is expected.
let result = run::<1>(&program, input);

assert_eq!(
	// The returned result is an `Option`, signifying that there is a match,
	// containing an array of save group slots for each expected save group.
	// If there is a match, the corresponding save group will be marked
	// complete with a start and end position representing a non-inclusive
	// range for the match. In this case a start of 2 and an end of 4
	// signifies the match covers index 2 and 3 of the input, or `ll`.
	Some([SaveGroupSlot::Complete { start: 2, end: 4 }]),
	result
)
```

## Grammar
The grammar can be found at [regex.ebnf](./docs/regex.ebnf) and based on excellent grammar from [kean](https://kean.blog). Additionally this includes an [xhtml version](./docs/regex.xhtml) for viewing in a browser. This was directly generated from [regex.ebnf](./docs/regex.ebnf) with [rr](https://githug.com/ncatelli/rr-docker.git).

## Acknowledgements
This was mostly for learning purposes and built heavily by reference from:

- [Russ Cox's writeups on regexp](https://swtch.com/~rsc/regexp/)
- [kean](https://kean.blog) - For the reference grammar and introducing me to rr. 
