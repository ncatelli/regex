# regex
## Table of Contents
<!-- TOC -->

- [regex](#regex)
	- [Table of Contents](#table-of-contents)
	- [General](#general)
		- [Examples](#examples)
			- [Building](#building)
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

## Grammar
The grammar can be found at [regex.ebnf](./docs/regex.ebnf) and based on excellent grammar from [kean](https://kean.blog). Additionally this includes an [xhtml version](./docs/regex.xhtml) for viewing in a browser. This was directly generated from [regex.ebnf](./docs/regex.ebnf) with [rr](https://githug.com/ncatelli/rr-docker.git).

## Acknowledgements
This was mostly for learning purposes and built heavily by reference from:

- [Russ Cox's writeups on regexp](https://swtch.com/~rsc/regexp/)
- [kean](https://kean.blog) - For the reference grammar and introducing me to rr. 
