# Binary Format

## General
This document contains information about he binary format of the bytecode regex vm.

## Header
The header provides a descriptor for, instruction counts, set counts and fast-forward settings and must be the first 256-bits of a binary program.

![header](./header.svg)

### Fast-Forward
Fast-forward is denoted by the bits 16-17 with bits 128-159 either storing an explicit character value or a set id.

## Character Sets
Character sets are n number of charcter set descriptors as noted in the [header](#header). The format is described below.  

![character set](./character_set.svg)

Following the below descriptor are `n` number of one of the following variants corresponding to both the `set variant` and `element count` fields above the character set header.

All sets are aligned to 128-bits and padded with 0s

### Range
Range include an upper and lower bound for a character set, equivalent to a `[a-z]`. Range will always have an element count of `1`.

![character alphabet range](./character_alphabet_range.svg)

### Explicit
Explicit sets contain `n` number of explicit characters.

![character alphabet explicit](./character_alphabet_explicit.svg)

### Ranges
Ranges includes `n` number for [range](#range) sets.

![character alphabet ranges](./character_alphabet_ranges.svg)


## Instructions
Instructions are defined as n number of descriptors as noted in the [header](#header). They follow the general format of

![opcode](./opcode.svg)

### Any
![any](./operations/any.svg)

### Consume
![consume](./operations/consume.svg)

### Consume Set
![consume_set](./operations/consume_set.svg)

### Epsilon
![epsilon](./operations/epsilon.svg)

### Split
![split](./operations/split.svg)

### Jmp
![jmp](./operations/jmp.svg)

### StartSave
![start_save](./operations/start_save.svg)

### EndSave
![end_save](./operations/end_save.svg)

### Match
![match](./operations/match.svg)

### Meta
![meta](./operations/meta.svg)
