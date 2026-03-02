#lang scribble/manual

@(require (for-label "../main.rkt" racket/base math/bigfloat))

@title{Rival3: Real Computation via Interval Arithmetic}
@author{Pavel Panchekha, Artem Yadrov, Oliver Flatt, Aditya Kumar}
@defmodule[rival3-racket]

Rival3 is Racket FFI bindings to a Rust implementation of Rival,
an advanced interval arithmetic library for
arbitrary-precision computation of complex mathematical expressions.
Its interval arithmetic is valid and attempts to be tight.
Besides the standard intervals, Rival also supports boolean intervals,
error intervals, and movability flags, as described in
@hyperlink["https://arxiv.org/abs/2107.05784"]{"An Interval Arithmetic
for Robust Error Estimation"}.

Rival is a part of the @hyperlink["https://herbie.uwplse.org"]{Herbie project},
and is developed @hyperlink["https://github.com/herbie-fp/rival3"]{on Github}.

Rival can be used programmatically:

@codeblock{
(define expr '(- (sin x) (- x (/ (pow x 3) 6))))
(define machine (rival-compile (list expr) '(x) (list flonum-discretization)))
(rival-apply machine (vector (bf 0.5)))
(rival-apply machine (vector (bf 1e-100)))
}

Rival works by evaluating the expression with high-precision interval
arithmetic, repeating the evaluation with ever-higher precision until
a narrow-enough output interval is found. Runtime
information can be accessed programmatically using
@racket[rival-profile].

Rival is fast, accurate, and sound. We believe it to be a
state-of-the-art implementation, competitive with Sollya/MPFI,
Calcium/Arb, and Mathematica.

@include-section["eval.scrbl"]
@include-section["profile.scrbl"]
