#lang info

(define collection "rival3")
(define version "1.0")

(define pkg-desc "Racket bindings to Rival 3")
(define license 'MIT)

(define deps '(("base" #:version "8.0") "math-lib"))
(define build-deps '("scribble-lib" "racket-doc" "math-doc"))
(define scribblings '(("scribblings/rival3.scrbl" (multi-page) (library))))
