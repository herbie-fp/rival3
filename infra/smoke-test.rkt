#lang racket

(require rival3
         math/bigfloat)

(define machine
  (rival-compile (list '(+ x 1))
                 '(x)
                 (list flonum-discretization)))

(define out (rival-apply machine (vector (bf 1.0))))

(unless (equal? out #(2.0))
  (error 'smoke-test
         "unexpected output: ~a"
         out))

(displayln out)
