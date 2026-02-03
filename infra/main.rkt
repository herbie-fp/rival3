#lang racket

(require "../rival3-racket/main.rkt")

(provide rival-compile
         rival-apply
         rival-analyze-with-hints
         rival-analyze
         rival-profile
         rival-set-profiling!
         rival-profiling-enabled?
         (struct-out exn:rival)
         (struct-out exn:rival:invalid)
         (struct-out exn:rival:unsamplable)
         (struct-out execution)
         (struct-out discretization)
         boolean-discretization
         flonum-discretization
         rival-machine?
         rival-hints?
         *rival-max-precision*
         *rival-max-iterations*
         *rival-profile-executions*
         make-execution
         execution-name
         execution-number
         execution-precision
         execution-time
         execution-memory
         execution-iteration)

(provide baseline-compile
         baseline-apply)

(define baseline-compile rival-compile)
(define baseline-apply rival-apply)
