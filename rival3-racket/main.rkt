#lang racket

(require ffi/unsafe
         ffi/unsafe/define
         racket/runtime-path
         math/bigfloat
         math/flonum
         (only-in math/private/bigfloat/mpfr _mpfr-pointer)
         "ops.rkt")

(provide rival-compile
         rival-apply
         baseline-compile
         baseline-apply
         rival-analyze-with-hints
         rival-analyze
         baseline-analyze-with-hints
         baseline-analyze
         rival-profile
         rival-set-profiling!
         rival-profiling-enabled?
         (struct-out exn:rival)
         (struct-out exn:rival:invalid)
         (struct-out exn:rival:unsamplable)
         (struct-out execution)
         (struct-out discretization)
         (struct-out ival)
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
         execution-iteration
         (all-from-out "ops.rkt"))

(struct exn:rival exn:fail ())
(struct exn:rival:invalid exn:rival (pt))
(struct exn:rival:unsamplable exn:rival (pt))
(struct execution (name number precision time memory iteration) #:prefab)
(struct discretization (target convert distance type))
(struct ival (lo hi) #:transparent)

(define (make-execution name number precision time memory iteration)
  (execution name number precision time memory iteration))

(define (bf->bool x)
  (and (not (bfzero? x)) #t))

(define boolean-discretization (discretization 53 bf->bool (lambda (x y) (if (eq? x y) 0 2)) 'bool))
(define flonum-discretization (discretization 53 bigfloat->flonum (compose abs flonums-between) 'f64))

(define *rival-max-precision* (make-parameter 10000))
(define *rival-max-iterations* (make-parameter 5))
(define *rival-profile-executions* (make-parameter 1000))

(define-runtime-path librival-path
                     (build-path ".."
                                 "rival3-ffi"
                                 "target"
                                 "release"
                                 (string-append (case (system-type)
                                                  [(windows) "rival3_ffi"]
                                                  [else "librival3_ffi"])
                                                (bytes->string/utf-8 (system-type 'so-suffix)))))

(define-ffi-definer define-rival (ffi-lib librival-path))

(define _rival-error (_enum '(ok = 0 invalid_input = -1 unsamplable = -2 panic = -99) _int32))
(define _analyze-result (_list-struct _rival-error _stdbool _stdbool _stdbool _pointer))
(define _profile-summary (_list-struct _pointer _size _uint32 _uint32))
(define _execution-record (_list-struct _int32 _uint32 _double _uint32))
(define execution-record-size (ctype-sizeof _execution-record))
(define _aggregated-entry (_list-struct _int32 _uint32 _double _size))
(define aggregated-entry-size (ctype-sizeof _aggregated-entry))

(define _rival-profiling-mode (_enum '(off = 0 on = 1) _int32))
(define _rival-disc-type (_enum '(bool = 0 f32 = 1 f64 = 2) _uint32))

(define-rival rival_version (_fun -> _uint32))
(define-rival rival_disc_f64 (_fun _uint32 -> _pointer))
(define-rival rival_disc_f32 (_fun _uint32 -> _pointer))
(define-rival rival_disc_bool (_fun -> _pointer))
(define-rival rival_disc_mixed (_fun _pointer _size _uint32 -> _pointer))
(define-rival rival_disc_free (_fun _pointer -> _void))

(define-rival rival_expr_arena_new (_fun -> _pointer))
(define-rival rival_expr_arena_free (_fun _pointer -> _void))

(define-rival rival_expr_var (_fun _pointer _string -> _uint32))
(define-rival rival_expr_f64 (_fun _pointer _double -> _uint32))
(define-rival rival_expr_rational (_fun _pointer _int64 _int64 -> _uint32))
(define-rival rival_expr_bigint (_fun _pointer _string -> _uint32))
(define-rival rival_expr_bigrational (_fun _pointer _string _string -> _uint32))
(define-rival rival_expr_pi (_fun _pointer -> _uint32))
(define-rival rival_expr_e (_fun _pointer -> _uint32))

(define-rival rival_expr_neg (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_fabs (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_sqrt (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_cbrt (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_pow2 (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_exp (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_exp2 (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_expm1 (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_log (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_log2 (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_log10 (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_log1p (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_logb (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_sin (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_cos (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_tan (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_asin (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_acos (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_atan (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_sinh (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_cosh (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_tanh (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_asinh (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_acosh (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_atanh (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_erf (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_erfc (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_rint (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_round (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_ceil (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_floor (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_trunc (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_not (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_assert (_fun _pointer _uint32 -> _uint32))
(define-rival rival_expr_error (_fun _pointer _uint32 -> _uint32))

(define-rival rival_expr_add (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_sub (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_mul (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_div (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_pow (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_hypot (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_fmin (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_fmax (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_fdim (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_copysign (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_fmod (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_remainder (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_atan2 (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_and (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_or (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_eq (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_ne (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_lt (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_le (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_gt (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_ge (_fun _pointer _uint32 _uint32 -> _uint32))

(define-rival rival_expr_sinu (_fun _pointer _uint64 _uint32 -> _uint32))
(define-rival rival_expr_cosu (_fun _pointer _uint64 _uint32 -> _uint32))
(define-rival rival_expr_tanu (_fun _pointer _uint64 _uint32 -> _uint32))

(define-rival rival_expr_fma (_fun _pointer _uint32 _uint32 _uint32 -> _uint32))
(define-rival rival_expr_if (_fun _pointer _uint32 _uint32 _uint32 -> _uint32))

(define-rival rival_machine_new
              (_fun _pointer _pointer _size _pointer _size _pointer _uint32 _size -> _pointer))
(define-rival rival_machine_free (_fun _pointer -> _void))
(define-rival rival_machine_configure_baseline (_fun _pointer -> _stdbool))
(define-rival rival_machine_instruction_count (_fun _pointer -> _size))
(define-rival rival_machine_var_count (_fun _pointer -> _size))
(define-rival rival_machine_expr_count (_fun _pointer -> _size))
(define-rival rival_machine_iterations (_fun _pointer -> _uint32))
(define-rival rival_machine_bumps (_fun _pointer -> _uint32))
(define-rival rival_machine_set_profiling (_fun _pointer _rival-profiling-mode -> _void))
(define-rival rival_machine_get_profiling (_fun _pointer -> _rival-profiling-mode))

(define-rival rival_apply
              (_fun _pointer _pointer _size _pointer _size _pointer _size _uint32 -> _rival-error))

(define-rival rival_apply_baseline
              (_fun _pointer _pointer _size _pointer _size _pointer _uint32 -> _rival-error))

(define-rival rival_analyze_with_hints (_fun _pointer _pointer _size _pointer -> _analyze-result))
(define-rival rival_analyze_baseline_with_hints
              (_fun _pointer _pointer _size _pointer -> _analyze-result))

(define-rival rival_hints_free (_fun _pointer -> _void))
(define-rival rival_hints_len (_fun _pointer -> _size))

(define-rival rival_profiler_count (_fun _pointer -> _size))
(define-rival rival_profiler_reset (_fun _pointer -> _void))
(define-rival rival_profiler_aggregate (_fun _pointer _uint32 -> _profile-summary))
(define-rival rival_profiler_executions
              (_fun _pointer (out : (_ptr o _size)) -> (ptr : _pointer) -> (values ptr out)))

(define-rival rival_instruction_names
              (_fun _pointer (out : (_ptr o _size)) -> (ptr : _pointer) -> (values ptr out)))

(struct machine-wrapper (ptr n-vars n-exprs discs arg-buf arg-bfs out-buf out-bfs rect-buf name-table)
  #:property prop:cpointer
  (struct-field-index ptr))

(struct hints-wrapper (ptr) #:property prop:cpointer (struct-field-index ptr))

(define rival-machine? machine-wrapper?)
(define rival-hints? hints-wrapper?)

(define (machine-destroy wrapper)
  (when (machine-wrapper-ptr wrapper)
    (rival_machine_free (machine-wrapper-ptr wrapper)))
  (free-ptr (machine-wrapper-arg-buf wrapper))
  (free-ptr (machine-wrapper-out-buf wrapper))
  (free-ptr (machine-wrapper-rect-buf wrapper)))

(define (hints-destroy wrapper)
  (when (hints-wrapper-ptr wrapper)
    (rival_hints_free (hints-wrapper-ptr wrapper))))

(define (bytes-from-ptr ptr len)
  (define b (make-bytes len))
  (memcpy b ptr len)
  b)

;; Fold a list of args using a binary FFI function
(define (fold-binary-ffi arena binary-fn args)
  (foldl (lambda (arg acc) (binary-fn arena acc (expr->ffi arena arg)))
         (expr->ffi arena (car args))
         (cdr args)))

;; Build chained comparisons (< a b c) => (and (< a b) (< b c))
(define (chain-compare-ffi arena cmp-fn args)
  (define ffi-args (map (lambda (a) (expr->ffi arena a)) args))
  (define comparisons
    (for/list ([i (in-range (sub1 (length ffi-args)))])
      (cmp-fn arena (list-ref ffi-args i) (list-ref ffi-args (add1 i)))))
  (foldl (lambda (cmp acc) (rival_expr_and arena acc cmp)) (car comparisons) (cdr comparisons)))

;; Convert a Racket expression to FFI handle using the arena
(define (expr->ffi arena expr)
  (match expr
    ;; Constants
    ['PI (rival_expr_pi arena)]
    ['E (rival_expr_e arena)]
    ['TRUE (rival_expr_f64 arena 1.0)]
    ['FALSE (rival_expr_f64 arena 0.0)]
    ['INFINITY (rival_expr_f64 arena +inf.0)]
    ['NAN (rival_expr_f64 arena +nan.0)]
    [`(PI) (rival_expr_pi arena)]
    [`(E) (rival_expr_e arena)]
    [`(TRUE) (rival_expr_f64 arena 1.0)]
    [`(FALSE) (rival_expr_f64 arena 0.0)]
    [`(INFINITY) (rival_expr_f64 arena +inf.0)]
    [`(NAN) (rival_expr_f64 arena +nan.0)]
    ;; Variables
    [(? symbol?) (rival_expr_var arena (symbol->string expr))]
    ;; Numeric literals
    [(? exact-integer?) (rival_expr_bigint arena (number->string expr))]
    [(? rational?)
     (if (integer? expr)
         (rival_expr_bigint arena (number->string (inexact->exact expr)))
         (rival_expr_bigrational arena
                                 (number->string (numerator expr))
                                 (number->string (denominator expr))))]
    [(? real?) (rival_expr_f64 arena (exact->inexact expr))] ; +inf.0, -inf.0, +nan.0, -nan.0
    ;; Unary operators
    [`(- ,x) (rival_expr_neg arena (expr->ffi arena x))]
    [`(neg ,x) (rival_expr_neg arena (expr->ffi arena x))]
    [`(fabs ,x) (rival_expr_fabs arena (expr->ffi arena x))]
    [`(sqrt ,x) (rival_expr_sqrt arena (expr->ffi arena x))]
    [`(cbrt ,x) (rival_expr_cbrt arena (expr->ffi arena x))]
    [`(exp ,x) (rival_expr_exp arena (expr->ffi arena x))]
    [`(exp2 ,x) (rival_expr_exp2 arena (expr->ffi arena x))]
    [`(expm1 ,x) (rival_expr_expm1 arena (expr->ffi arena x))]
    [`(log ,x) (rival_expr_log arena (expr->ffi arena x))]
    [`(log2 ,x) (rival_expr_log2 arena (expr->ffi arena x))]
    [`(log10 ,x) (rival_expr_log10 arena (expr->ffi arena x))]
    [`(log1p ,x) (rival_expr_log1p arena (expr->ffi arena x))]
    [`(logb ,x) (rival_expr_logb arena (expr->ffi arena x))]
    [`(sin ,x) (rival_expr_sin arena (expr->ffi arena x))]
    [`(cos ,x) (rival_expr_cos arena (expr->ffi arena x))]
    [`(tan ,x) (rival_expr_tan arena (expr->ffi arena x))]
    [`(asin ,x) (rival_expr_asin arena (expr->ffi arena x))]
    [`(acos ,x) (rival_expr_acos arena (expr->ffi arena x))]
    [`(atan ,x) (rival_expr_atan arena (expr->ffi arena x))]
    [`(sinh ,x) (rival_expr_sinh arena (expr->ffi arena x))]
    [`(cosh ,x) (rival_expr_cosh arena (expr->ffi arena x))]
    [`(tanh ,x) (rival_expr_tanh arena (expr->ffi arena x))]
    [`(asinh ,x) (rival_expr_asinh arena (expr->ffi arena x))]
    [`(acosh ,x) (rival_expr_acosh arena (expr->ffi arena x))]
    [`(atanh ,x) (rival_expr_atanh arena (expr->ffi arena x))]
    [`(erf ,x) (rival_expr_erf arena (expr->ffi arena x))]
    [`(erfc ,x) (rival_expr_erfc arena (expr->ffi arena x))]
    [`(rint ,x) (rival_expr_rint arena (expr->ffi arena x))]
    [`(round ,x) (rival_expr_round arena (expr->ffi arena x))]
    [`(ceil ,x) (rival_expr_ceil arena (expr->ffi arena x))]
    [`(floor ,x) (rival_expr_floor arena (expr->ffi arena x))]
    [`(trunc ,x) (rival_expr_trunc arena (expr->ffi arena x))]
    [`(not ,x) (rival_expr_not arena (expr->ffi arena x))]
    [`(assert ,x) (rival_expr_assert arena (expr->ffi arena x))]
    [`(error ,x) (rival_expr_error arena (expr->ffi arena x))]
    ;; Multi-arity arithmetic operators (fold into binary)
    [`(+ ,x) (expr->ffi arena x)] ; unary + is identity
    [`(+ ,x ,y) (rival_expr_add arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(+ ,x ,y ,rest ...) (fold-binary-ffi arena rival_expr_add (list* x y rest))]
    [`(* ,x) (expr->ffi arena x)] ; unary * is identity
    [`(* ,x ,y) (rival_expr_mul arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(* ,x ,y ,rest ...) (fold-binary-ffi arena rival_expr_mul (list* x y rest))]
    ;; Binary-only arithmetic operators
    [`(- ,x ,y) (rival_expr_sub arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(/ ,x ,y) (rival_expr_div arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(pow ,x ,y) (rival_expr_pow arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(hypot ,x ,y) (rival_expr_hypot arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(fmin ,x ,y) (rival_expr_fmin arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(fmax ,x ,y) (rival_expr_fmax arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(fdim ,x ,y) (rival_expr_fdim arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(copysign ,x ,y) (rival_expr_copysign arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(fmod ,x ,y) (rival_expr_fmod arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(remainder ,x ,y) (rival_expr_remainder arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(atan2 ,x ,y) (rival_expr_atan2 arena (expr->ffi arena x) (expr->ffi arena y))]
    ;; Multi-arity logical operators (fold into binary)
    [`(and ,x) (expr->ffi arena x)] ; unary and is identity
    [`(and ,x ,y) (rival_expr_and arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(and ,x ,y ,rest ...) (fold-binary-ffi arena rival_expr_and (list* x y rest))]
    [`(or ,x) (expr->ffi arena x)] ; unary or is identity
    [`(or ,x ,y) (rival_expr_or arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(or ,x ,y ,rest ...) (fold-binary-ffi arena rival_expr_or (list* x y rest))]
    ;; Binary equality operators
    [`(== ,x ,y) (rival_expr_eq arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(!= ,x ,y) (rival_expr_ne arena (expr->ffi arena x) (expr->ffi arena y))]
    ;; Chained comparison operators: (< a b c) => (and (< a b) (< b c))
    [`(< ,x ,y) (rival_expr_lt arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(< ,x ,y ,rest ...) (chain-compare-ffi arena rival_expr_lt (list* x y rest))]
    [`(<= ,x ,y) (rival_expr_le arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(<= ,x ,y ,rest ...) (chain-compare-ffi arena rival_expr_le (list* x y rest))]
    [`(> ,x ,y) (rival_expr_gt arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(> ,x ,y ,rest ...) (chain-compare-ffi arena rival_expr_gt (list* x y rest))]
    [`(>= ,x ,y) (rival_expr_ge arena (expr->ffi arena x) (expr->ffi arena y))]
    [`(>= ,x ,y ,rest ...) (chain-compare-ffi arena rival_expr_ge (list* x y rest))]
    ;; Ternary operators
    [`(fma ,a ,b ,c)
     (rival_expr_fma arena (expr->ffi arena a) (expr->ffi arena b) (expr->ffi arena c))]
    [`(if ,c ,t ,f) (rival_expr_if arena (expr->ffi arena c) (expr->ffi arena t) (expr->ffi arena f))]
    [_ (error 'expr->ffi "Unknown expression: ~a" expr)]))

(define (disc->ffi disc)
  (case (discretization-type disc)
    [(bool) (rival_disc_bool)]
    [(f32) (rival_disc_f32 (discretization-target disc))]
    [(f64) (rival_disc_f64 (discretization-target disc))]
    [else (rival_disc_f64 (discretization-target disc))]))

(define (discs->ffi discs expected-len)
  (cond
    [(null? discs) (rival_disc_f64 53)]
    [(= (length discs) 1) (disc->ffi (car discs))]
    [else
     (define target (discretization-target (car discs)))
     (unless (andmap (lambda (d) (= (discretization-target d) target)) (cdr discs))
       (error 'rival-compile "All discretizations must have the same target"))
     (define n (length discs))
     (define types-arr (malloc _rival-disc-type n 'raw))
     (for ([i (in-range n)]
           [d (in-list discs)])
       (ptr-set! types-arr _rival-disc-type i (discretization-type d)))
     (define disc-ptr (rival_disc_mixed types-arr n target))
     (free-ptr types-arr)
     disc-ptr]))

(define (rival-compile exprs vars discs)
  (define n-vars (length vars))
  (define n-exprs (length exprs))

  (define arena (rival_expr_arena_new))
  (define expr-handles (map (lambda (e) (expr->ffi arena e)) exprs))
  (define exprs-arr (malloc _uint32 n-exprs 'raw))
  (for ([i (in-naturals)]
        [handle (in-list expr-handles)])
    (ptr-set! exprs-arr _uint32 i handle))

  (define vars-arr (malloc _pointer n-vars 'raw))
  (for ([i (in-naturals)]
        [var (in-list vars)])
    (ptr-set! vars-arr _pointer i (cast (symbol->string var) _string _pointer)))

  (define disc-ptr (discs->ffi discs n-exprs))

  (define machine-ptr
    (rival_machine_new arena
                       exprs-arr
                       n-exprs
                       vars-arr
                       n-vars
                       disc-ptr
                       (*rival-max-precision*)
                       (*rival-profile-executions*)))

  (free-ptr exprs-arr)
  (free-ptr vars-arr)
  (rival_disc_free disc-ptr)
  (rival_expr_arena_free arena)

  (when (not machine-ptr)
    (error 'rival-compile "Failed to create machine"))

  (define arg-buf (malloc _pointer n-vars 'raw))
  (define out-buf (malloc _pointer n-exprs 'raw))
  (define rect-buf (malloc _pointer (* 2 n-vars) 'raw))

  (define arg-bfs
    (build-vector n-vars
                  (lambda (_)
                    (parameterize ([bf-precision 53])
                      (bf 0.0)))))
  (for ([i (in-range n-vars)]
        [bf (in-vector arg-bfs)])
    (ptr-set! arg-buf _mpfr-pointer i bf))

  (define out-bfs
    (build-vector n-exprs
                  (lambda (_)
                    (parameterize ([bf-precision (*rival-max-precision*)])
                      (bf 0.0)))))
  (for ([i (in-range n-exprs)]
        [bf (in-vector out-bfs)])
    (ptr-set! out-buf _mpfr-pointer i bf))

  (define-values (names-ptr names-len) (rival_instruction_names machine-ptr))
  (define name-table
    (if (and names-ptr (> names-len 0))
        (list->vector (string-split (bytes->string/utf-8 (bytes-from-ptr names-ptr names-len)) "\0"))
        (vector)))

  (define wrapper
    (machine-wrapper machine-ptr
                     n-vars
                     n-exprs
                     discs
                     arg-buf
                     arg-bfs
                     out-buf
                     out-bfs
                     rect-buf
                     name-table))
  (register-finalizer wrapper machine-destroy)
  wrapper)

(define (baseline-compile exprs vars discs)
  (define machine (rival-compile exprs vars discs))
  (unless (rival_machine_configure_baseline (machine-wrapper-ptr machine))
    (error 'baseline-compile "Failed to configure baseline machine"))
  machine)

(define (rival-apply machine pt [hints #f])
  (define n-args (vector-length pt))
  (define arg-ptrs (machine-wrapper-arg-buf machine))

  (when (> n-args 0)
    (for ([i (in-range n-args)]
          [bf (in-vector pt)])
      (ptr-set! arg-ptrs _mpfr-pointer i bf)))

  (define n-outs (machine-wrapper-n-exprs machine))
  (define out-bfs (machine-wrapper-out-bfs machine))
  (define out-ptrs (machine-wrapper-out-buf machine))

  (define hints-ptr (and hints (hints-wrapper-ptr hints)))
  (define args-to-pass (and (> n-args 0) arg-ptrs))

  (define status-code
    (rival_apply (machine-wrapper-ptr machine)
                 args-to-pass
                 n-args
                 out-ptrs
                 n-outs
                 hints-ptr
                 (*rival-max-iterations*)
                 (*rival-max-precision*)))

  (match status-code
    ['ok
     (define discs (machine-wrapper-discs machine))
     (for/vector #:length n-outs
                 ([bf (in-vector out-bfs)]
                  [disc (in-list discs)])
       ((discretization-convert disc) bf))]
    ['invalid_input (raise (exn:rival:invalid "Invalid input" (current-continuation-marks) pt))]
    ['unsamplable (raise (exn:rival:unsamplable "Unsamplable input" (current-continuation-marks) pt))]
    ['panic (error 'rival-apply "Rival panic")]
    [else (error 'rival-apply "Unknown result code: ~a" status-code)]))

(define (baseline-apply machine pt [hints #f])
  (define n-args (vector-length pt))
  (define arg-ptrs (machine-wrapper-arg-buf machine))

  (when (> n-args 0)
    (for ([i (in-range n-args)]
          [bf (in-vector pt)])
      (ptr-set! arg-ptrs _mpfr-pointer i bf)))

  (define n-outs (machine-wrapper-n-exprs machine))
  (define out-bfs (machine-wrapper-out-bfs machine))
  (define out-ptrs (machine-wrapper-out-buf machine))

  (define hints-ptr (and hints (hints-wrapper-ptr hints)))
  (define args-to-pass (and (> n-args 0) arg-ptrs))

  (define status-code
    (rival_apply_baseline (machine-wrapper-ptr machine)
                          args-to-pass
                          n-args
                          out-ptrs
                          n-outs
                          hints-ptr
                          (*rival-max-precision*)))

  (match status-code
    ['ok
     (define discs (machine-wrapper-discs machine))
     (for/vector #:length n-outs
                 ([bf (in-vector out-bfs)]
                  [disc (in-list discs)])
       ((discretization-convert disc) bf))]
    ['invalid_input (raise (exn:rival:invalid "Invalid input" (current-continuation-marks) pt))]
    ['unsamplable (raise (exn:rival:unsamplable "Unsamplable input" (current-continuation-marks) pt))]
    ['panic (error 'baseline-apply "Rival panic")]
    [else (error 'baseline-apply "Unknown result code: ~a" status-code)]))

(define (rival-analyze-with-hints machine rect [hint #f])
  (define n-args (vector-length rect))
  (define rect-ptrs (machine-wrapper-rect-buf machine))

  (for ([i (in-range n-args)]
        [iv (in-vector rect)])
    (ptr-set! rect-ptrs _mpfr-pointer (* 2 i) (ival-lo iv))
    (ptr-set! rect-ptrs _mpfr-pointer (+ (* 2 i) 1) (ival-hi iv)))

  (define hint-ptr
    (if hint
        (hints-wrapper-ptr hint)
        #f))

  (match-define (list status-code is-error maybe-error converged hints-ptr)
    (rival_analyze_with_hints (machine-wrapper-ptr machine) rect-ptrs n-args hint-ptr))

  (match status-code
    ['panic (error 'rival-analyze-with-hints "Rival panic")]
    ['invalid_input (raise (exn:rival:invalid "Invalid input" (current-continuation-marks) rect))]
    ['ok (void)]
    [else (error 'rival-analyze-with-hints "Unknown result code: ~a" status-code)])

  (define new-hints
    (if hints-ptr
        (let ([wrapper (hints-wrapper hints-ptr)])
          (register-finalizer wrapper hints-destroy)
          wrapper)
        #f))

  (list (ival is-error maybe-error) new-hints converged))

(define (rival-analyze machine rect)
  (car (rival-analyze-with-hints machine rect)))

(define (baseline-analyze-with-hints machine rect [hint #f])
  (define n-args (vector-length rect))
  (define rect-ptrs (machine-wrapper-rect-buf machine))

  (for ([i (in-range n-args)]
        [iv (in-vector rect)])
    (ptr-set! rect-ptrs _mpfr-pointer (* 2 i) (ival-lo iv))
    (ptr-set! rect-ptrs _mpfr-pointer (+ (* 2 i) 1) (ival-hi iv)))

  (define hint-ptr
    (if hint
        (hints-wrapper-ptr hint)
        #f))

  (match-define (list status-code is-error maybe-error converged hints-ptr)
    (rival_analyze_baseline_with_hints (machine-wrapper-ptr machine) rect-ptrs n-args hint-ptr))

  (match status-code
    ['panic (error 'baseline-analyze-with-hints "Rival panic")]
    ['invalid_input (raise (exn:rival:invalid "Invalid input" (current-continuation-marks) rect))]
    ['ok (void)]
    [else (error 'baseline-analyze-with-hints "Unknown result code: ~a" status-code)])

  (define new-hints
    (if hints-ptr
        (let ([wrapper (hints-wrapper hints-ptr)])
          (register-finalizer wrapper hints-destroy)
          wrapper)
        #f))

  (list (ival is-error maybe-error) new-hints converged))

(define (baseline-analyze machine rect)
  (car (baseline-analyze-with-hints machine rect)))

(define (rival-profile machine param)
  (match param
    ['instructions (rival_machine_instruction_count (machine-wrapper-ptr machine))]
    ['iterations (rival_machine_iterations (machine-wrapper-ptr machine))]
    ['bumps (rival_machine_bumps (machine-wrapper-ptr machine))]
    ['executions
     (define-values (ptr len) (rival_profiler_executions (machine-wrapper-ptr machine)))
     (cond
       [(or (not ptr) (zero? len)) (vector)]
       [else
        (define names (machine-wrapper-name-table machine))
        (for/vector #:length len
                    ([i (in-range len)])
          (define rec-ptr (ptr-add ptr (* i execution-record-size)))
          (match-define (list instr-idx prec time-ms iter) (ptr-ref rec-ptr _execution-record))
          (define name
            (cond
              [(and (vector? names) (<= 0 instr-idx) (< instr-idx (vector-length names)))
               (vector-ref names instr-idx)]
              [(< instr-idx 0) "adjust"]
              [else ""]))
          (execution name instr-idx prec time-ms 0 iter))])]
    ;; Aggergated path (for herbie)
    ['summary
     (define bucket-size (max 1 (quotient (*rival-max-precision*) 25)))
     (match-define (list entries-ptr entries-len bumps iterations)
       (rival_profiler_aggregate (machine-wrapper-ptr machine) bucket-size))
     (define names (machine-wrapper-name-table machine))
     (define summary
       (if (or (not entries-ptr) (zero? entries-len))
           (vector)
           (for/vector #:length entries-len
                       ([i (in-range entries-len)])
             (define entry-ptr (ptr-add entries-ptr (* i aggregated-entry-size)))
             (define instr (ptr-ref entry-ptr _int32))
             (define name
               (cond
                 [(and (vector? names) (<= 0 instr) (< instr (vector-length names)))
                  (vector-ref names instr)]
                 [(< instr 0) "adjust"]
                 [else ""]))
             (list name
                   (ptr-ref (ptr-add entry-ptr 4) _uint32)
                   (ptr-ref (ptr-add entry-ptr 8) _double)
                   (ptr-ref (ptr-add entry-ptr 16) _size)))))
     (list summary bumps iterations)]))

(define (rival-set-profiling! machine enabled)
  (rival_machine_set_profiling (machine-wrapper-ptr machine) (if enabled 'on 'off)))

(define (rival-profiling-enabled? machine)
  (eq? (rival_machine_get_profiling (machine-wrapper-ptr machine)) 'on))

(define ffi-free (get-ffi-obj "free" #f (_fun _pointer -> _void)))
(define (free-ptr p)
  (when p
    (ffi-free p)))
