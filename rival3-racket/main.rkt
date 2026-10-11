#lang racket

(require ffi/unsafe
         ffi/unsafe/define
         (only-in racket/contract [-> c:->] [->i c:->i])
         racket/runtime-path
         math/bigfloat
         (only-in math/private/bigfloat/mpfr _mpfr-pointer)
         "ops.rkt")

;; No contract for functions that tend to be extremely hot.
(provide rival-apply
         rival-apply/f64
         rival-analyze-with-hints
         rival-analyze)

(provide (contract-out [rival-compile compile/c]
                       [baseline-compile compile/c]
                       [rival-profile (c:-> rival-machine? profile-key/c any)]
                       [rival-set-profiling! (c:-> rival-machine? any/c void?)]
                       [rival-profiling-enabled? (c:-> rival-machine? boolean?)])
         rival-machine?
         rival-hints?
         boolean-discretization
         flonum-discretization
         (struct-out exn:rival)
         (struct-out exn:rival:invalid)
         (struct-out exn:rival:unsamplable)
         (struct-out execution)
         (struct-out discretization)
         (struct-out ival)
         *rival-max-precision*
         *rival-max-iterations*
         *rival-profile-executions*
         (all-from-out "ops.rkt"))

(struct exn:rival exn:fail ())
(struct exn:rival:invalid exn:rival (pt))
(struct exn:rival:unsamplable exn:rival (pt))
(struct execution (name number precision time memory iteration) #:prefab)
(struct discretization (type target convert))
(struct ival (lo hi) #:transparent)

(define (bf->bool x)
  (and (not (bfzero? x)) #t))

(define (input->bf x)
  (if (boolean? x)
      (bf (if x 1 0))
      x))

(define (input->double x)
  (if (boolean? x)
      (if x 1.0 0.0)
      x))

(define boolean-discretization (discretization 'bool 53 bf->bool))
(define flonum-discretization (discretization 'f64 53 bigfloat->flonum))

(define *rival-max-precision* (make-parameter 10000))
(define *rival-max-iterations* (make-parameter 5))
(define *rival-profile-executions* (make-parameter 1000))

(define-runtime-path native-root (build-path "private" "native"))
;; Dev fallback for local builds of Rival 3
(define dev-native-root (build-path native-root ".." ".." ".." "rival3-ffi" "target" "release"))

(define lib-name
  (string-append (case (system-type)
                   [(windows) "rival3_ffi"]
                   [else "librival3_ffi"])
                 (bytes->string/utf-8 (system-type 'so-suffix))))

(define lib-path
  (let ([pkg-path (build-path native-root (system-library-subpath #f) lib-name)])
    (if (file-exists? pkg-path)
        pkg-path
        (build-path dev-native-root lib-name))))

(define-ffi-definer define-rival (ffi-lib lib-path))

(define abi-version 3)
(define-rival rival_version (_fun -> _uint32))
(let ([v (rival_version)])
  (unless (= v abi-version)
    (error 'rival3 "ABI version mismatch: expected ~a, got ~a" abi-version v)))

;; Operator names in Rival expressions, mapped to the codes of the Rust enums in
;; rival3-ffi/src/expr.rs.
(define unary-ops
  #hasheq((neg . 0)
          (fabs . 1)
          (sqrt . 2)
          (cbrt . 3)
          (pow2 . 4)
          (exp . 5)
          (exp2 . 6)
          (expm1 . 7)
          (log . 8)
          (log2 . 9)
          (log10 . 10)
          (log1p . 11)
          (logb . 12)
          (sin . 13)
          (cos . 14)
          (tan . 15)
          (asin . 16)
          (acos . 17)
          (atan . 18)
          (sinh . 19)
          (cosh . 20)
          (tanh . 21)
          (asinh . 22)
          (acosh . 23)
          (atanh . 24)
          (erf . 25)
          (erfc . 26)
          (lgamma . 27)
          (tgamma . 28)
          (rint . 29)
          (round . 30)
          (ceil . 31)
          (floor . 32)
          (trunc . 33)
          (not . 34)
          (assert . 35)
          (error . 36)))

(define unary-param-ops #hasheq((cosu . 0) (sinu . 1) (tanu . 2)))

(define binary-ops
  #hasheq((+ . 0)
          (- . 1)
          (* . 2)
          (/ . 3)
          (pow . 4)
          (hypot . 5)
          (fmin . 6)
          (fmax . 7)
          (fdim . 8)
          (copysign . 9)
          (fmod . 10)
          (remainder . 11)
          (atan2 . 12)
          (and . 13)
          (or . 14)
          (== . 15)
          (!= . 16)
          (< . 17)
          (<= . 18)
          (> . 19)
          (>= . 20)))

(define ternary-ops #hasheq((fma . 0) (if . 1)))

(define variadic-ops (seteq '+ '* 'and 'or))
(define chainable-cmp-ops (seteq '< '<= '> '>=))

(define _rival-error (_enum '(ok = 0 invalid_input = -1 unsamplable = -2 misuse = -3) _int32))
(define _execution-record (_list-struct _int32 _uint32 _double _uint32))

(define _rival-strategy (_enum '(adaptive = 0 baseline = 1) _uint32))
(define _rival-output-policy (_enum '(require_all = 0 allow_partial = 1) _uint32))
(define _rival-disc-type (_enum '(bool = 0 f32 = 1 f64 = 2) _uint32))
;; Out-parameters are left unset on error, so accept any value here.
(define _rival-analysis (_enum '(valid = 0 maybe_invalid = 1 invalid = 2) _uint32 #:unknown values))

(define RIVAL_EXPR_INVALID #xFFFFFFFF)

(define-rival rival_expr_builder_new
              (_fun (vars : (_list i _string))
                    (_size = (length vars))
                    (out : (_ptr o _pointer))
                    ->
                    (status : _rival-error)
                    ->
                    (values status out)))
(define-rival rival_expr_builder_free (_fun _pointer -> _void))

(define-rival rival_expr_var (_fun _pointer _string -> _uint32))
(define-rival rival_expr_f64 (_fun _pointer _double -> _uint32))
(define-rival rival_expr_bigint (_fun _pointer _string -> _uint32))
(define-rival rival_expr_bigrational (_fun _pointer _string _string -> _uint32))
(define-rival rival_expr_pi (_fun _pointer -> _uint32))
(define-rival rival_expr_e (_fun _pointer -> _uint32))
(define-rival rival_expr_infinity (_fun _pointer -> _uint32))

(define-rival rival_expr_unary (_fun _pointer _uint32 _uint32 -> _uint32))
(define-rival rival_expr_unary_param (_fun _pointer _uint32 _uint32 _uint32 -> _uint32))
(define-rival rival_expr_binary (_fun _pointer _uint32 _uint32 _uint32 -> _uint32))
(define-rival rival_expr_ternary (_fun _pointer _uint32 _uint32 _uint32 _uint32 -> _uint32))

(define-rival rival_machine_options_new
              (_fun (out : (_ptr o _pointer)) -> (status : _rival-error) -> (values status out)))
(define-rival rival_machine_options_free (_fun _pointer -> _void))
(define-rival rival_machine_options_set_target_precision (_fun _pointer _uint32 -> _rival-error))
(define-rival rival_machine_options_set_max_precision (_fun _pointer _uint32 -> _rival-error))
(define-rival rival_machine_options_set_profile_capacity (_fun _pointer _size -> _rival-error))
(define-rival rival_machine_options_set_strategy (_fun _pointer _rival-strategy -> _rival-error))

(define-rival rival_machine_new
              (_fun _pointer
                    (handles : (_list i _uint32))
                    (types : (_list i _rival-disc-type))
                    (_size = (length handles))
                    _pointer
                    (out : (_ptr o _pointer))
                    ->
                    (status : _rival-error)
                    ->
                    (values status out)))
(define-rival rival_machine_free (_fun _pointer -> _void))
(define-rival rival_machine_instruction_count (_fun _pointer -> _size))
(define-rival rival_machine_iterations (_fun _pointer -> _uint32))
(define-rival rival_machine_bumps (_fun _pointer -> _uint32))
(define-rival rival_machine_set_profiling (_fun _pointer _stdbool -> _rival-error))
(define-rival rival_machine_get_profiling (_fun _pointer -> _stdbool))

(define _apply-fun
  (_fun _pointer _pointer _size _pointer _uint32 _rival-output-policy _pointer _size -> _rival-error))
(define-rival rival_machine_apply _apply-fun)
(define-rival rival_machine_apply_f64 _apply-fun)

(define-rival rival_machine_analyze
              (_fun _pointer
                    _pointer
                    _size
                    _pointer
                    _rival-output-policy
                    (analysis : (_ptr o _rival-analysis))
                    (converged : (_ptr o _stdbool))
                    (hints : (_ptr o _pointer))
                    ->
                    (status : _rival-error)
                    ->
                    (values status analysis converged hints)))
(define-rival rival_machine_analyze/no-hints
              (_fun _pointer
                    _pointer
                    _size
                    (_pointer = #f)
                    _rival-output-policy
                    (analysis : (_ptr o _rival-analysis))
                    (_pointer = #f)
                    (_pointer = #f)
                    ->
                    (status : _rival-error)
                    ->
                    (values status analysis))
              #:c-id rival_machine_analyze)

(define-rival rival_hints_free (_fun _pointer -> _void))

(define-rival rival_machine_profiler_executions
              (_fun _pointer
                    _pointer
                    _size
                    (len : (_ptr o _size))
                    ->
                    (status : _rival-error)
                    ->
                    (values status len)))

(define-rival rival_machine_instruction_names
              (_fun _pointer
                    (names : (_ptr o _pointer))
                    (len : (_ptr o _size))
                    ->
                    (status : _rival-error)
                    ->
                    (values status names len)))

;; `arg-buf` holds 2 * n-vars mpfr pointers: apply uses the first n-vars,
;; analyze uses all of them as lo/hi pairs. `arg-bfs` and `out-bfs` keep the
;; pointed-to bigfloats alive, since the buffers are not traced.
(struct rival-machine
        (ptr n-vars
             discs
             arg-buf
             arg-bfs
             out-buf
             out-bfs
             f64-arg-buf
             f64-out-buf
             instruction-names
             exec-buf
             exec-capacity)
  #:property prop:cpointer
  0)

(struct rival-hints (ptr machine) #:property prop:cpointer 0)

(define discretization/c
  (struct/c discretization (or/c 'bool 'f32 'f64) exact-positive-integer? procedure?))

(define compile/c
  (c:->i ([exprs list?] [vars (listof symbol?)]
                        [discs
                         (exprs)
                         (and/c (listof discretization/c)
                                (lambda (discs) (= (length discs) (length exprs))))])
         [machine rival-machine?]))

(define profile-key/c (or/c 'instructions 'iterations 'bumps 'executions))

(define (free-machine machine)
  (rival_machine_free (rival-machine-ptr machine)))

(define (free-hints hints)
  (rival_hints_free (rival-hints-ptr hints)))

(define (make-expr-compiler builder who)
  ;; Compile a shared subexpression once, so that a shared input expression
  ;; does not expand into a tree of FFI calls.
  (define cache (make-hasheq))

  (define (fold-binary op args)
    (foldl (lambda (arg acc) (rival_expr_binary builder op acc (compile arg)))
           (compile (car args))
           (cdr args)))

  ;; Build chained comparisons (< a b c) => (and (< a b) (< b c))
  (define (chain-compare op args)
    (define ffi-args (map compile args))
    (define comparisons
      (for/list ([lhs (in-list ffi-args)]
                 [rhs (in-list (cdr ffi-args))])
        (rival_expr_binary builder op lhs rhs)))
    (foldl (lambda (comparison acc)
             (rival_expr_binary builder (hash-ref binary-ops 'and) acc comparison))
           (car comparisons)
           (cdr comparisons)))

  (define (compile expr)
    (define handle (hash-ref! cache expr (lambda () (compile-node expr))))
    (when (= handle RIVAL_EXPR_INVALID)
      (raise-arguments-error who "Rival could not compile a subexpression" "subexpression" expr))
    handle)

  (define (compile-node expr)
    (match expr
      [(or 'PI '(PI)) (rival_expr_pi builder)]
      [(or 'E '(E)) (rival_expr_e builder)]
      [(or 'TRUE '(TRUE)) (rival_expr_f64 builder 1.0)]
      [(or 'FALSE '(FALSE)) (rival_expr_f64 builder 0.0)]
      [(or 'INFINITY '(INFINITY)) (rival_expr_infinity builder)]
      [(or 'NAN '(NAN)) (rival_expr_f64 builder +nan.0)]
      [(? symbol?) (rival_expr_var builder (symbol->string expr))]
      [(? exact-integer?) (rival_expr_bigint builder (number->string expr))]
      [(and (? rational?) (? exact?))
       (rival_expr_bigrational builder
                               (number->string (numerator expr))
                               (number->string (denominator expr)))]
      [(? flonum?) (rival_expr_f64 builder expr)]
      [`(- ,x) (rival_expr_unary builder (hash-ref unary-ops 'neg) (compile x))]
      [`(,op ,x)
       #:when (hash-has-key? unary-ops op)
       (rival_expr_unary builder (hash-ref unary-ops op) (compile x))]
      [`((,op ,n) ,x)
       #:when (hash-has-key? unary-param-ops op)
       (rival_expr_unary_param builder (hash-ref unary-param-ops op) n (compile x))]
      [`(,op ,x ,y)
       #:when (hash-has-key? binary-ops op)
       (rival_expr_binary builder (hash-ref binary-ops op) (compile x) (compile y))]
      [`(,op ,x ,y ,z)
       #:when (hash-has-key? ternary-ops op)
       (rival_expr_ternary builder (hash-ref ternary-ops op) (compile x) (compile y) (compile z))]
      [`(,op ,x ,y ,rest ...)
       #:when (set-member? variadic-ops op)
       (fold-binary (hash-ref binary-ops op) (list* x y rest))]
      [`(,op ,x ,y ,rest ...)
       #:when (set-member? chainable-cmp-ops op)
       (chain-compare (hash-ref binary-ops op) (list* x y rest))]
      [_ (raise-arguments-error who "unknown expression" "expression" expr)]))

  compile)

(define (discs-target discs who)
  (cond
    [(null? discs) 53]
    [else
     (define target (discretization-target (car discs)))
     (unless (andmap (lambda (d) (= (discretization-target d) target)) (cdr discs))
       (raise-arguments-error who
                              "all discretizations must have the same target"
                              "discretizations"
                              discs))
     target]))

(define (rival-compile exprs vars discs)
  (compile-inner exprs vars discs 'adaptive 'rival-compile))

(define (baseline-compile exprs vars discs)
  (compile-inner exprs vars discs 'baseline 'baseline-compile))

(define (make-machine-options who target max-precision profile-capacity strategy)
  (define-values (status options) (rival_machine_options_new))
  (unless (eq? status 'ok)
    (raise-arguments-error who "Rival could not allocate machine options"))
  (define (set-option! setter what value)
    (unless (eq? (setter options value) 'ok)
      (rival_machine_options_free options)
      (raise-arguments-error who (string-append "Rival rejected the " what) what value)))
  (set-option! rival_machine_options_set_target_precision "target precision" target)
  (set-option! rival_machine_options_set_max_precision "max precision" max-precision)
  (set-option! rival_machine_options_set_profile_capacity "profile capacity" profile-capacity)
  (set-option! rival_machine_options_set_strategy "strategy" strategy)
  options)

(define (compile-inner exprs vars discs strategy who)
  (define n-vars (length vars))
  (define n-exprs (length exprs))
  (define target (discs-target discs who))
  (define max-precision (*rival-max-precision*))
  (define profile-capacity (*rival-profile-executions*))

  (define options (make-machine-options who target max-precision profile-capacity strategy))
  (define-values (builder-status builder) (rival_expr_builder_new (map symbol->string vars)))
  (unless (eq? builder-status 'ok)
    (rival_machine_options_free options)
    (raise-arguments-error who "Rival rejected the variables" "vars" vars))

  (define-values (machine-status machine-ptr)
    (dynamic-wind void
                  (lambda ()
                    (define compile-expr (make-expr-compiler builder who))
                    (define expr-handles (map compile-expr exprs))
                    (rival_machine_new builder expr-handles (map discretization-type discs) options))
                  (lambda ()
                    (rival_expr_builder_free builder)
                    (rival_machine_options_free options))))

  (unless (eq? machine-status 'ok)
    (raise-arguments-error who "Rival rejected the expressions" "exprs" exprs))

  (define arg-buf (malloc _pointer (* 2 n-vars) 'atomic-interior))
  (define out-buf (malloc _pointer n-exprs 'atomic-interior))
  (define f64-arg-buf (malloc _double n-vars 'atomic-interior))
  (define f64-out-buf (malloc _double n-exprs 'atomic-interior))
  (define exec-buf (malloc _execution-record profile-capacity 'atomic-interior))

  (define arg-bfs (make-vector (* 2 n-vars) #f))

  ;; Rival writes outputs already rounded to their type, so a double's precision holds them.
  (define out-bfs
    (parameterize ([bf-precision 53])
      (build-vector n-exprs (lambda (_) (bf 0.0)))))
  (for ([i (in-range n-exprs)]
        [x (in-vector out-bfs)])
    (ptr-set! out-buf _mpfr-pointer i x))

  (define-values (names-status names-ptr names-len) (rival_machine_instruction_names machine-ptr))
  (check-ok! who names-status exprs)
  (define instruction-names
    (for/vector #:length names-len
                ([i (in-range names-len)])
      (ptr-ref names-ptr _string i)))

  (define machine
    (rival-machine machine-ptr
                   n-vars
                   discs
                   arg-buf
                   arg-bfs
                   out-buf
                   out-bfs
                   f64-arg-buf
                   f64-out-buf
                   instruction-names
                   exec-buf
                   profile-capacity))
  (register-finalizer machine free-machine)
  machine)

(define (check-arity! who what machine n)
  (unless (= n (rival-machine-n-vars machine))
    (raise-arguments-error who
                           (string-append what " has the wrong number of variables")
                           "expected"
                           (rival-machine-n-vars machine)
                           "given"
                           n)))

(define (check-hints! who machine hints)
  (when (and hints (not (eq? (rival-hints-machine hints) machine)))
    (raise-arguments-error who "hints do not belong to this machine")))

(define (check-ok! who status input)
  (match status
    ['ok (void)]
    ['invalid_input (raise (exn:rival:invalid "Invalid input" (current-continuation-marks) input))]
    ['unsamplable
     (raise (exn:rival:unsamplable "Unsamplable input" (current-continuation-marks) input))]
    ['misuse (raise-arguments-error who "Rival rejected the arguments" "input" input)]))

(define (output-policy partial?)
  (if partial? 'allow_partial 'require_all))

(define (rival-apply machine pt [hints #f] #:partial? [partial? #f])
  (define n-args (vector-length pt))
  (check-arity! 'rival-apply "point" machine n-args)
  (check-hints! 'rival-apply machine hints)
  (define arg-buf (rival-machine-arg-buf machine))
  (define arg-bfs (rival-machine-arg-bfs machine))
  (for ([i (in-range n-args)]
        [arg (in-vector pt)])
    (define x (input->bf arg))
    (vector-set! arg-bfs i x)
    (ptr-set! arg-buf _mpfr-pointer i x))
  (define out-bfs (rival-machine-out-bfs machine))
  (define n-outs (vector-length out-bfs))
  (check-ok! 'rival-apply
             (rival_machine_apply machine
                                  arg-buf
                                  n-args
                                  hints
                                  (*rival-max-iterations*)
                                  (output-policy partial?)
                                  (rival-machine-out-buf machine)
                                  n-outs)
             pt)
  (for/vector #:length n-outs
              ([bf (in-vector out-bfs)]
               [disc (in-list (rival-machine-discs machine))])
    (if (bfnan? bf)
        'invalid
        ((discretization-convert disc) bf))))

(define (rival-apply/f64 machine pt [hints #f] #:partial? [partial? #f])
  (define n-args (vector-length pt))
  (check-arity! 'rival-apply/f64 "point" machine n-args)
  (check-hints! 'rival-apply/f64 machine hints)
  (define args (rival-machine-f64-arg-buf machine))
  (for ([i (in-range n-args)]
        [arg (in-vector pt)])
    (ptr-set! args _double i (input->double arg)))
  (define discs (rival-machine-discs machine))
  (define n-outs (length discs))
  (define outs (rival-machine-f64-out-buf machine))
  (check-ok! 'rival-apply/f64
             (rival_machine_apply_f64 machine
                                      args
                                      n-args
                                      hints
                                      (*rival-max-iterations*)
                                      (output-policy partial?)
                                      outs
                                      n-outs)
             pt)
  (for/vector #:length n-outs
              ([i (in-range n-outs)]
               [disc (in-list discs)])
    (define x (ptr-ref outs _double i))
    (cond
      [(nan? x) 'invalid]
      [(eq? (discretization-type disc) 'bool) (not (zero? x))]
      [else x])))

;; Loads `rect` into the machine's argument buffer and returns its length.
(define (load-rect! who machine rect)
  (define n-args (vector-length rect))
  (check-arity! who "rectangle" machine n-args)
  (define arg-buf (rival-machine-arg-buf machine))
  (define arg-bfs (rival-machine-arg-bfs machine))
  (for ([i (in-range n-args)]
        [iv (in-vector rect)])
    (define lo (input->bf (ival-lo iv)))
    (define hi (input->bf (ival-hi iv)))
    (vector-set! arg-bfs (* 2 i) lo)
    (vector-set! arg-bfs (+ (* 2 i) 1) hi)
    (ptr-set! arg-buf _mpfr-pointer (* 2 i) lo)
    (ptr-set! arg-buf _mpfr-pointer (+ (* 2 i) 1) hi))
  n-args)

(define (analysis->ival analysis)
  (match analysis
    ['valid (ival #f #f)]
    ['maybe_invalid (ival #f #t)]
    ['invalid (ival #t #t)]))

(define (rival-analyze-with-hints machine rect [hints #f] #:partial? [partial? #f])
  (define n-args (load-rect! 'rival-analyze-with-hints machine rect))
  (check-hints! 'rival-analyze-with-hints machine hints)
  (define-values (status analysis converged hints-ptr)
    (rival_machine_analyze machine
                           (rival-machine-arg-buf machine)
                           n-args
                           hints
                           (output-policy partial?)))
  (check-ok! 'rival-analyze-with-hints status rect)
  (define new-hints (rival-hints hints-ptr machine))
  (register-finalizer new-hints free-hints)
  (list (analysis->ival analysis) new-hints converged))

(define (rival-analyze machine rect #:partial? [partial? #f])
  (define n-args (load-rect! 'rival-analyze machine rect))
  (define-values (status analysis)
    (rival_machine_analyze/no-hints machine
                                    (rival-machine-arg-buf machine)
                                    n-args
                                    (output-policy partial?)))
  (check-ok! 'rival-analyze status rect)
  (analysis->ival analysis))

(define (rival-profile machine param)
  (match param
    ['instructions (rival_machine_instruction_count machine)]
    ['iterations (rival_machine_iterations machine)]
    ['bumps (rival_machine_bumps machine)]
    ['executions
     (define buf (rival-machine-exec-buf machine))
     (define capacity (rival-machine-exec-capacity machine))
     (define-values (status len) (rival_machine_profiler_executions machine buf capacity))
     (check-ok! 'rival-profile status param)
     (define names (rival-machine-instruction-names machine))
     (define n (min len capacity))
     (for/vector #:length n
                 ([i (in-range n)])
       (match-define (list instr-idx prec time-ms iter) (ptr-ref buf _execution-record i))
       (define name
         (if (negative? instr-idx)
             "adjust"
             (vector-ref names instr-idx)))
       (execution name instr-idx prec time-ms 0 iter))]))

(define (rival-set-profiling! machine enabled)
  (check-ok! 'rival-set-profiling! (rival_machine_set_profiling machine enabled) enabled))

(define (rival-profiling-enabled? machine)
  (rival_machine_get_profiling machine))
