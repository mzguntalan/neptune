# neptune

JAX interop-able library for numeric and machine learning computations in Haskell. Neptune is Haskell library for machine learning and numerical computing that aims to be inter-operable with models done by the larger research and engineering community. To do this, Neptune would interop with at least one library/framework for numeric computing used by many (Currently targeting JAX). With this, Neptune will be save and load models from JAX. Neptune is a **work in progress** and is in very early development and can't be used for anything substantial as of now.

The neptune library probably won't be look like numpy, but it might. Neptune would hopefully make numeric computing in Haskell feel like Haskell.

## Demo

Currently, neptune can arbitarily compose (designated) functions that mirror lax module functions in jax. These function output a `Trace` that are compilable to `jaxpr`.

Here are examples

### Example 1: |x+y|

In Python

```python
def f(x, y):
    return jnp.abs(x + y)
```

In Haskell-lax mirror (this is not the neptune api, but rather the lax mirrored one which is strict on shapes and types). I apped an `l` in the beginning for now.

```haskell
f x y = labs (ladd x y)
```

Jaxpr for python

```python
x = jnp.ones([2,2], dtype=jnp.float32)
y = jnp.ones([2,2], dtype=jnp.float32)

make_jaxpr(f)(x,y)

Output:
{ lambda ; a:f32[2,2] b:f32[2,2]. let
    c:f32[2,2] = add a b
    d:f32[2,2] = abs c
  in (d,) }
```

Jaxpr from Haskell (the variable naming might different, but equivalent)

```haskell
f a b = labs (ladd a b)

Output:
{ lambda  ; a:f32[2,2] b:f32[2,2]. let
        c:f32[2,2] = add a b
        d:f32[2,2] = abs c
 in (d,) }
```

### Example 2: ((a+b) + (c+d)) + (some tensor created in a function)

```python
def f2(a,b,c,d):
    z = jnp.array([[1.2, 3.4], [2.3,1.1]], dtype=jnp.float32) # could have been any
    return (((a+b)+ (c+d)) + z)

# with f32[2,2] tensors x y
Output:
{ lambda a:f32[2,2]; b:f32[2,2] c:f32[2,2] d:f32[2,2] e:f32[2,2]. let
    f:f32[2,2] = add b c
    g:f32[2,2] = add d e
    h:f32[2,2] = add f g
    i:f32[2,2] = add h a
  in (i,) }

```

In haskell

```haskell
f2 a b c d = ((a `ladd` b) `ladd` (c `ladd` d)) `ladd` (lit (tensor Tf32 [2,2] "z" Tlit))
-- a nicer api will come soon

Output:
{ lambda a:f32[2,2] ; b:f32[2,2] c:f32[2,2] d:f32[2,2] e:f32[2,2]. let
        f:f32[2,2] = add b c
        g:f32[2,2] = add d e
        h:f32[2,2] = add f g
        i:f32[2,2] = add h a
 in (i,) }

```

### Example 3: concat([a+b, c+d], axis=1)

In python

```python
def f3(a,b,c,d):
    x = a + b
    y = c + d
    z = jnp.concatenate([x,y], axis=1)
    return z

Output:
{ lambda ; a:f32[2,2] b:f32[2,2] c:f32[2,2] d:f32[2,2]. let
    e:f32[2,2] = add a b
    f:f32[2,2] = add c d
    g:f32[2,4] = concatenate[dimension=1] e f
  in (g,) }
```

In haskell,

```haskell
f3 a b c d = lconcatenate [x,y] 1 where x = a `ladd` b; y = c `ladd` d

Output:
{ lambda  ; a:f32[2,2] b:f32[2,2] c:f32[2,2] d:f32[2,2]. let
        e:f32[2,2] = add a b
        f:f32[2,2] = add c d
        g:f32[2,4] = concatenate[dimension=1] e f
 in (g,) }
```

## Current Goals

- [x] Produce jaxpr
- [x] Map Some jaxpr primitives
- [x] Composing Primitives
- [ ] Develop Neptune representation
- [ ] interop with the XLA compiler to run them
- [ ] load and save JAX models in neptune Haskell
