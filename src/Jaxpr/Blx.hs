{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Jaxpr.Blx where

import Jaxpr.Blx.Lax
import Jaxpr.Blx.Primitives
import Jaxpr.Blx.Tensor
import Jaxpr.Blx.Trace

-- Blx Before Jax, Above Lax
-- This is a slightly modified mirror of Lax from Jax to Accomodate the Trace Approach
-- focusing on Tracing

-- test function
-- The resulting Trace of a function corresponds to Jaxpr
testFunction :: BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace
testFunction a b c d = lconcatenate [v1, v2] 0
  where
    v1 = ladd a b
    v2 = labs c `ladd` labs d

testFunction2 :: BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace
testFunction2 a b c d = ladd s z
  where
    z = lit Tf32 [2, 2] "z" -- how do i scope this to only testFunction2 automatically
    s = ladd a b `ladd` ladd c d

testFunction3 :: BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace
testFunction3 a b c d = ladd s z
  where
    z = lit Tf32 [2, 2] "z.3" -- how do i scope this to only testFunction3 vs 2 automatically and it's function stack number?
    s = ladd a b `ladd` ladd c (testFunction2 a b c d)

testFunction4 :: BlxTrace -> BlxTrace -> BlxTrace
testFunction4 a b = (a `ladd` b) `ladd` c
  where
    c = lbroadcastInDim v [2, 2]
    v = lit Tf32 [] "v"

testFunction5 :: BlxTrace -> BlxTrace -> Int -> BlxTrace
testFunction5 a b i = (a `ladd` b) `ladd` c
  where
    c = lbroadcastInDim v [2, 2]
    v = lit Tf32 [] ("Just[" ++ show i ++ "]")
